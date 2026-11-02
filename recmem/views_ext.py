from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from .common import DATA_RAW, proc_dir
from .conditions import category_vocabulary, distinctive_title_tokens, in_vocab, title_leak_share
from .prepare import AMAZON_CATEGORIES
from .prompts import MAX_NAME_CHARS

WORD_RE = re.compile(r"[a-z][a-z0-9'&+.-]*")
RARE_SHARE = 0.005
BRAND_STOPWORDS = {"the", "and", "of", "co", "inc", "llc", "ltd", "company", "corp", "corporation", "by", "for",
                   "brand", "brands", "products", "international", "group", "usa", "us"}


def display(name: str) -> str:
    return name if len(name) <= MAX_NAME_CHARS else name[: MAX_NAME_CHARS - 3] + "..."


def amazon_brands(dataset: str) -> dict[str, set[str]]:
    cat = AMAZON_CATEGORIES[dataset]
    path = next(p for p in sorted((DATA_RAW / "amazon").glob(f"meta_{cat}.filtered*.jsonl")))
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            details = m.get("details") or {}
            names = [str(details.get(k) or "") for k in ("Brand", "Manufacturer")] + [str(m.get("store") or "")]
            words = {w for n in names for w in re.split(r"[^a-z0-9]+", n.lower())
                     if len(w) >= 2 and w not in BRAND_STOPWORDS}
            out[m["parent_asin"]] = words
    return out


def word_core(token: str) -> str:
    return token.lower().strip(".,;:!?()[]{}\"'|/*-–—")


def generic_title(title: str, brand: set[str], rare: set[str], vocab: set[str]) -> str:
    kept = []
    for token in title.split():
        core = word_core(token)
        parts = [p for p in re.split(r"[^a-z0-9]+", core) if p]
        if WORD_RE.fullmatch(core) or any(re.search(r"[a-z]", p) for p in parts):
            if any(p in brand for p in parts) or (core in rare and not in_vocab(core, vocab)):
                continue
        kept.append(token)
    return tidy(" ".join(kept))


STOP = r"(?:the|of|to|a|an|and|or|for|with|in|on|by|from|at|&|\+)"


def tidy(text: str) -> str:
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    if text.count("(") != text.count(")"):
        text = text.replace("(", " ").replace(")", " ")
    if text.count("[") != text.count("]"):
        text = text.replace("[", " ").replace("]", " ")
    sep = r"(?:\s[-–—|/]\s|[,;:]\s|\(|\)|^|$)"
    for _ in range(3):
        text = re.sub(rf"({sep})\s*(?:{STOP}\s+)+", r"\1", text, flags=re.I)
        text = re.sub(rf"(?:\s+{STOP})+(?=\s*(?:[-–—|/,;:)]|$))", "", text, flags=re.I)
    text = re.sub(r"\s+([,;:|)])", r"\1", text)
    text = re.sub(r"([,;:|\-–—/])(\s*[,;:|\-–—/])+", r"\1", text)
    text = re.sub(r"^[\s,;:|\-–—/&+]+|[\s,;:|\-–—/&+(]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    content_words = [w for w in re.findall(r"[A-Za-z][A-Za-z']+", text)
                     if not re.fullmatch(STOP, w, flags=re.I)]
    return text if content_words else ""


def build(dataset: str, seed: int) -> tuple[pd.DataFrame, dict]:
    out = proc_dir(dataset)
    items = pd.read_parquet(out / "items.parquet")
    views = pd.read_parquet(out / f"views_s{seed}.parquet").set_index("item_id").loc[items["item_id"]]
    brands = amazon_brands(dataset) if dataset in AMAZON_CATEGORIES else {}
    titles = items["title"].astype(str)
    words = titles.str.lower().map(lambda t: {word_core(w) for w in t.split()} - {""})
    doc_freq = Counter(w for ws in words for w in ws)
    rare = {w for w, n in doc_freq.items() if n < RARE_SHARE * len(titles)}
    vocab = category_vocabulary(items["category"], items["attributes"])
    own = [{word_core(w) for w in d.split(": ", 1)[-1].lower().split()} for d in views["described"]]

    content = [generic_title(t, brands.get(i, set()), rare, vocab | o)
               for i, t, o in zip(items["item_id"], titles, own)]
    masked = views["masked"].to_numpy()
    desc_attrs = [d.split(": ", 1)[1] if ": " in d else "" for d in views["described"]]
    df = pd.DataFrame({
        "item_id": items["item_id"].to_numpy(),
        "content": [f"{m}: {c}" if c else m for m, c in zip(masked, content)],
        "both": [f"{display(r)}: {display(m + ': ' + a)[len(m) + 2:]}" if a else display(r)
                 for r, m, a in zip(views["real"], masked, desc_attrs)],
    })
    distinctive = distinctive_title_tokens(titles, category_vocabulary(items["category"], items["attributes"]))
    kept_share = [len(c.split()) / max(1, len(t.split())) for c, t in zip(content, titles)]
    stats = {
        "n_items": len(df),
        "share_content_empty": float(np.mean([not c for c in content])),
        "median_share_title_words_kept": float(np.median(kept_share)),
        "median_content_chars": float(np.median([len(c) for c in content])),
        "share_content_with_distinctive_title_word": title_leak_share(content, distinctive),
        "share_items_with_brand": float(np.mean([bool(brands.get(i)) for i in items["item_id"]])) if brands else 0.0,
        "examples": [{"title": t, "content": c} for t, c in list(zip(titles, content))[:12]],
    }
    return df, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df, stats = build(args.dataset, args.seed)
    out = proc_dir(args.dataset)
    df.to_parquet(out / f"views_ext_s{args.seed}.parquet", index=False)
    (out / f"views_ext_s{args.seed}.stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"{args.dataset}: -> {out / f'views_ext_s{args.seed}.parquet'}")
    print(json.dumps({k: v for k, v in stats.items() if k != "examples"}, indent=2))
    for ex in stats["examples"][:6]:
        print(f"  {ex['title'][:90]!r}\n    -> {ex['content'][:90]!r}")


if __name__ == "__main__":
    main()
