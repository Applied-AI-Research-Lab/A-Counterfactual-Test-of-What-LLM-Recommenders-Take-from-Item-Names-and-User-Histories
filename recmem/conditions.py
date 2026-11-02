from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from .common import proc_dir

MIN_STRATUM = 5
TOKEN_RE = re.compile(r"[a-z][a-z0-9']{4,}")


def distinctive_title_tokens(titles: pd.Series, content_vocab: set[str]) -> list[set[str]]:
    tokens = titles.str.lower().map(lambda t: set(TOKEN_RE.findall(t)))
    doc_freq = Counter(tok for toks in tokens for tok in toks)
    limit = 0.005 * len(titles)
    return [{tok for tok in toks if doc_freq[tok] < limit and not in_vocab(tok, content_vocab)} for toks in tokens]


def in_vocab(token: str, vocab: set[str]) -> bool:
    return token in vocab or token.rstrip("s") in vocab or f"{token}s" in vocab or f"{token}es" in vocab


def category_vocabulary(categories: pd.Series, attributes: pd.Series) -> set[str]:
    text = " ".join(categories.astype(str)) + " " + " ".join(a.split("; ")[0] for a in attributes)
    return set(TOKEN_RE.findall(text.lower()))


def strip_title_leaks(attributes: pd.Series, distinctive: list[set[str]]) -> list[str]:
    out = []
    for attr, toks in zip(attributes, distinctive):
        path, *fields = attr.split("; ")
        out.append("; ".join([path] + [f for f in fields if not any(t in f.lower() for t in toks)]))
    return out


def title_leak_share(descriptions, distinctive: list[set[str]]) -> float:
    return float(np.mean([any(t in d.lower() for t in toks) for d, toks in zip(descriptions, distinctive)]))


def derangement(n: int, rng) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    order = rng.permutation(n)
    donor = np.empty(n, dtype=int)
    donor[order] = np.roll(order, -1)
    return donor


def build_views(items: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df = items[["item_id", "title", "category", "popularity", "attributes"]].copy()
    df["decile"] = pd.qcut(df["popularity"].rank(method="first"), 10, labels=False)

    df["stratum"] = df["category"].astype(str) + "|" + df["decile"].astype(str)
    small = df.groupby("stratum")["item_id"].transform("size") < MIN_STRATUM
    df.loc[small, "stratum"] = df.loc[small, "category"].astype(str) + "|pooled"
    tiny = df.groupby("stratum")["item_id"].transform("size") < 2
    df.loc[tiny, "stratum"] = "__global__"

    df = df.reset_index(drop=True)
    donor = np.arange(len(df))
    for _, idx in df.groupby("stratum").indices.items():
        donor[idx] = idx[derangement(len(idx), rng)]
    df["swapped"] = df["title"].to_numpy()[donor]
    df["donor_item_id"] = df["item_id"].to_numpy()[donor]

    codes = rng.choice(np.arange(10_000, 100_000), size=len(df), replace=False)
    df["masked"] = [f"Item {c}" for c in codes]
    distinctive = distinctive_title_tokens(df["title"], category_vocabulary(df["category"], df["attributes"]))
    leak_before = title_leak_share(df["attributes"], distinctive)
    df["attributes"] = strip_title_leaks(df["attributes"], distinctive)
    df["described"] = df["masked"] + ": " + df["attributes"]
    df = df.rename(columns={"title": "real"})

    pop_pct = df["popularity"].rank(pct=True).to_numpy()
    stats = {
        "n_items": len(df),
        "share_title_changed": float((df["swapped"] != df["real"]).mean()),
        "share_same_category_swap": float((df["category"].to_numpy() == df["category"].to_numpy()[donor]).mean()),
        "share_pooled_strata": float(df["stratum"].str.endswith("|pooled").mean()),
        "median_abs_popularity_pct_shift": float(np.median(np.abs(pop_pct - pop_pct[donor]))),
        "share_described_unique": float(df["attributes"].nunique() / len(df)),
        "share_described_title_leak_before_filter": leak_before,
        "share_described_title_leak": title_leak_share(df["attributes"], distinctive),
        "median_described_chars": float(df["attributes"].str.len().median()),
    }
    cols = ["item_id", "category", "popularity", "real", "masked", "swapped", "described", "donor_item_id"]
    return df[cols], stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    views, stats = build_views(pd.read_parquet(out / "items.parquet"), args.seed)
    views.to_parquet(out / f"views_s{args.seed}.parquet", index=False)
    (out / f"views_s{args.seed}.stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{args.dataset}: views -> {out / f'views_s{args.seed}.parquet'}\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
