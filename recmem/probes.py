from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .common import DATASETS, RESULTS, proc_dir, read_jsonl, write_jsonl
from .llm import Mock, OpenAICompatible
from .run import default_split

YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def title_hit(guess: str, title: str) -> bool:
    g, t = normalise(guess), normalise(title)
    if not g or g in ("unknown", "i don t know"):
        return False
    main = normalise(re.split(r"\s+-\s+|\s*\|\s*|:\s|\s\(", title)[0])
    if max(SequenceMatcher(None, g, t).ratio(), SequenceMatcher(None, g, main).ratio()) >= 0.8:
        return True
    return (g in t and len(g) >= 0.5 * len(t)) or (t in g and len(t) >= 0.5 * len(g))


def reid_messages(desc: str, category: str, noun: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"An online catalogue of {noun} contains an item described only as:\n"
        f"\"{desc}\" (category: {category})\n"
        f"Which specific item is it? Reply with its exact title only, or \"unknown\" if you cannot tell."
    )}]


def know_messages(title: str, category: str, noun: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"In which year was this item from a catalogue of {noun} released (or first sold)?\n"
        f"\"{title}\" (category: {category})\n"
        f"Reply with the four-digit year only."
    )}]


def title_prefix(title: str) -> tuple[str, str] | None:
    words = title.split()
    if len(words) < 3:
        return None
    k = len(words) // 2
    return " ".join(words[:k]), " ".join(words[k:])


def complete_messages(prefix: str, category: str, noun: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"The title of an item from a catalogue of {noun} (category: {category}) begins with:\n"
        f"\"{prefix}\"\n"
        f"Reply with the item's complete exact title only."
    )}]


def completion_score(guess: str, prefix: str, rest: str) -> float:
    g, p, r = normalise(guess), normalise(prefix), normalise(rest)
    if g in ("unknown", "i don t know"):
        return 0.0
    if g.startswith(p):
        g = g[len(p):].strip()
    return SequenceMatcher(None, g, r).ratio() if g and r else 0.0


def run_all(backend, jobs: list[dict], workers: int, max_tokens: int = 48) -> tuple[list[dict], int]:
    def call(job):
        try:
            return job, backend.generate(job["messages"], max_tokens), None
        except Exception as exc:
            return job, None, exc

    rows, failed = [], 0
    with ThreadPoolExecutor(workers) as pool:
        for job, result, exc in pool.map(call, jobs):
            if exc is not None:
                failed += 1
                print(f"  FAILED item {job['item_id']}: {exc}", flush=True)
                continue
            rows.append({k: v for k, v in job.items() if k != "messages"} | {"response": result["response"]})
    return rows, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--eval-set", type=Path, default=None, help="defaults to the newest main test set")
    ap.add_argument("--views-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-disable-thinking", action="store_true")
    ap.add_argument("--kinds", default="reid,know,complete", help="comma-separated subset of reid,know,complete")
    args = ap.parse_args()
    kinds = set(args.kinds.split(","))

    out = proc_dir(args.dataset)
    eval_path = args.eval_set or default_split(out, "test")
    targets = sorted({row["target"] for row in read_jsonl(eval_path)})
    views = pd.read_parquet(out / f"views_s{args.views_seed}.parquet").set_index("item_id")
    years = pd.read_parquet(out / "items.parquet").set_index("item_id")["year"]
    noun = DATASETS[args.dataset]

    if args.backend == "mock":
        backend = Mock()
    else:
        if not args.model:
            raise SystemExit("--model is required for the openai backend")
        backend = OpenAICompatible(args.base_url, args.model, disable_thinking=not args.no_disable_thinking)

    reid_jobs, know_jobs, comp_jobs = [], [], []
    for item in targets:
        v = views.loc[item]
        desc = v["described"].split(": ", 1)[1] if ": " in v["described"] else v["described"]
        base = {"item_id": item, "title": v["real"], "category": v["category"], "popularity": int(v["popularity"])}
        reid_jobs.append(base | {"description": desc, "messages": reid_messages(desc, v["category"], noun)})
        if pd.notna(years.get(item)):
            know_jobs.append(base | {"year": int(years[item]), "messages": know_messages(v["real"], v["category"], noun)})
        split = title_prefix(v["real"])
        if split:
            comp_jobs.append(base | {"prefix": split[0], "rest": split[1],
                                     "messages": complete_messages(split[0], v["category"], noun)})
    reid_jobs, know_jobs = (reid_jobs if "reid" in kinds else []), (know_jobs if "know" in kinds else [])
    comp_jobs = comp_jobs if "complete" in kinds else []

    if args.backend != "mock":
        try:
            backend.generate((reid_jobs or know_jobs or comp_jobs)[0]["messages"], 48)
        except Exception as exc:
            raise SystemExit(f"Probe request to {args.base_url} (model {args.model}) failed: {exc}")

    print(f"[{args.tag}] {args.dataset}: probing {len(reid_jobs):,} items (reid), {len(know_jobs):,} (know), "
          f"{len(comp_jobs):,} (complete)", flush=True)
    reid, f1 = run_all(backend, reid_jobs, args.workers)
    know, f2 = run_all(backend, know_jobs, args.workers)
    comp, f3 = run_all(backend, comp_jobs, args.workers, max_tokens=96)
    for row in reid:
        row["hit"] = title_hit(row["response"], row["title"])
    for row in know:
        match = YEAR_RE.search(row["response"])
        guess = int(match.group(1)) if match else None
        row.update(guess_year=guess, exact=guess == row["year"],
                   within1=guess is not None and abs(guess - row["year"]) <= 1)
    for row in comp:
        row["score"] = completion_score(row["response"], row["prefix"], row["rest"])
        row["hit"] = row["score"] >= 0.8

    pdir = RESULTS / "probes" / args.dataset / eval_path.stem
    summary = {"tag": args.tag}
    if "reid" in kinds:
        write_jsonl(pdir / f"{args.tag}_reid.jsonl", reid)
        summary.update(n_items=len(reid),
                       reid_hit_rate=float(pd.Series([r["hit"] for r in reid]).mean()) if reid else float("nan"))
    if "know" in kinds:
        write_jsonl(pdir / f"{args.tag}_know.jsonl", know)
        summary.update(
            n_items_with_year=len(know),
            know_year_exact=float(pd.Series([r["exact"] for r in know]).mean()) if know else float("nan"),
            know_year_within1=float(pd.Series([r["within1"] for r in know]).mean()) if know else float("nan"))
    if "complete" in kinds:
        write_jsonl(pdir / f"{args.tag}_complete.jsonl", comp)
        summary.update(
            n_items_complete=len(comp),
            complete_hit_rate=float(pd.Series([r["hit"] for r in comp]).mean()) if comp else float("nan"),
            complete_mean_score=float(pd.Series([r["score"] for r in comp]).mean()) if comp else float("nan"))
    spath = RESULTS / "summary" / f"{args.dataset}_{eval_path.stem}_probes.csv"
    spath.parent.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(spath) if spath.exists() else pd.DataFrame({"tag": []})
    old = table[table["tag"] == args.tag].to_dict("records")
    row = (old[0] if old else {}) | summary
    table = pd.concat([table[table["tag"] != args.tag], pd.DataFrame([row])])
    table.to_csv(spath, index=False)
    print(f"[{args.tag}] {args.dataset} probes: " + ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items() if k != "tag"))

    if f1 + f2 + f3:
        sys.exit(f"[{args.tag}] {f1 + f2 + f3} probe requests failed; rerun to retry.")


if __name__ == "__main__":
    main()
