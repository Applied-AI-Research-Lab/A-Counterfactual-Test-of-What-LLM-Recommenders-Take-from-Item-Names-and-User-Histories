from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .common import DATASETS, RESULTS, read_jsonl, write_jsonl
from .llm import Mock, OpenAICompatible
from .probes import normalise, reid_messages, run_all, title_hit
from .run_ext import ext_views


def strict_hit(response: str, title: str, content: str) -> bool:
    shown = set(normalise(content).split())
    removed = {w for w in normalise(title).split() if w not in shown and len(w) >= 3}
    return title_hit(response, title) and bool(removed & set(normalise(response).split()))


def summarise(dataset: str, eval_path: Path, tag: str, rows: list[dict], n_targets: int) -> dict:
    summary = {"tag": tag, "n_items": n_targets, "n_items_content": len(rows),
               "reid_content_hit_rate": float(pd.Series([r["hit"] for r in rows]).mean()) if rows else float("nan"),
               "reid_content_strict_rate": float(pd.Series([r["hit_strict"] for r in rows]).mean()) if rows else float("nan")}
    spath = RESULTS / "ext" / "summary" / f"{dataset}_{eval_path.stem}_probes_ext.csv"
    spath.parent.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(spath) if spath.exists() else pd.DataFrame({"tag": []})
    table = pd.concat([table[table["tag"] != tag], pd.DataFrame([summary])])
    table.to_csv(spath, index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8210/v1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rescore", action="store_true", help="recompute hits from the saved answers (no model calls)")
    args = ap.parse_args()
    out_path = RESULTS / "ext" / "probes" / args.dataset / args.eval_set.stem / f"{args.tag}_reid_content.jsonl"
    if args.rescore:
        rows = read_jsonl(out_path)
        for row in rows:
            row["hit"] = title_hit(row["response"], row["title"])
            row["hit_strict"] = strict_hit(row["response"], row["title"], row["content"])
        write_jsonl(out_path, rows)
        n_targets = len({r["target"] for r in read_jsonl(args.eval_set)})
        s = summarise(args.dataset, args.eval_set, args.tag, rows, n_targets)
        print(f"[{args.tag}] {args.dataset} reid_content: lenient {s['reid_content_hit_rate']:.3f}, "
              f"strict {s['reid_content_strict_rate']:.3f}")
        return

    views = ext_views(args.dataset)
    noun = DATASETS[args.dataset]
    targets = sorted({row["target"] for row in read_jsonl(args.eval_set)})
    jobs = []
    for item in targets:
        text = views["content"][item].split(": ", 1)[1] if ": " in views["content"][item] else ""
        if text:
            jobs.append({"item_id": item, "title": views["real"][item], "category": views["category"][item],
                         "content": text, "messages": reid_messages(text, views["category"][item], noun)})
    backend = Mock() if args.backend == "mock" else OpenAICompatible(args.base_url, args.model)
    print(f"[{args.tag}] {args.dataset}: reid from the content view, {len(jobs):,} of {len(targets):,} items",
          flush=True)
    rows, failed = run_all(backend, jobs, args.workers)
    for row in rows:
        row["hit"] = title_hit(row["response"], row["title"])
        row["hit_strict"] = strict_hit(row["response"], row["title"], row["content"])
    write_jsonl(out_path, rows)
    s = summarise(args.dataset, args.eval_set, args.tag, rows, len(targets))
    print(f"[{args.tag}] {args.dataset} reid_content: lenient {s['reid_content_hit_rate']:.3f}, "
          f"strict {s['reid_content_strict_rate']:.3f} ({s['n_items_content']:,} items)")
    if failed:
        sys.exit(f"[{args.tag}] {failed} probe requests failed; rerun to retry.")


if __name__ == "__main__":
    main()
