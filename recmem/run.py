from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .common import CONDITIONS, DATASETS, RESULTS, proc_dir, read_jsonl
from .llm import Mock, OpenAICompatible
from .prompts import answer_tokens, build_messages, labels


def default_split(out: Path, split: str) -> Path:
    main_set = re.compile(rf"{split}_n\d+_c20_s\d+\.jsonl")
    files = sorted((p for p in out.glob(f"{split}_*.jsonl") if main_set.fullmatch(p.name)),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No main {split} set in {out}. Run recmem.split first.")
    print(f"Using {split} set {files[-1].name}", flush=True)
    return files[-1]


def make_job(row: dict, order_idx: int, views: dict, condition: str, noun: str, history_len: int,
             merge_system: bool = True, variant: int = 1):
    names, cats = views[condition], views["category"]
    hist = [(names[i], cats[i]) for i in row["history"][-history_len:]]
    cand_ids = row["candidates"][order_idx]
    cands = [(names[i], cats[i]) for i in cand_ids]
    return {
        "user_id": row["user_id"], "order_idx": order_idx, "condition": condition,
        "candidates": cand_ids, "target": row["target"],
        "target_label": labels(len(cand_ids))[cand_ids.index(row["target"])],
        "prompt_variant": variant,
        "max_tokens": answer_tokens(len(cand_ids)),
        "messages": build_messages(hist, cands, noun, merge_system, variant),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True, help="short model name used in result paths")
    ap.add_argument("--condition", default="all", choices=["all", *CONDITIONS])
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--model", default=None, help="model id served by the endpoint")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--eval-set", type=Path, default=None, help="defaults to newest <split>_*.jsonl")
    ap.add_argument("--views-seed", type=int, default=42)
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--orders", type=int, default=3, help="how many fixed candidate orders to run")
    ap.add_argument("--limit", type=int, default=None, help="only the first N users")
    ap.add_argument("--workers", type=int, default=32, help="concurrent requests")
    ap.add_argument("--no-disable-thinking", action="store_true")
    ap.add_argument("--system-role", action="store_true",
                    help="send the instruction as a separate system message (default: merged into the user "
                         "turn, identical for every model and safe for templates without a system role)")
    ap.add_argument("--prompt-variant", type=int, default=1, choices=[1, 2],
                    help="2 = alternative wording (prompt-sensitivity check), stored under <tag>-p2")
    ap.add_argument("--dump-prompts", type=int, default=0, help="print N prompts per condition and exit")
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    eval_path = args.eval_set or default_split(out, args.split)
    rows = read_jsonl(eval_path)[: args.limit]
    views_df = pd.read_parquet(out / f"views_s{args.views_seed}.parquet").set_index("item_id")
    views = {col: views_df[col].to_dict() for col in ("category", *CONDITIONS)}
    noun = DATASETS[args.dataset]
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)

    if args.dump_prompts:
        for cond in conditions:
            for row in rows[: args.dump_prompts]:
                job = make_job(row, 0, views, cond, noun, args.history_len, not args.system_role, args.prompt_variant)
                print(f"===== {cond} | user {job['user_id']} | target {job['target_label']}")
                print(job["messages"][-1]["content"], "\n")
        return

    if args.backend == "mock":
        backend = Mock()
    else:
        if not args.model:
            raise SystemExit("--model is required for the openai backend")
        backend = OpenAICompatible(args.base_url, args.model, disable_thinking=not args.no_disable_thinking)
        probe = make_job(rows[0], 0, views, conditions[0], noun, args.history_len, not args.system_role,
                         args.prompt_variant)
        try:
            reply = backend.generate(probe["messages"], probe["max_tokens"])
        except Exception as exc:
            raise SystemExit(f"Probe request to {args.base_url} (model {args.model}) failed: {exc}")
        print(f"Probe OK ({reply['latency_s']}s): {reply['response'][:80]!r}", flush=True)

    tag = args.tag if args.prompt_variant == 1 else f"{args.tag}-p{args.prompt_variant}"
    total_failed = 0
    for cond in conditions:
        path = RESULTS / "raw" / args.dataset / eval_path.stem / tag / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["user_id"], r["order_idx"]) for r in read_jsonl(path)} if path.exists() else set()
        jobs = [make_job(row, k, views, cond, noun, args.history_len, not args.system_role, args.prompt_variant)
                for row in rows for k in range(min(args.orders, len(row["candidates"])))
                if (row["user_id"], k) not in done]
        print(f"[{tag}] {args.dataset}/{cond}: {len(jobs):,} to run ({len(done):,} already done)", flush=True)

        lock = threading.Lock()
        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(backend.generate, job["messages"], job["max_tokens"]): job for job in jobs}
            failed = 0
            for n, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    failed += 1
                    print(f"  FAILED user {job['user_id']} order {job['order_idx']}: {exc}", flush=True)
                    continue
                record = {k: v for k, v in job.items() if k not in ("messages", "max_tokens")}
                record.update(model=args.model or args.backend, eval_set=eval_path.name, **result)
                with lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if n % 500 == 0:
                        fh.flush()
                        print(f"  {n:,}/{len(jobs):,}", flush=True)
        total_failed += failed
        if failed:
            print(f"  {failed:,} requests failed; rerun the same command to retry them.", flush=True)

    if total_failed:
        sys.exit(f"[{tag}] {total_failed:,} requests failed in total; rerun to retry them.")


if __name__ == "__main__":
    main()
