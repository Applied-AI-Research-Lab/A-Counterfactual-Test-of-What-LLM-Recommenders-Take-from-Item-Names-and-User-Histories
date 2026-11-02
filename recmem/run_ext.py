from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CONDITIONS, DATASETS, RESULTS, proc_dir, read_jsonl
from .conditions import derangement
from .llm import Mock, OpenAICompatible
from .prompts import SYSTEMS, labels
from .run import make_job

EXT_CONDITIONS = ("content", "both", "realshuf", "maskshuf")
NAMES_OF = {"content": "content", "both": "both", "realshuf": "real", "maskshuf": "masked"}
EXT_RESULTS = RESULTS / "ext" / "raw"


def ext_views(dataset: str, seed: int = 42) -> dict:
    out = proc_dir(dataset)
    v = pd.read_parquet(out / f"views_s{seed}.parquet").set_index("item_id")
    e = pd.read_parquet(out / f"views_ext_s{seed}.parquet").set_index("item_id")
    views = {col: v[col].to_dict() for col in ("category", *CONDITIONS)}
    views.update({col: e[col].to_dict() for col in ("content", "both")})
    return views


def history_donors(rows: list[dict], seed: int = 42) -> list[int]:
    return list(derangement(len(rows), np.random.default_rng(seed)))


def build_messages_untruncated(history, candidates, noun: str) -> list[dict]:
    n, labs = len(candidates), labels(len(candidates))
    hist = "\n".join(f"- {name} [{cat}]" for name, cat in history)
    cands = "\n".join(f"{lab}. {name} [{cat}]" for lab, (name, cat) in zip(labs, candidates))
    user = (
        f"A user interacted with the following {noun}, in chronological order (oldest first):\n"
        f"{hist}\n\n"
        f"Rank these {n} candidate {noun} by how likely the user is to interact with each one next:\n"
        f"{cands}\n\n"
        f"Answer with all {n} labels ({labs[0]}-{labs[-1]}) ranked from most to least likely, "
        f"separated by commas, and nothing else."
    )
    return [{"role": "user", "content": f"{SYSTEMS[1]}\n\n{user}"}]


def ext_job(rows: list[dict], i: int, order_idx: int, views: dict, condition: str, noun: str, history_len: int,
            donors: list[int]) -> dict:
    row = rows[i]
    if condition in ("realshuf", "maskshuf"):
        row = {**row, "history": rows[donors[i]]["history"]}
    job = make_job(row, order_idx, views, NAMES_OF[condition], noun, history_len)
    job["condition"] = condition
    if condition in ("realshuf", "maskshuf"):
        job["history_from"] = rows[donors[i]]["user_id"]
    if condition == "both":
        names, cats = views["both"], views["category"]
        hist = [(names[x], cats[x]) for x in row["history"][-history_len:]]
        cands = [(names[x], cats[x]) for x in row["candidates"][order_idx]]
        job["messages"] = build_messages_untruncated(hist, cands, noun)
    return job


def ext_jobs(rows, views, condition, noun, history_len, orders, done=frozenset()) -> list[dict]:
    donors = history_donors(rows)
    return [ext_job(rows, i, k, views, condition, noun, history_len, donors)
            for i, row in enumerate(rows) for k in range(min(orders, len(row["candidates"])))
            if (row["user_id"], k) not in done]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--condition", default="all", choices=["all", *EXT_CONDITIONS])
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8210/v1")
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--dump-prompts", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.eval_set)[: args.limit]
    views = ext_views(args.dataset)
    noun = DATASETS[args.dataset]
    conditions = EXT_CONDITIONS if args.condition == "all" else (args.condition,)

    if args.dump_prompts:
        for cond in conditions:
            for job in ext_jobs(rows[: args.dump_prompts], views, cond, noun, args.history_len, 1):
                print(f"===== {cond} | user {job['user_id']} | target {job['target_label']}")
                print(job["messages"][-1]["content"], "\n")
        return

    if args.backend == "mock":
        backend = Mock()
    else:
        backend = OpenAICompatible(args.base_url, args.model)
        probe = ext_jobs(rows[:1], views, conditions[0], noun, args.history_len, 1)[0]
        try:
            reply = backend.generate(probe["messages"], probe["max_tokens"])
        except Exception as exc:
            raise SystemExit(f"Probe request to {args.base_url} (model {args.model}) failed: {exc}")
        print(f"Probe OK ({reply['latency_s']}s): {reply['response'][:80]!r}", flush=True)

    total_failed = 0
    for cond in conditions:
        path = EXT_RESULTS / args.dataset / args.eval_set.stem / args.tag / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["user_id"], r["order_idx"]) for r in read_jsonl(path)} if path.exists() else set()
        jobs = ext_jobs(rows, views, cond, noun, args.history_len, args.orders, done)
        print(f"[{args.tag}] {args.dataset}/{cond}: {len(jobs):,} to run ({len(done):,} already done)", flush=True)
        lock, failed = threading.Lock(), 0
        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(backend.generate, j["messages"], j["max_tokens"]): j for j in jobs}
            for n, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    failed += 1
                    print(f"  FAILED user {job['user_id']} order {job['order_idx']}: {exc}", flush=True)
                    continue
                record = {k: v for k, v in job.items() if k not in ("messages", "max_tokens")}
                record.update(model=args.model or args.backend, eval_set=args.eval_set.name, **result)
                with lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    if n % 1000 == 0:
                        print(f"  {n:,}/{len(jobs):,}", flush=True)
        total_failed += failed
    if total_failed:
        sys.exit(f"[{args.tag}] {total_failed:,} requests failed; rerun to retry them.")


if __name__ == "__main__":
    main()
