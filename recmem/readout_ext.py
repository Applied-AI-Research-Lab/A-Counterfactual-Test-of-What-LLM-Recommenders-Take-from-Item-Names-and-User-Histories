from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import CONDITIONS, DATASETS, RESULTS, read_jsonl
from .prompts import labels
from .run import make_job
from .run_ext import EXT_CONDITIONS, ext_job, ext_views, history_donors

READOUT = RESULTS / "ext" / "readout"
ALL_CONDITIONS = (*CONDITIONS, *EXT_CONDITIONS)


def build_jobs(rows, views, cond, noun, history_len, orders, done=frozenset()) -> list[dict]:
    donors = history_donors(rows)
    jobs = []
    for i, row in enumerate(rows):
        for k in range(min(orders, len(row["candidates"]))):
            if (row["user_id"], k) in done:
                continue
            if cond in CONDITIONS:
                jobs.append(make_job(row, k, views, cond, noun, history_len))
            else:
                jobs.append(ext_job(rows, i, k, views, cond, noun, history_len, donors))
    return jobs


class Readout:

    def __init__(self, base_url: str, model: str, top_k: int, timeout: int = 300):
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.top_k, self.timeout = model, top_k, timeout
        self.ids: dict[int, dict[str, int]] = {}

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from None
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def label_ids(self, n: int) -> dict[str, int]:
        if n not in self.ids:
            out = {}
            for lab in labels(n):
                toks = self._post(self.root + "/tokenize", {"model": self.model, "prompt": lab,
                                                            "add_special_tokens": False})["tokens"]
                if len(toks) != 1:
                    raise SystemExit(f"label {lab!r} is not a single token: {toks}")
                out[lab] = toks[0]
            self.ids[n] = out
        return self.ids[n]

    def check_restricted(self, messages: list[dict], n: int) -> None:
        lab, tid = next(iter(self.label_ids(n).items()))
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 1, "seed": 0,
                "logprobs": True, "top_logprobs": 1, "allowed_token_ids": [tid],
                "chat_template_kwargs": {"enable_thinking": False}}
        top = self._post(self.url, body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        if top[0]["token"] != lab or abs(top[0]["logprob"]) > 1e-3:
            raise SystemExit(f"server returns unrestricted log-probabilities ({top[0]}); start it with "
                             f"--logprobs-mode processed_logprobs (scripts/rev_vllm_job2.sh)")

    def score(self, messages: list[dict], n: int) -> dict:
        ids = self.label_ids(n)
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 1, "seed": 0,
                "logprobs": True, "top_logprobs": n, "allowed_token_ids": list(ids.values()),
                "chat_template_kwargs": {"enable_thinking": False}}
        start = time.perf_counter()
        data = self._post(self.url, body)
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {e["token"]: e["logprob"] for e in top}
        if set(lp) != set(ids):
            raise SystemExit(f"server returned other tokens than the {n} labels ({sorted(lp)[:5]}...); start it "
                             f"with --logprobs-mode processed_logprobs")
        ranked = sorted(lp, key=lambda lab: (-lp[lab], lab))
        usage = data.get("usage") or {}
        return {"response": ", ".join(ranked), "latency_s": round(time.perf_counter() - start, 4),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": 0,
                "label_mass": round(sum(math.exp(v) for v in lp.values()), 6), "n_found": len(ranked),
                "top_token": ranked[0], "top_logprob": lp[ranked[0]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--conditions", default="all", help="comma-separated subset of C0-C7 names, in this order")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8210/v1")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out-root", type=Path, default=READOUT, help="for checks: write somewhere else")
    args = ap.parse_args()

    rows = read_jsonl(args.eval_set)[: args.limit]
    views = ext_views(args.dataset)
    noun = DATASETS[args.dataset]
    conditions = ALL_CONDITIONS if args.conditions == "all" else tuple(args.conditions.split(","))
    unknown = set(conditions) - set(ALL_CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    backend = Readout(args.base_url, args.model, args.top_k)
    probe = build_jobs(rows[:1], views, conditions[0], noun, args.history_len, 1)[0]
    backend.check_restricted(probe["messages"], len(probe["candidates"]))
    try:
        r = backend.score(probe["messages"], len(probe["candidates"]))
    except Exception as exc:
        raise SystemExit(f"Probe request to {args.base_url} (model {args.model}) failed: {exc}")
    print(f"Probe OK: {r['response'][:60]!r}, label mass {r['label_mass']:.3f}, top token {r['top_token']!r}",
          flush=True)

    total_failed = 0
    for cond in conditions:
        path = args.out_root / args.dataset / args.eval_set.stem / args.tag / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(x["user_id"], x["order_idx"]) for x in read_jsonl(path)} if path.exists() else set()
        jobs = build_jobs(rows, views, cond, noun, args.history_len, args.orders, done)
        print(f"[{args.tag}] {args.dataset}/{cond}: {len(jobs):,} to score ({len(done):,} already done)", flush=True)
        lock, failed = threading.Lock(), 0
        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(backend.score, j["messages"], len(j["candidates"])): j for j in jobs}
            for n, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    failed += 1
                    print(f"  FAILED user {job['user_id']} order {job['order_idx']}: {exc}", flush=True)
                    continue
                record = {k: v for k, v in job.items() if k not in ("messages", "max_tokens")}
                record.update(model=args.model, eval_set=args.eval_set.name, readout="vllm-labels",
                              **result)
                with lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    if n % 5000 == 0:
                        print(f"  {n:,}/{len(jobs):,}", flush=True)
        total_failed += failed
    if total_failed:
        sys.exit(f"[{args.tag}] {total_failed:,} requests failed; rerun to retry them.")


if __name__ == "__main__":
    main()
