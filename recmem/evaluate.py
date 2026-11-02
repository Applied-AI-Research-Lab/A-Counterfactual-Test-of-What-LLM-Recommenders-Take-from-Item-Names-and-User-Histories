from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import RESULTS, proc_dir, read_jsonl
from .prompts import labels
from .run import default_split

METRICS = ("hr@1", "hr@5", "ndcg@5", "ndcg@10", "mrr")

CONTRASTS = [
    ("real", "described", "identity knowledge beyond content"),
    ("real", "masked", "total effect of the name"),
    ("real", "swapped", "dependence + interference"),
    ("masked", "swapped", "interference (misleading vs no identity)"),
    ("described", "masked", "value of the identity-free content"),
]
LIST_RE = re.compile(r"(?<![A-Za-z])[A-Z](?:\s*[,;>]\s*[A-Z](?![A-Za-z]))+")
NUMBERED_RE = re.compile(r"^\s*\d+\s*[.)]\s*([A-Z])(?![A-Za-z])", re.M)
NUM_LIST_RE = re.compile(r"(?<![\d.])\d{1,3}(?:\s*[,;>]\s*\d{1,3}(?![\d.]))+")


def parse_ranking(text: str, n: int) -> list[str]:
    text = re.sub(r"<think>.*?(</think>|$)", " ", text, flags=re.S)
    valid = labels(n)
    if valid[0] == "1":
        lists = NUM_LIST_RE.findall(text)
        if lists:
            letters = re.findall(r"\d{1,3}", max(lists, key=len))
        elif re.fullmatch(r"\s*\d{1,3}\s*\.?\s*", text):
            letters = [text.strip().rstrip(".")]
        else:
            letters = []
    else:
        lists = LIST_RE.findall(text)
        if lists:
            letters = re.findall(r"[A-Z]", max(lists, key=len))
        elif NUMBERED_RE.search(text):
            letters = NUMBERED_RE.findall(text)
        elif re.fullmatch(r"\s*([A-Z])\s*\.?\s*", text):
            letters = [text.strip()[0]]
        else:
            letters = []
    valid, seen, out = set(valid), set(), []
    for letter in letters:
        if letter in valid and letter not in seen:
            seen.add(letter)
            out.append(letter)
    return out


def rank_metrics(rank: int) -> dict:
    return {
        "hr@1": float(rank <= 1),
        "hr@5": float(rank <= 5),
        "ndcg@5": 1 / math.log2(rank + 1) if rank <= 5 else 0.0,
        "ndcg@10": 1 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "mrr": 1 / rank,
    }


def expected_metrics(ranks: range) -> dict:
    per = [rank_metrics(r) for r in ranks]
    return {m: float(np.mean([p[m] for p in per])) for m in METRICS}


def record_metrics(response: str, target_label: str, n: int) -> dict:
    parsed = parse_ranking(response, n)
    if target_label in parsed:
        metrics = rank_metrics(parsed.index(target_label) + 1)
    else:
        metrics = expected_metrics(range(len(parsed) + 1, n + 1))
    metrics.update(n_parsed=len(parsed), no_letters=float(not parsed), complete=float(len(parsed) == n))
    return metrics


def load_records(base: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(base.glob("*/*.jsonl")):
        for rec in read_jsonl(path):
            n = len(rec["candidates"])
            rows.append({"tag": path.parent.name, "condition": rec["condition"], "user_id": rec["user_id"],
                         "order_idx": rec["order_idx"], "n_cand": n,
                         **record_metrics(rec["response"], rec["target_label"], n)})
    if not rows:
        raise SystemExit(f"No results under {base}")
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ratio_ci(real: np.ndarray, other: np.ndarray, rand: float,
                       n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(real), size=(n_boot, len(real)))
    ratios = (real[idx] - other[idx]).mean(axis=1) / (real[idx].mean(axis=1) - rand)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted, running = np.empty(len(pvals)), 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(pvals) - i) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", default="ndcg@10", choices=METRICS, help="metric for contrasts")
    ap.add_argument("--eval-set", type=Path, default=None, help="defaults to the newest test_*.jsonl")
    ap.add_argument("--min-headroom", type=float, default=0.05,
                    help="report KDI only if metric(real) - metric(random) is at least this (default 0.05)")
    args = ap.parse_args()

    eval_path = args.eval_set or default_split(proc_dir(args.dataset), "test")
    recs = load_records(RESULTS / "raw" / args.dataset / eval_path.stem)
    n_cand = int(recs["n_cand"].mode()[0])
    random_ref = expected_metrics(range(1, n_cand + 1))
    per_user = recs.groupby(["tag", "condition", "user_id"])[[*METRICS, "no_letters", "complete"]].mean().reset_index()

    summary = []
    for (tag, cond), g in per_user.groupby(["tag", "condition"]):
        lo, hi = bootstrap_ci(g[args.metric].to_numpy())
        summary.append({"tag": tag, "condition": cond, "n_users": len(g),
                        **{m: g[m].mean() for m in METRICS},
                        f"{args.metric}_ci_lo": lo, f"{args.metric}_ci_hi": hi,
                        "no_letters_rate": g["no_letters"].mean(), "complete_rate": g["complete"].mean()})
    summary = pd.DataFrame(summary)
    summary.loc[len(summary)] = {"tag": "random", "condition": "-", "n_users": None, **random_ref}

    contrasts = []
    rand = random_ref[args.metric]
    wide = per_user.pivot_table(index=["tag", "user_id"], columns="condition", values=args.metric)
    for tag, g in wide.groupby(level="tag"):
        for first, second, meaning in CONTRASTS:
            if first not in g or second not in g:
                continue
            pair = g[[first, second]].dropna()
            if pair.empty:
                continue
            a, b = pair[first].to_numpy(), pair[second].to_numpy()
            diff = a - b
            lo, hi = bootstrap_ci(diff)
            row = {"tag": tag, "contrast": f"{first}-{second}", "meaning": meaning, "n_users": len(pair),
                   "first": a.mean(), "second": b.mean(), "delta": diff.mean(),
                   "delta_ci_lo": lo, "delta_ci_hi": hi,
                   "headroom": a.mean() - rand if first == "real" else float("nan"),
                   "KDI": float("nan"), "KDI_ci_lo": float("nan"), "KDI_ci_hi": float("nan"),
                   "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0}
            if first == "real" and row["headroom"] >= args.min_headroom:
                row["KDI"] = diff.mean() / row["headroom"]
                row["KDI_ci_lo"], row["KDI_ci_hi"] = bootstrap_ratio_ci(a, b, rand)
            contrasts.append(row)
    contrasts = pd.DataFrame(contrasts)
    if len(contrasts):
        contrasts["p_holm"] = holm(contrasts["p_wilcoxon"].tolist())

    out = RESULTS / "summary"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{eval_path.stem}"
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    contrasts.to_csv(out / f"{stem}_contrasts.csv", index=False)
    per_user.to_parquet(out / f"{stem}_per_user.parquet", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n== {args.dataset} ({eval_path.stem}): summary (random-ranking reference included)")
    print(summary.round(4).to_string(index=False))
    if len(contrasts):
        print(f"\n== {args.dataset}: contrasts on {args.metric}")
        print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
