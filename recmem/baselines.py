from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import RESULTS, proc_dir, read_jsonl, write_jsonl
from .evaluate import rank_metrics
from .prompts import labels
from .run import default_split

LAMBDAS = [10, 50, 100, 250, 500, 1000, 2500, 5000]


def train_matrix(inter: pd.DataFrame, drop_last: int, uidx: dict, iidx: dict) -> sp.csr_matrix:
    pos_from_end = inter.groupby("user_id").cumcount(ascending=False)
    kept = inter[pos_from_end >= drop_last]
    rows = kept["user_id"].map(uidx).to_numpy()
    cols = kept["item_id"].map(iidx).to_numpy()
    return sp.csr_matrix((np.ones(len(kept), dtype=np.float32), (rows, cols)), shape=(len(uidx), len(iidx)))


def ease_weights(X: sp.csr_matrix, lam: float) -> np.ndarray:
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = P / -np.diag(P)
    B[np.diag_indices_from(B)] = 0.0
    return B


def ease_scorer(X: sp.csr_matrix, B: np.ndarray, uidx: dict):
    return lambda u, c: np.asarray(X[uidx[u]] @ B[:, c]).ravel()


def score_rows(rows, scorer, iidx) -> list[list[int]]:
    ranks = []
    for row in rows:
        per_order = []
        for cands in row["candidates"]:
            s = scorer(row["user_id"], np.array([iidx[c] for c in cands]))
            order = np.argsort(-s, kind="stable")
            per_order.append(int(np.where(order == cands.index(row["target"]))[0][0]) + 1)
        ranks.append(per_order)
    return ranks


def mean_ndcg10(ranks: list[list[int]]) -> float:
    return float(np.mean([np.mean([rank_metrics(r)["ndcg@10"] for r in per]) for per in ranks]))


def to_records(rows, scorer, iidx, name: str, eval_set: str) -> list[dict]:
    out = []
    for row in rows:
        for k, cands in enumerate(row["candidates"]):
            s = scorer(row["user_id"], np.array([iidx[c] for c in cands]))
            labs = labels(len(cands))
            ranked = [labs[j] for j in np.argsort(-s, kind="stable")]
            out.append({
                "user_id": row["user_id"], "order_idx": k, "condition": "idonly",
                "candidates": cands, "target": row["target"],
                "target_label": labs[cands.index(row["target"])],
                "model": name, "eval_set": eval_set, "response": ", ".join(ranked),
                "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
            })
    return out


def full_catalogue_metrics(rows, X: sp.csr_matrix, score_users, uidx: dict, iidx: dict, k: int = 10) -> dict:
    users = np.array([uidx[r["user_id"]] for r in rows])
    targets = np.array([iidx[r["target"]] for r in rows])
    hits, ndcgs = [], []
    for start in range(0, len(rows), 256):
        u, t = users[start:start + 256], targets[start:start + 256]
        scores = score_users(u).astype(np.float64)
        seen = X[u].toarray() > 0
        seen[np.arange(len(t)), t] = False
        scores[seen] = -np.inf
        target_scores = scores[np.arange(len(t)), t]
        ranks = 1 + (scores > target_scores[:, None]).sum(axis=1)
        hits.extend(ranks <= k)
        ndcgs.extend(1 / math.log2(r + 1) if r <= k else 0.0 for r in ranks)
    return {f"hr@{k}": float(np.mean(hits)), f"ndcg@{k}": float(np.mean(ndcgs)), "n_users": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, default=None,
                    help="test set (default: newest main set); its valid_* twin is used for tuning")
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    inter = pd.read_parquet(out / "interactions.parquet").sort_values(["user_id", "ts"], kind="stable")
    items = pd.read_parquet(out / "items.parquet")
    uidx = {u: i for i, u in enumerate(inter["user_id"].unique())}
    iidx = {it: i for i, it in enumerate(items["item_id"])}
    test_path = args.eval_set or default_split(out, "test")
    valid_path = test_path.with_name(test_path.name.replace("test_", "valid_", 1))
    if not valid_path.exists():
        raise SystemExit(f"Validation twin {valid_path.name} not found next to {test_path.name}")
    valid, test = read_jsonl(valid_path), read_jsonl(test_path)

    Xv = train_matrix(inter, 2, uidx, iidx)
    tuning = {}
    for lam in LAMBDAS:
        B = ease_weights(Xv, lam)
        tuning[lam] = mean_ndcg10(score_rows(valid, ease_scorer(Xv, B, uidx), iidx))
        print(f"EASE lambda={lam:>5}: valid NDCG@10={tuning[lam]:.4f}", flush=True)
    best = max(tuning, key=tuning.get)

    Xt = train_matrix(inter, 1, uidx, iidx)
    B = ease_weights(Xt, best)
    pop = np.asarray(Xt.sum(axis=0)).ravel()
    scorers = {
        "ease": ease_scorer(Xt, B, uidx),
        "mostpop": lambda u, c: pop[c],
    }
    base = RESULTS / "raw" / args.dataset / test_path.stem
    for name, scorer in scorers.items():
        path = base / name / "idonly.jsonl"
        write_jsonl(path, to_records(test, scorer, iidx, name, test_path.name))
        print(f"{name}: wrote {path}")
    (base / "ease" / "tuning.json").write_text(
        json.dumps({"best_lambda": best, "valid_ndcg@10": tuning}, indent=2))

    full = {
        "ease": full_catalogue_metrics(test, Xt, lambda u: np.asarray(Xt[u] @ B), uidx, iidx),
        "mostpop": full_catalogue_metrics(test, Xt, lambda u: np.tile(pop, (len(u), 1)), uidx, iidx),
        "random_hr@10": 10 / len(iidx),
    }
    (base / "fullrank.json").write_text(json.dumps(full, indent=2))
    print(f"full-catalogue ranking ({len(iidx):,} items): {json.dumps(full)}")


if __name__ == "__main__":
    main()
