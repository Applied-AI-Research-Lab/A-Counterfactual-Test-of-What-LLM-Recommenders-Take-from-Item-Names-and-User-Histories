from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import proc_dir, write_jsonl


def sample_negatives(rng, item_ids, probs, exclude: set, k: int) -> list:
    chosen: list = []
    while len(chosen) < k:
        draw = rng.choice(len(item_ids), size=2 * k, replace=False, p=probs)
        for idx in draw:
            item = item_ids[idx]
            if item not in exclude and item not in chosen:
                chosen.append(item)
                if len(chosen) == k:
                    break
    return chosen


def build(seqs: pd.Series, users, offset: int, item_ids, probs, n_cand, n_orders, rng) -> list[dict]:
    rows = []
    for user in users:
        seq = seqs[user]
        target, history = seq[-offset], seq[:-offset]
        negs = sample_negatives(rng, item_ids, probs, set(seq), n_cand - 1)
        cands = [target] + negs
        orders = [list(rng.permutation(cands)) for _ in range(n_orders)]
        rows.append({"user_id": user, "history": list(history), "target": target, "candidates": orders})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-users", type=int, default=2500)
    ap.add_argument("--n-candidates", type=int, default=20)
    ap.add_argument("--n-orders", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0, help="negative sampling ~ popularity**alpha (0 = uniform)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    inter = pd.read_parquet(out / "interactions.parquet")
    items = pd.read_parquet(out / "items.parquet")
    seqs = inter.sort_values(["user_id", "ts"], kind="stable").groupby("user_id")["item_id"].apply(list)

    rng = np.random.default_rng(args.seed)
    users = sorted(seqs.index)
    users = list(rng.choice(users, size=min(args.n_users, len(users)), replace=False))

    item_ids = items["item_id"].to_numpy()
    weights = items["popularity"].to_numpy(dtype=float) ** args.alpha
    probs = weights / weights.sum()

    tag = f"n{args.n_users}_c{args.n_candidates}_s{args.seed}" + (f"_a{args.alpha:g}" if args.alpha != 1.0 else "")
    for name, offset in (("test", 1), ("valid", 2)):
        rows = build(seqs, users, offset, item_ids, probs, args.n_candidates, args.n_orders, rng)
        write_jsonl(out / f"{name}_{tag}.jsonl", rows)
        print(f"{args.dataset}: wrote {len(rows):,} {name} users -> {out / f'{name}_{tag}.jsonl'}")


if __name__ == "__main__":
    main()
