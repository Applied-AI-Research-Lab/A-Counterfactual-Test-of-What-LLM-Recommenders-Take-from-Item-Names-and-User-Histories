from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import RESULTS
from .evaluate import CONTRASTS, METRICS, bootstrap_ci, bootstrap_ratio_ci, expected_metrics, holm, load_records

EXT_CONTRASTS = [
    ("real", "content", "identity (brand + name) beyond the title's generic words"),
    ("content", "masked", "generic content of the title"),
    ("content", "described", "title content vs metadata description"),
    ("both", "described", "name on top of the description"),
    ("both", "real", "description on top of the name"),
    ("real", "realshuf", "use of the user's history (names)"),
    ("masked", "maskshuf", "use of the user's history (codes only)"),
]
SEED_TAG = re.compile(r"^(?P<config>.+-lora-(?:real|masked))-s(?P<seed>\d+)$")


def load_both(dataset: str, stem: str) -> pd.DataFrame:
    parts = []
    for base in (RESULTS / "raw" / dataset / stem, RESULTS / "ext" / "raw" / dataset / stem):
        if base.exists() and any(base.glob("*/*.jsonl")):
            parts.append(load_records(base))
    recs = pd.concat(parts, ignore_index=True)
    dup = recs.duplicated(["tag", "condition", "user_id", "order_idx"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate records across results/raw and results/ext/raw")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--metric", default="ndcg@10", choices=METRICS)
    ap.add_argument("--min-headroom", type=float, default=0.05)
    args = ap.parse_args()

    stem = args.eval_set.stem
    recs = load_both(args.dataset, stem)
    n_cand = int(recs["n_cand"].mode()[0])
    random_ref = expected_metrics(range(1, n_cand + 1))
    per_user = recs.groupby(["tag", "condition", "user_id"])[[*METRICS, "no_letters", "complete"]].mean().reset_index()

    summary = []
    for (tag, cond), g in per_user.groupby(["tag", "condition"]):
        lo, hi = bootstrap_ci(g[args.metric].to_numpy())
        summary.append({"tag": tag, "condition": cond, "n_users": len(g), **{m: g[m].mean() for m in METRICS},
                        f"{args.metric}_ci_lo": lo, f"{args.metric}_ci_hi": hi,
                        "no_letters_rate": g["no_letters"].mean(), "complete_rate": g["complete"].mean()})
    summary = pd.DataFrame(summary)
    summary.loc[len(summary)] = {"tag": "random", "condition": "-", "n_users": None, **random_ref}

    rand = random_ref[args.metric]
    wide = per_user.pivot_table(index=["tag", "user_id"], columns="condition", values=args.metric)
    contrasts = []
    for tag, g in wide.groupby(level="tag"):
        for first, second, meaning in CONTRASTS + EXT_CONTRASTS:
            if first not in g or second not in g:
                continue
            pair = g[[first, second]].dropna()
            if pair.empty:
                continue
            a, b = pair[first].to_numpy(), pair[second].to_numpy()
            diff = a - b
            lo, hi = bootstrap_ci(diff)
            row = {"tag": tag, "contrast": f"{first}-{second}", "meaning": meaning, "n_users": len(pair),
                   "first": a.mean(), "second": b.mean(), "delta": diff.mean(), "delta_ci_lo": lo, "delta_ci_hi": hi,
                   "headroom": a.mean() - rand if first == "real" else float("nan"),
                   "KDI": float("nan"), "KDI_ci_lo": float("nan"), "KDI_ci_hi": float("nan"),
                   "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0}
            if first == "real" and row["headroom"] >= args.min_headroom:
                row["KDI"] = diff.mean() / row["headroom"]
                row["KDI_ci_lo"], row["KDI_ci_hi"] = bootstrap_ratio_ci(a, b, rand)
            contrasts.append(row)
    contrasts = pd.DataFrame(contrasts)
    contrasts["p_holm"] = holm(contrasts["p_wilcoxon"].tolist())

    s = summary[summary["tag"] != "random"].copy()
    m = s["tag"].str.extract(SEED_TAG)
    s["config"] = m["config"].fillna(s["tag"])
    s = s[s["config"].str.contains("-lora-")]
    seeds = (s.groupby(["config", "condition"])[args.metric].agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"count": "n_seeds"}))

    out = RESULTS / "ext" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset}_{stem}"
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    contrasts.to_csv(out / f"{prefix}_contrasts.csv", index=False)
    seeds.to_csv(out / f"{prefix}_seeds.csv", index=False)
    per_user.to_parquet(out / f"{prefix}_per_user.parquet", index=False)

    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 20)
    print(f"== {args.dataset} ({stem}): {args.metric} by condition")
    print(summary.pivot_table(index="tag", columns="condition", values=args.metric).round(3).to_string())
    print(f"\n== new contrasts")
    show = contrasts[contrasts["contrast"].isin([f"{a}-{b}" for a, b, _ in EXT_CONTRASTS])]
    print(show[["tag", "contrast", "n_users", "delta", "delta_ci_lo", "delta_ci_hi", "KDI", "p_holm"]]
          .round(4).to_string(index=False))
    if len(seeds):
        print("\n== LoRA seeds")
        print(seeds.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
