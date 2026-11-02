from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import RESULTS, proc_dir
from .evaluate import CONTRASTS, bootstrap_ci, load_records
from .run import default_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--metric", default="ndcg@10")
    args = ap.parse_args()

    eval_path = args.eval_set or default_split(proc_dir(args.dataset), "test")
    recs = load_records(RESULTS / "raw" / args.dataset / eval_path.stem)
    recs = recs[recs["condition"] != "idonly"]
    rates = recs.groupby(["tag", "condition"])[["complete", "no_letters"]].mean()
    wide = recs.pivot_table(index=["tag", "user_id", "order_idx"], columns="condition",
                            values=[args.metric, "complete"])

    rows = []
    for tag, g in wide.groupby(level="tag"):
        for first, second, meaning in CONTRASTS:
            if (args.metric, first) not in g or (args.metric, second) not in g:
                continue
            pair = g[[(args.metric, first), (args.metric, second), ("complete", first), ("complete", second)]].dropna()
            if pair.empty:
                continue
            diff = pair[(args.metric, first)] - pair[(args.metric, second)]
            both = (pair[("complete", first)] == 1) & (pair[("complete", second)] == 1)
            per_user = diff[both].groupby(level="user_id").mean()
            lo, hi = bootstrap_ci(per_user.to_numpy()) if len(per_user) > 1 else (float("nan"), float("nan"))
            rows.append({
                "tag": tag, "contrast": f"{first}-{second}", "meaning": meaning,
                "complete_rate_first": rates.loc[(tag, first), "complete"],
                "complete_rate_second": rates.loc[(tag, second), "complete"],
                "delta_all": diff.mean(), "delta_complete": diff[both].mean(),
                "delta_complete_ci_lo": lo, "delta_complete_ci_hi": hi,
                "share_complete": both.mean(), "n_users_complete": len(per_user),
            })
    out = pd.DataFrame(rows)
    path = RESULTS / "summary" / f"{args.dataset}_{eval_path.stem}_answer_quality.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 500)
    show = out[out["contrast"].isin(["real-described", "real-masked"])]
    print(f"== {args.dataset} ({eval_path.stem}): {args.metric}, all pairs vs pairs complete in both conditions")
    print(show.drop(columns=["meaning"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
