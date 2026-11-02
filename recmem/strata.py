from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import RESULTS, proc_dir, read_jsonl
from .evaluate import bootstrap_ci
from .run import default_split

DELTAS = [("real", "described"), ("real", "masked"), ("real", "swapped"), ("masked", "swapped")]


def user_table(dataset: str, eval_path: Path) -> pd.DataFrame:
    items = pd.read_parquet(proc_dir(dataset) / "items.parquet").set_index("item_id")
    rows = read_jsonl(eval_path)
    df = pd.DataFrame({"user_id": [r["user_id"] for r in rows], "target": [r["target"] for r in rows],
                       "history_len": [len(r["history"]) for r in rows]})
    df["popularity"] = df["target"].map(items["popularity"]).astype(float)
    df["year"] = df["target"].map(items["year"]).astype("Float64")
    return df


def probe_scores(dataset: str, stem: str, tag: str) -> pd.DataFrame:
    pdir = RESULTS / "probes" / dataset / stem
    out = pd.DataFrame(columns=["target"])
    comp, know = pdir / f"{tag}_complete.jsonl", pdir / f"{tag}_know.jsonl"
    if comp.exists():
        c = pd.DataFrame(read_jsonl(comp))
        out = c[["item_id", "score"]].rename(columns={"item_id": "target", "score": "probe_complete"})
    if know.exists():
        k = pd.DataFrame(read_jsonl(know))
        if len(k):
            k["probe_year"] = np.where(k["exact"], "exact", np.where(k["within1"], "off by 1", "wrong"))
            out = out.merge(k[["item_id", "probe_year"]].rename(columns={"item_id": "target"}), on="target", how="outer")
    return out


def bins(users: pd.DataFrame) -> dict[str, pd.Series]:
    out = {
        "popularity": pd.qcut(users["popularity"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]),
        "history": pd.qcut(users["history_len"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]),
        "year": users["year"].map(lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) else None),
    }
    if "probe_complete" in users:
        out["probe_complete"] = pd.cut(users["probe_complete"], [-0.01, 0.0, 0.5, 0.8, 1.01],
                                       labels=["0", "(0, 0.5)", "[0.5, 0.8)", ">= 0.8"])
    if "probe_year" in users:
        out["probe_year"] = users["probe_year"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--metric", default="ndcg@10")
    args = ap.parse_args()

    eval_path = args.eval_set or default_split(proc_dir(args.dataset), "test")
    stem = f"{args.dataset}_{eval_path.stem}"
    per_user = pd.read_parquet(RESULTS / "summary" / f"{stem}_per_user.parquet")
    wide = per_user.pivot_table(index=["tag", "user_id"], columns="condition", values=args.metric).reset_index()
    users = user_table(args.dataset, eval_path)

    cells, corrs = [], []
    for tag, g in wide.groupby("tag"):
        if "real" not in g or g["real"].isna().all():
            continue
        d = users.merge(g, on="user_id")
        probes = probe_scores(args.dataset, eval_path.stem, tag)
        if len(probes):
            d = d.merge(probes, on="target", how="left")
        for first, second in DELTAS:
            if first in d and second in d:
                d[f"{first}-{second}"] = d[first] - d[second]
        for name, labels in bins(d).items():
            for label, cell in d.groupby(labels, observed=True):
                row = {"tag": tag, "stratifier": name, "bin": label, "n_users": len(cell)}
                for cond in ("real", "described", "masked", "swapped"):
                    if cond in cell:
                        row[cond] = cell[cond].mean()
                for first, second in DELTAS:
                    col = f"{first}-{second}"
                    if col in cell and cell[col].notna().any():
                        vals = cell[col].dropna().to_numpy()
                        lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (np.nan, np.nan)
                        row.update({col: vals.mean(), f"{col}_ci_lo": lo, f"{col}_ci_hi": hi})
                cells.append(row)
        for var in ("popularity", "history_len", "probe_complete"):
            if var not in d:
                continue
            for first, second in DELTAS:
                col = f"{first}-{second}"
                ok = d[[var, col]].dropna() if col in d else pd.DataFrame()
                if len(ok) > 10:
                    rho, p = spearmanr(ok[var], ok[col])
                    corrs.append({"tag": tag, "variable": var, "delta": col, "n_users": len(ok), "spearman": rho, "p": p})

    out = RESULTS / "summary"
    cells, corrs = pd.DataFrame(cells), pd.DataFrame(corrs)
    cells.to_csv(out / f"{stem}_strata.csv", index=False)
    corrs.to_csv(out / f"{stem}_strata_corr.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 500)
    show = ["tag", "stratifier", "bin", "n_users", "real", "real-described", "real-masked", "masked-swapped"]
    print(f"== {stem}: strata of {args.metric}")
    print(cells[[c for c in show if c in cells]].round(3).to_string(index=False))
    if len(corrs):
        print(f"\n== {stem}: Spearman correlation of per-user deltas")
        print(corrs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
