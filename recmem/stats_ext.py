from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist, ttest_1samp, wilcoxon

from .common import RESULTS, read_jsonl
from .evaluate import (CONTRASTS, METRICS, bootstrap_ci, bootstrap_ratio_ci, expected_metrics, holm, load_records,
                       record_metrics)
from .evaluate_ext import EXT_CONTRASTS, load_both

SUM = RESULTS / "ext" / "summary"
DATASETS = ["ml-1m", "amazon-videogames", "amazon-beauty", "amazon-grocery"]
MODELS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.8-27b", "gemma4-e4b", "gemma4-12b", "gemma4-31b"]
MAIN = "test_n2500_c20_s42"


def per_user_table(recs: pd.DataFrame) -> pd.DataFrame:
    return recs.groupby(["tag", "condition", "user_id"])[[*METRICS, "no_letters", "complete"]].mean().reset_index()


def summarise(per_user: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    rows = []
    for (tag, cond), g in per_user.groupby(["tag", "condition"]):
        lo, hi = bootstrap_ci(g[metric].to_numpy())
        rows.append({"tag": tag, "condition": cond, "n_users": len(g), **{m: g[m].mean() for m in METRICS},
                     f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi, "complete_rate": g["complete"].mean()})
    return pd.DataFrame(rows)


def paired_contrasts(per_user: pd.DataFrame, pairs, rand: float, metric: str = "ndcg@10",
                     min_headroom: float = 0.05) -> pd.DataFrame:
    wide = per_user.pivot_table(index=["tag", "user_id"], columns="condition", values=metric)
    rows = []
    for tag, g in wide.groupby(level="tag"):
        for first, second, meaning in pairs:
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
                   "KDI": float("nan"), "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0}
            if first == "real" and a.mean() - rand >= min_headroom:
                row["KDI"] = diff.mean() / (a.mean() - rand)
                row["KDI_ci_lo"], row["KDI_ci_hi"] = bootstrap_ratio_ci(a, b, rand)
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    return out


def cmd_history(stem: str) -> None:
    rows = []
    for ds in DATASETS:
        try:
            recs = load_both(ds, stem)
        except SystemExit:
            continue
        rand = expected_metrics(range(1, int(recs["n_cand"].mode()[0]) + 1))["ndcg@10"]
        pu = per_user_table(recs)
        for (tag, cond), g in pu.groupby(["tag", "condition"]):
            if not ((tag in MODELS and cond in ("realshuf", "maskshuf")) or tag == "mostpop"):
                continue
            x = g["ndcg@10"].to_numpy() - rand
            lo, hi = bootstrap_ci(x)
            rows.append({"dataset": ds, "tag": tag, "condition": cond, "n_users": len(g), "ndcg@10": x.mean() + rand,
                         "random": rand, "delta_vs_random": x.mean(), "ci_lo": lo, "ci_hi": hi,
                         "p_ttest": ttest_1samp(x, 0.0).pvalue, "complete_rate": g["complete"].mean()})
    out = pd.DataFrame(rows)
    llm = out["tag"] != "mostpop"
    out.loc[llm, "p_holm"] = holm(out.loc[llm, "p_ttest"].tolist())
    out.loc[~llm, "p_holm"] = holm(out.loc[~llm, "p_ttest"].tolist())
    out.to_csv(SUM / f"rev_history_{stem}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(out.round(4).to_string(index=False))


def cmd_readout(stem: str) -> None:
    for ds in DATASETS:
        base = RESULTS / "ext" / "readout" / ds / stem
        if not base.exists() or not any(base.glob("*/*.jsonl")):
            print(f"{ds}: no readout yet")
            continue
        recs = load_records(base)
        n_cand = int(recs["n_cand"].mode()[0])
        rand = expected_metrics(range(1, n_cand + 1))["ndcg@10"]
        pu = per_user_table(recs)
        summ = summarise(pu)
        contr = paired_contrasts(pu, CONTRASTS + EXT_CONTRASTS, rand)
        shares = []
        for tag in summ["tag"].unique():
            d = {r.contrast: r.delta for r in contr[contr.tag == tag].itertuples()}
            if "real-masked" in d and "real-content" in d:
                shares.append({"tag": tag, "contrast": "idshare", "delta": d["real-content"] / d["real-masked"]})
        contr = pd.concat([contr, pd.DataFrame(shares)], ignore_index=True)
        prefix = f"{ds}_{stem}_readout"
        summ.to_csv(SUM / f"{prefix}_summary.csv", index=False)
        contr.to_csv(SUM / f"{prefix}_contrasts.csv", index=False)
        pu.to_parquet(SUM / f"{prefix}_per_user.parquet", index=False)
        print(f"== {ds} readout NDCG@10")
        print(summ.pivot_table(index="tag", columns="condition", values="ndcg@10").round(3).to_string())
        print(contr[contr.contrast.isin(["real-masked", "real-content", "content-masked", "idshare"])]
              .pivot_table(index="tag", columns="contrast", values="delta").round(3).to_string())


TEXT_PAIRS = [
    ("real", "masked", "total effect of the name"), ("real", "described", "name vs description"),
    ("real", "content", "identity"), ("content", "masked", "generic title content"),
    ("both", "described", "name on top of the description"), ("described", "masked", "description vs code"),
]


def cmd_baselines(stem: str) -> None:
    for ds in DATASETS:
        base = RESULTS / "ext" / "baselines" / ds / stem
        if not base.exists() or not any(base.glob("*/*.jsonl")):
            print(f"{ds}: no references yet")
            continue
        recs = load_records(base)
        rand = expected_metrics(range(1, int(recs["n_cand"].mode()[0]) + 1))["ndcg@10"]
        pu = per_user_table(recs)
        summ = summarise(pu)
        contr = paired_contrasts(pu, TEXT_PAIRS, rand)
        sas = pd.DataFrame([{"user_id": r["user_id"], **record_metrics(r["response"], r["target_label"], len(r["candidates"]))}
                            for r in read_jsonl(RESULTS / "raw" / ds / stem / "sasrec" / "idonly.jsonl")])
        sas_pu = sas.groupby("user_id")["ndcg@10"].mean()
        extra = []
        for (tag, cond), g in pu.groupby(["tag", "condition"]):
            if not tag.startswith("textsasrec"):
                continue
            x = g.set_index("user_id")["ndcg@10"]
            diff = (x - sas_pu.loc[x.index]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            extra.append({"tag": tag, "contrast": f"{cond}-sasrec", "meaning": "text-aware vs ID-only SASRec",
                          "n_users": len(diff), "delta": diff.mean(), "delta_ci_lo": lo, "delta_ci_hi": hi,
                          "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
        extra = pd.DataFrame(extra)
        if len(extra):
            extra["p_holm"] = holm(extra["p_wilcoxon"].tolist())
        contr = pd.concat([contr, extra], ignore_index=True)
        prefix = f"{ds}_{stem}_baselines"
        summ.to_csv(SUM / f"{prefix}_summary.csv", index=False)
        contr.to_csv(SUM / f"{prefix}_contrasts.csv", index=False)
        print(f"== {ds} references NDCG@10")
        print(summ.pivot_table(index="tag", columns="condition", values="ndcg@10").round(3).to_string())
        print(contr[["tag", "contrast", "delta", "delta_ci_lo", "delta_ci_hi", "p_holm"]].round(4).to_string(index=False))


def cmd_catonly(stem: str) -> None:
    rows = []
    for ds in DATASETS:
        base = RESULTS / "ext" / "baselines" / ds / stem
        recs = load_records(base)
        pu = per_user_table(recs).pivot_table(index="user_id", columns=["tag", "condition"], values="ndcg@10")
        sas = pd.DataFrame([{"user_id": r["user_id"], **record_metrics(r["response"], r["target_label"], len(r["candidates"]))}
                            for r in read_jsonl(RESULTS / "raw" / ds / stem / "sasrec" / "idonly.jsonl")])
        sas = sas.groupby("user_id")["ndcg@10"].mean()
        cat = pu[("catsasrec-textid", "catonly")]
        for name, other in (("catonly-sasrec", sas), ("masked-catonly", pu[("textsasrec-textid", "masked")]),
                            ("real-catonly", pu[("textsasrec-textid", "real")])):
            a, b = (cat, other) if name == "catonly-sasrec" else (other, cat)
            diff = (a - b.loc[a.index]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            rows.append({"dataset": ds, "contrast": name, "n_users": len(diff), "delta": diff.mean(), "ci_lo": lo,
                         "ci_hi": hi, "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    out.to_csv(SUM / f"rev_catonly_{stem}.csv", index=False)
    print(out.round(4).to_string(index=False))


def cmd_vsref(stem: str) -> None:
    rows = []
    for ds in DATASETS:
        pu = pd.read_parquet(SUM / f"{ds}_{stem}_per_user.parquet")
        wide = pu.pivot_table(index="user_id", columns=["tag", "condition"], values="ndcg@10")
        for ref in ("sasrec", "ease"):
            r = wide[(ref, "idonly")]
            for tag in MODELS:
                for cond in ("real", "masked"):
                    diff = (wide[(tag, cond)] - r).dropna().to_numpy()
                    lo, hi = bootstrap_ci(diff)
                    rows.append({"dataset": ds, "tag": tag, "condition": cond, "reference": ref, "n_users": len(diff),
                                 "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                                 "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    out.to_csv(SUM / f"rev_vsref_{stem}.csv", index=False)
    print(out.round(4).to_string(index=False))


def cmd_vsbest(stem: str) -> None:
    rows = []
    for ds in DATASETS:
        pu = pd.read_parquet(SUM / f"{ds}_{stem}_per_user.parquet")
        wide = pu.pivot_table(index="user_id", columns=["tag", "condition"], values="ndcg@10")
        best = max(MODELS, key=lambda m: wide[(m, "real")].mean())
        llm = wide[(best, "real")]
        refs = per_user_table(load_records(RESULTS / "ext" / "baselines" / ds / stem)).pivot_table(
            index="user_id", columns=["tag", "condition"], values="ndcg@10")
        for name, ref in (("textid-real", refs[("textsasrec-textid", "real")]),
                          ("textid-catonly", refs[("catsasrec-textid", "catonly")])):
            diff = (ref - llm.loc[ref.index]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            rows.append({"dataset": ds, "reference": name, "best_llm": best, "n_users": len(diff),
                         "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                         "p_wilcoxon": wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    out.to_csv(SUM / f"rev_vsbest_{stem}.csv", index=False)
    print(out.round(4).to_string(index=False))


def ols_fe(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, float, float]:
    levels = sorted(set(groups))
    X = np.column_stack([x] + [(groups == g).astype(float) for g in levels])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[0, 0]))
    tval = beta[0] / se
    q = t_dist.ppf(0.975, dof)
    return float(beta[0]), float(2 * t_dist.sf(abs(tval), dof)), float(beta[0] - q * se), float(beta[0] + q * se)


def cmd_cells() -> None:
    recs = []
    for ds in DATASETS:
        c = pd.read_csv(SUM / f"{ds}_{MAIN}_contrasts.csv")
        p = pd.read_csv(RESULTS / "summary" / f"{ds}_{MAIN}_probes.csv").set_index("tag")
        for tag in MODELS:
            row = {"dataset": ds, "tag": tag, "comp": float(p.loc[tag, "complete_mean_score"])}
            for key, name in (("d53", "both-described"), ("d04", "real-content"), ("d03", "real-described"),
                              ("d01", "real-masked")):
                row[key] = float(c[(c.tag == tag) & (c.contrast == name)]["delta"].iloc[0])
            recs.append(row)
    df = pd.DataFrame(recs)
    out = []
    for key in ("d53", "d04", "d03"):
        rho, p = spearmanr(df["comp"], df[key])
        out.append({"effect": key, "analysis": "pooled spearman (28 cells)", "estimate": rho, "p": p, "n": len(df)})
        for ds, g in df.groupby("dataset"):
            rho, p = spearmanr(g["comp"], g[key])
            out.append({"effect": key, "analysis": f"spearman within {ds}", "estimate": rho, "p": p, "n": len(g)})
        cen = df.assign(x=df["comp"] - df.groupby("dataset")["comp"].transform("mean"),
                        y=df[key] - df.groupby("dataset")[key].transform("mean"))
        rho, p = spearmanr(cen["x"], cen["y"])
        out.append({"effect": key, "analysis": "spearman on dataset-centred values", "estimate": rho, "p": p,
                    "n": len(df)})
        b, p, lo, hi = ols_fe(df["comp"].to_numpy(), df[key].to_numpy(), df["dataset"].to_numpy())
        out.append({"effect": key, "analysis": "OLS slope with dataset fixed effects", "estimate": b, "p": p,
                    "n": len(df), "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(out)
    out.to_csv(SUM / "rev_cells.csv", index=False)
    print(out.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["history", "readout", "baselines", "cells", "vsref", "catonly", "vsbest"])
    ap.add_argument("--eval-set-stem", default=MAIN)
    args = ap.parse_args()
    if args.command == "cells":
        cmd_cells()
    else:
        {"history": cmd_history, "readout": cmd_readout, "baselines": cmd_baselines,
         "vsref": cmd_vsref, "catonly": cmd_catonly, "vsbest": cmd_vsbest}[args.command](args.eval_set_stem)


if __name__ == "__main__":
    main()
