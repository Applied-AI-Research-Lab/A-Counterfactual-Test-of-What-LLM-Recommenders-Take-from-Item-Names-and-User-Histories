#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
SUM, EXT = ROOT / "results" / "summary", ROOT / "results" / "ext" / "summary"
MAIN = "test_n2500_c20_s42"

DS = {"ml": "ml-1m", "vg": "amazon-videogames", "be": "amazon-beauty", "gr": "amazon-grocery"}
DS_NAME = {"ml": "MovieLens-1M", "vg": "Video Games", "be": "Beauty", "gr": "Grocery"}
MODELS = {"q4": "qwen3.5-4b", "q9": "qwen3.5-9b", "q27": "qwen3.5-27b", "q38": "qwen3.8-27b",
          "g4": "gemma4-e4b", "g12": "gemma4-12b", "g31": "gemma4-31b"}
MODEL_NAME = {"q4": "Qwen3.5-4B", "q9": "Qwen3.5-9B", "q27": "Qwen3.5-27B", "q38": "Qwen3.8-27B",
              "g4": "Gemma-4-E4B", "g12": "Gemma-4-12B", "g31": "Gemma-4-31B"}
COND = {"c0": "real", "c1": "masked", "c2": "swapped", "c3": "described", "c4": "content", "c5": "both",
        "c6": "realshuf", "c7": "maskshuf"}
CONTRAST = {
    "d03": ("real", "described"), "d01": ("real", "masked"), "d12": ("masked", "swapped"),
    "d04": ("real", "content"), "d41": ("content", "masked"), "d53": ("both", "described"),
    "d50": ("both", "real"), "d06": ("real", "realshuf"), "d17": ("masked", "maskshuf"), "d31": ("described", "masked"),
}

V: dict[str, float | int | str] = {}
T: dict[str, str] = {}


def put(key: str, value, text: str) -> None:
    if key in V:
        raise SystemExit(f"duplicate key {key}")
    V[key], T[key] = value, text


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float, nd: int = 3) -> str:
    s = f"{abs(x):.{nd}f}"
    if round(x, nd) == 0:
        return s
    return ("+" if x > 0 else "\\textminus") + s


def mn(x: float, nd: int = 2) -> str:
    t = f"{abs(x):.{nd}f}"
    return ("\\textminus" + t) if (x < 0 and round(x, nd) != 0) else t


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}\\%"


def thou(n: int) -> str:
    return f"{n:,}"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def data_stats() -> None:
    for d, ds in DS.items():
        meta = json.loads((ROOT / "data/processed" / ds / "prepare_meta.json").read_text())
        items = pd.read_parquet(ROOT / "data/processed" / ds / "items.parquet")
        vs = json.loads((ROOT / "data/processed" / ds / "views_s42.stats.json").read_text())
        ve = json.loads((ROOT / "data/processed" / ds / "views_ext_s42.stats.json").read_text())
        rows = [json.loads(l) for l in open(ROOT / "data/processed" / ds / f"{MAIN}.jsonl")]
        hist = [len(r["history"]) for r in rows]
        put(f"{d}.users", meta["users"], thou(meta["users"]))
        put(f"{d}.items", meta["items"], thou(meta["items"]))
        put(f"{d}.inter", meta["interactions"], thou(meta["interactions"]))
        put(f"{d}.ncat", int(items["category"].nunique()), str(int(items["category"].nunique())))
        put(f"{d}.testusers", len(rows), thou(len(rows)))
        put(f"{d}.histmed", float(statistics.median(hist)), f"{statistics.median(hist):g}")
        put(f"{d}.titlechars", float(items["title"].str.len().median()), f"{items['title'].str.len().median():g}")
        put(f"{d}.swapcat", vs["share_same_category_swap"], pct(vs["share_same_category_swap"], 2))
        put(f"{d}.swappop", vs["median_abs_popularity_pct_shift"], f"{100 * vs['median_abs_popularity_pct_shift']:.1f}")
        put(f"{d}.descuniq", vs["share_described_unique"], pct(vs["share_described_unique"]))
        put(f"{d}.descchars", vs["median_described_chars"], f"{vs['median_described_chars']:g}")
        put(f"{d}.descleak", vs["share_described_title_leak"], pct(vs["share_described_title_leak"], 2))
        put(f"{d}.contempty", ve["share_content_empty"], pct(ve["share_content_empty"]))
        put(f"{d}.contkept", ve["median_share_title_words_kept"], pct(ve["median_share_title_words_kept"], 0))
        put(f"{d}.contleak", ve["share_content_with_distinctive_title_word"],
            pct(ve["share_content_with_distinctive_title_word"], 2))
        lens = {len(r["candidates"][0]) for r in rows} | {len(r["candidates"]) for r in rows}
        put(f"{d}.ncand", len(rows[0]["candidates"][0]), str(len(rows[0]["candidates"][0])))
        put(f"{d}.norders", len(rows[0]["candidates"]), str(len(rows[0]["candidates"])))
        assert lens == {20, 3}, lens
        if meta.get("max_users"):
            put(f"{d}.sampled", meta["max_users"], thou(meta["max_users"]))
    log = (ROOT / "logs" / "prep-amazon.log").read_text()
    for d, ds in (("be", "amazon-beauty"), ("gr", "amazon-grocery")):
        m = re.search(rf"{ds}: sampled ([\d,]+) of ([\d,]+) eligible users", log)
        put(f"{d}.eligible", int(m.group(2).replace(",", "")), m.group(2))


def setup_stats() -> None:
    for d, ds in DS.items():
        tun = json.loads((ROOT / "results/raw" / ds / MAIN / "ease" / "tuning.json").read_text())
        put(f"{d}.easelambda", tun["best_lambda"], thou(int(tun["best_lambda"])))
        sas = json.loads((ROOT / "results/raw" / ds / MAIN / "sasrec" / "tuning.json").read_text())
        best = max(sas["runs"], key=lambda r: r["valid_ndcg@10"])
        put(f"{d}.sasdrop", best["config"]["dropout"], f"{best['config']['dropout']:g}")
        put(f"{d}.sasepoch", best["best_epoch"], str(best["best_epoch"]))
        fr = json.loads((ROOT / "results/raw" / ds / MAIN / "fullrank.json").read_text())
        for b in ("ease", "sasrec", "mostpop"):
            put(f"{d}.full.{b}.hr", fr[b]["hr@10"], f3(fr[b]["hr@10"]))
        put(f"{d}.full.random.hr", fr["random_hr@10"], f"{fr['random_hr@10']:.4f}")
    for d, ds in DS.items():
        toks = [json.loads(l)["prompt_tokens"] for l in open(ROOT / "results/raw" / ds / MAIN / "qwen3.8-27b" / "real.jsonl")]
        put(f"{d}.ptok", float(np.mean(toks)), thou(int(round(np.mean(toks)))))
    infos = [json.loads(p.read_text()) for p in sorted((ROOT / "checkpoints").glob("*/*/train_info.json"))]
    infos += [json.loads(p.read_text()) for p in sorted((ROOT / "checkpoints_ext").glob("*/*/train_info.json"))]
    put("lora.ntrain", len(infos), str(len(infos)))
    t = infos[0]
    put("lora.r", t["lora"]["r"], str(t["lora"]["r"]))
    put("lora.alpha", t["lora"]["alpha"], str(t["lora"]["alpha"]))
    put("lora.dropout", t["lora"]["dropout"], f"{t['lora']['dropout']:g}")
    put("lora.lr", t["train"]["lr"], "2\\times10^{-4}" if t["train"]["lr"] == 2e-4 else str(t["train"]["lr"]))
    put("lora.bs", t["train"]["batch_size"], str(t["train"]["batch_size"]))
    assert all(i["lora"] == t["lora"] and i["train"]["lr"] == t["train"]["lr"]
               and i["train"]["batch_size"] == t["train"]["batch_size"] for i in infos), "LoRA recipes differ"
    ex = sorted({i["n_examples"] for i in infos})
    put("lora.examples.max", max(ex), thou(max(ex)))
    put("lora.examples.ml", min(ex), thou(min(ex)))
    st = sorted({i["steps"] for i in infos})
    put("lora.steps.max", max(st), str(max(st)))
    put("lora.steps.ml", min(st), str(min(st)))
    mins = [i["train_seconds"] / 60 for i in infos]
    put("lora.minutes.min", min(mins), f"{min(mins):.0f}")
    put("lora.minutes.max", max(mins), f"{max(mins):.0f}")


def main_grid() -> dict:
    allc = {}
    for d, ds in DS.items():
        s = read(EXT / f"{ds}_{MAIN}_summary.csv")
        c = read(EXT / f"{ds}_{MAIN}_contrasts.csv")
        w = s.pivot(index="tag", columns="condition", values="ndcg@10")
        cr = s.pivot(index="tag", columns="condition", values="complete_rate")
        allc[d] = c
        rand = float(s.loc[s.tag == "random", "ndcg@10"].iloc[0])
        if d == "ml":
            put("random", rand, f3(rand))
        for b in ("ease", "sasrec", "mostpop"):
            put(f"{d}.{b}", float(w.loc[b, "idonly"]), f3(w.loc[b, "idonly"]))
        for m, tag in MODELS.items():
            for k, cond in COND.items():
                put(f"{d}.{m}.{k}", float(w.loc[tag, cond]), f3(w.loc[tag, cond]))
                put(f"{d}.{m}.{k}.complete", float(cr.loc[tag, cond]), pct(cr.loc[tag, cond]))
            for k, (a, b) in CONTRAST.items():
                row = c[(c.tag == tag) & (c.contrast == f"{a}-{b}")].iloc[0]
                put(f"{d}.{m}.{k}", float(row.delta), sg(row.delta))
                put(f"{d}.{m}.{k}.lo", float(row.delta_ci_lo), sg(row.delta_ci_lo))
                put(f"{d}.{m}.{k}.hi", float(row.delta_ci_hi), sg(row.delta_ci_hi))
                put(f"{d}.{m}.{k}.p", float(row.p_holm), f"{row.p_holm:.2g}")
                if a == "real" and not np.isnan(row.KDI):
                    put(f"{d}.{m}.{k}.kdi", float(row.KDI), mn(row.KDI))
                    put(f"{d}.{m}.{k}.kdilo", float(row.KDI_ci_lo), mn(row.KDI_ci_lo))
                    put(f"{d}.{m}.{k}.kdihi", float(row.KDI_ci_hi), mn(row.KDI_ci_hi))
        for k, (a, b) in CONTRAST.items():
            sub = c[c.tag.isin(MODELS.values()) & (c.contrast == f"{a}-{b}")]
            put(f"{d}.{k}.min", float(sub.delta.min()), sg(sub.delta.min()))
            put(f"{d}.{k}.max", float(sub.delta.max()), sg(sub.delta.max()))
            put(f"{d}.{k}.nsig", int((sub.p_holm < 0.05).sum()), str(int((sub.p_holm < 0.05).sum())))
            if a == "real":
                kd = sub.KDI.dropna()
                if len(kd):
                    put(f"{d}.{k}.kdimin", float(kd.min()), mn(kd.min()))
                    put(f"{d}.{k}.kdimax", float(kd.max()), mn(kd.max()))
        for k, cond in COND.items():
            vals = [float(w.loc[t, cond]) for t in MODELS.values()]
            put(f"{d}.{k}.min", min(vals), f3(min(vals)))
            put(f"{d}.{k}.max", max(vals), f3(max(vals)))
        crs = cr.loc[list(MODELS.values()), list(COND.values())]
        put(f"{d}.complete.min", float(crs.values.min()), pct(crs.values.min()))
        nl = s[s.tag.isin(MODELS.values())]["no_letters_rate"]
        put(f"{d}.noletters.max", float(nl.max()), pct(nl.max(), 2))
    for k in CONTRAST:
        vals = [V[f"{d}.{m}.{k}"] for d in DS for m in MODELS]
        put(f"all.{k}.min", min(vals), sg(min(vals)))
        put(f"all.{k}.max", max(vals), sg(max(vals)))
        put(f"all.{k}.nsig", sum(V[f"{d}.{k}.nsig"] for d in DS), str(sum(V[f"{d}.{k}.nsig"] for d in DS)))
    for k in ("c6", "c7"):
        vals = [V[f"{d}.{m}.{k}"] for d in DS for m in MODELS]
        put(f"all.{k}.min", min(vals), f3(min(vals)))
        put(f"all.{k}.max", max(vals), f3(max(vals)))
    shares = [V[f"{d}.{m}.d04"] / V[f"{d}.{m}.d01"] for d in ("vg", "be", "gr") for m in MODELS]
    put("amz.idshare.min", min(shares), pct(min(shares), 0))
    put("amz.idshare.max", max(shares), pct(max(shares), 0))
    put("amz.idshare.median", float(np.median(shares)), pct(float(np.median(shares)), 0))
    put("n.models", len(MODELS), str(len(MODELS)))
    return allc


def size_trends() -> None:
    ladders = {"qwen": ("q4", "q9", "q27"), "gemma": ("g4", "g12", "g31")}
    for key in ("c0", "c3", "c1", "d01", "d03", "d04", "d41", "d53", "d06"):
        inc, total, exceptions = 0, 0, []
        for lname, lad in ladders.items():
            for d in DS:
                vals = [V[f"{d}.{m}.{key}"] for m in lad]
                total += 1
                if vals[0] < vals[1] < vals[2]:
                    inc += 1
                else:
                    exceptions.append(f"{lname}/{d}")
        put(f"size.mono.{key}", inc, str(inc))
        put(f"size.mono.{key}.of", total, str(total))
    for key in ("c0", "c1", "c3", "c4", "c5", "c6", "d01", "d03", "d04", "d53", "d06"):
        diffs = [V[f"{d}.q38.{key}"] - V[f"{d}.q27.{key}"] for d in DS]
        put(f"gen.{key}.maxabs", max(abs(x) for x in diffs), f3(max(abs(x) for x in diffs)))
    for d in DS:
        put(f"{d}.gain.qwen", V[f"{d}.q27.c0"] - V[f"{d}.q4.c0"], sg(V[f"{d}.q27.c0"] - V[f"{d}.q4.c0"]))
        put(f"{d}.gain.gemma", V[f"{d}.g31.c0"] - V[f"{d}.g4.c0"], sg(V[f"{d}.g31.c0"] - V[f"{d}.g4.c0"]))
    for k in ("c6", "c7"):
        dev = max(abs(V[f"{d}.{m}.{k}"] - V["random"]) for d in DS for m in MODELS)
        put(f"all.{k}.maxdev", dev, f3(dev))
    dev6 = max(abs(V[f"{d}.{m}.c6"] - V["random"]) for d in DS for m in MODELS)
    for d in DS:
        put(f"{d}.c0.abovesasrec", sum(V[f"{d}.{m}.c0"] > V[f"{d}.sasrec"] for m in MODELS),
            str(sum(V[f"{d}.{m}.c0"] > V[f"{d}.sasrec"] for m in MODELS)))
        put(f"{d}.c0.aboveease", sum(V[f"{d}.{m}.c0"] > V[f"{d}.ease"] for m in MODELS),
            str(sum(V[f"{d}.{m}.c0"] > V[f"{d}.ease"] for m in MODELS)))
        put(f"{d}.c1.aboveease", sum(V[f"{d}.{m}.c1"] > V[f"{d}.ease"] for m in MODELS),
            str(sum(V[f"{d}.{m}.c1"] > V[f"{d}.ease"] for m in MODELS)))
        put(f"{d}.c1.abovesasrec", sum(V[f"{d}.{m}.c1"] > V[f"{d}.sasrec"] for m in MODELS),
            str(sum(V[f"{d}.{m}.c1"] > V[f"{d}.sasrec"] for m in MODELS)))
        put(f"{d}.c5.best", max(V[f"{d}.{m}.c5"] for m in MODELS), f3(max(V[f"{d}.{m}.c5"] for m in MODELS)))
    for d in DS:
        for m in ("q4", "g4"):
            put(f"{d}.{m}real.c0.vssasrec", V[f"{d}.{m}real.c0"] - V[f"{d}.sasrec"], sg(V[f"{d}.{m}real.c0"] - V[f"{d}.sasrec"]))
            put(f"{d}.{m}.ft.gain", V[f"{d}.{m}real.c0"] - V[f"{d}.{m}ro.c0"], sg(V[f"{d}.{m}real.c0"] - V[f"{d}.{m}ro.c0"]))
            put(f"{d}.{m}ro.d01", V[f"{d}.{m}ro.c0"] - V[f"{d}.{m}ro.c1"], sg(V[f"{d}.{m}ro.c0"] - V[f"{d}.{m}ro.c1"]))


def probes() -> None:
    for d, ds in DS.items():
        p = read(SUM / f"{ds}_{MAIN}_probes.csv").set_index("tag")
        pe = read(EXT / f"{ds}_{MAIN}_probes_ext.csv").set_index("tag")
        put(f"{d}.probe.nitems", int(p.loc["qwen3.8-27b", "n_items"]), thou(int(p.loc["qwen3.8-27b", "n_items"])))
        put(f"{d}.probe.nyear", int(p.loc["qwen3.8-27b", "n_items_with_year"]), thou(int(p.loc["qwen3.8-27b", "n_items_with_year"])))
        put(f"{d}.probe.ncomp", int(p.loc["qwen3.8-27b", "n_items_complete"]), thou(int(p.loc["qwen3.8-27b", "n_items_complete"])))
        put(f"{d}.probe.ncontent", int(pe.loc["qwen3.8-27b", "n_items_content"]), thou(int(pe.loc["qwen3.8-27b", "n_items_content"])))
        for m, tag in MODELS.items():
            put(f"{d}.{m}.reid", float(p.loc[tag, "reid_hit_rate"]), pct(p.loc[tag, "reid_hit_rate"]))
            put(f"{d}.{m}.year", float(p.loc[tag, "know_year_exact"]), pct(p.loc[tag, "know_year_exact"]))
            put(f"{d}.{m}.comp", float(p.loc[tag, "complete_mean_score"]), f"{p.loc[tag, 'complete_mean_score']:.2f}")
            put(f"{d}.{m}.comphit", float(p.loc[tag, "complete_hit_rate"]), pct(p.loc[tag, "complete_hit_rate"]))
            put(f"{d}.{m}.reidc", float(pe.loc[tag, "reid_content_strict_rate"]), pct(pe.loc[tag, "reid_content_strict_rate"]))
            put(f"{d}.{m}.reidcl", float(pe.loc[tag, "reid_content_hit_rate"]), pct(pe.loc[tag, "reid_content_hit_rate"]))
        for col, key, f in (("reid_hit_rate", "reid", pct), ("know_year_exact", "year", pct),
                            ("complete_mean_score", "comp", lambda x: f"{x:.2f}"), ("complete_hit_rate", "comphit", pct)):
            vals = p.loc[list(MODELS.values()), col]
            put(f"{d}.{key}.min", float(vals.min()), f(vals.min()))
            put(f"{d}.{key}.max", float(vals.max()), f(vals.max()))
        for col, key in (("reid_content_strict_rate", "reidc"), ("reid_content_hit_rate", "reidcl")):
            vals = pe.loc[list(MODELS.values()), col]
            put(f"{d}.{key}.min", float(vals.min()), pct(vals.min()))
            put(f"{d}.{key}.max", float(vals.max()), pct(vals.max()))
    for key in ("reid", "reidc"):
        vals = [V[f"{d}.{m}.{key}"] for d in DS for m in MODELS]
        put(f"all.{key}.max", max(vals), pct(max(vals)))
    xs = [V[f"{d}.{m}.comp"] for d in DS for m in MODELS]
    for k in ("d53", "d03", "d04"):
        ys = [V[f"{d}.{m}.{k}"] for d in DS for m in MODELS]
        from scipy.stats import spearmanr
        rho, pv = spearmanr(xs, ys)
        put(f"cell.rho.comp.{k}", float(rho), mn(rho))
        put(f"cell.rho.comp.{k}.p", float(pv), f"{pv:.2g}")


def strata() -> None:
    for d, ds in DS.items():
        c = read(SUM / f"{ds}_{MAIN}_strata_corr.csv")
        c = c[c.tag.isin(MODELS.values())]
        for var, vk in (("probe_complete", "comp"), ("popularity", "pop"), ("history_len", "hist")):
            for delta, dk in (("real-described", "d03"), ("real-masked", "d01")):
                sub = c[(c.variable == var) & (c.delta == delta)]
                if len(sub):
                    put(f"{d}.rho.{vk}.{dk}.min", float(sub.spearman.min()), mn(sub.spearman.min()))
                    put(f"{d}.rho.{vk}.{dk}.max", float(sub.spearman.max()), mn(sub.spearman.max()))
    for var, vk in (("probe_complete", "comp"), ("popularity", "pop"), ("history_len", "hist")):
        for dk in ("d03", "d01"):
            mins = [V[f"{d}.rho.{vk}.{dk}.min"] for d in DS if f"{d}.rho.{vk}.{dk}.min" in V]
            maxs = [V[f"{d}.rho.{vk}.{dk}.max"] for d in DS if f"{d}.rho.{vk}.{dk}.max" in V]
            put(f"all.rho.{vk}.{dk}.min", min(mins), mn(min(mins)))
            put(f"all.rho.{vk}.{dk}.max", max(maxs), mn(max(maxs)))
    for vk, var in (("comp", "probe_complete"), ("pop", "popularity")):
        for dk in ("d03", "d01"):
            mins = [V[f"{d}.rho.{vk}.{dk}.min"] for d in ("vg", "be", "gr")]
            maxs = [V[f"{d}.rho.{vk}.{dk}.max"] for d in ("vg", "be", "gr")]
            put(f"amz.rho.{vk}.{dk}.min", min(mins), mn(min(mins)))
            put(f"amz.rho.{vk}.{dk}.max", max(maxs), mn(max(maxs)))


def robustness() -> None:
    for d, ds in DS.items():
        for set_key, stem in (("c100", "test_n500_c100_s42"), ("a0", "test_n500_c20_s42_a0")):
            c = read(SUM / f"{ds}_{stem}_contrasts.csv")
            s = read(SUM / f"{ds}_{stem}_summary.csv")
            if set_key == "a0":
                for b in ("ease", "sasrec", "mostpop"):
                    put(f"{d}.a0.{b}", float(s[(s.tag == b)]["ndcg@10"].iloc[0]), f3(s[(s.tag == b)]["ndcg@10"].iloc[0]))
            for k, (a, b) in (("d03", ("real", "described")), ("d01", ("real", "masked")), ("d12", ("masked", "swapped"))):
                sub = c[c.tag.isin(MODELS.values()) & (c.contrast == f"{a}-{b}")]
                put(f"{d}.{set_key}.{k}.min", float(sub.delta.min()), sg(sub.delta.min()))
                put(f"{d}.{set_key}.{k}.max", float(sub.delta.max()), sg(sub.delta.max()))
                put(f"{d}.{set_key}.{k}.nsig", int((sub.p_holm < 0.05).sum()), str(int((sub.p_holm < 0.05).sum())))
            rr = s[s.tag.isin(MODELS.values()) & (s.condition == "real")]["ndcg@10"]
            put(f"{d}.{set_key}.c0.min", float(rr.min()), f3(rr.min()))
            put(f"{d}.{set_key}.c0.max", float(rr.max()), f3(rr.max()))
            put(f"{d}.{set_key}.random", float(s[s.tag == "random"]["ndcg@10"].iloc[0]), f3(s[s.tag == "random"]["ndcg@10"].iloc[0]))
        aq = read(SUM / f"{ds}_{MAIN}_answer_quality.csv")
        p2 = aq[aq.tag.isin([t + "-p2" for t in MODELS.values()])]
        for k, cname in (("d03", "real-described"), ("d01", "real-masked")):
            sub = p2[p2.contrast == cname]
            put(f"{d}.p2.{k}.min", float(sub.delta_complete.min()), sg(sub.delta_complete.min()))
            put(f"{d}.p2.{k}.max", float(sub.delta_complete.max()), sg(sub.delta_complete.max()))
            put(f"{d}.p2.{k}.allmax", float(sub.delta_all.max()), sg(sub.delta_all.max()))
        sub = p2[p2.contrast == "real-described"]
        put(f"{d}.p2.sharecomplete.min", float(sub.share_complete.min()), pct(sub.share_complete.min(), 0))
        mp = aq[aq.tag.isin(MODELS.values()) & (aq.contrast == "real-described")]
        diff = (mp.delta_complete - mp.delta_all).abs().max()
        put(f"{d}.aq.d03.maxdiff", float(diff), f3(diff))
    put("all.aq.d03.maxdiff", max(V[f"{d}.aq.d03.maxdiff"] for d in DS), f3(max(V[f"{d}.aq.d03.maxdiff"] for d in DS)))
    for d, ds in DS.items():
        for m in ("q4", "g4"):
            c = read(EXT / f"{ds}_{MAIN}_contrasts.csv")
            row = c[(c.tag == MODELS[m] + "-ro") & (c.contrast == "real-described")].iloc[0]
            put(f"{d}.{m}ro.d03", float(row.delta), sg(row.delta))


def answer_quality_all() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from recmem.evaluate import load_records
    for d, ds in DS.items():
        recs = pd.concat([load_records(ROOT / "results/raw" / ds / MAIN), load_records(ROOT / "results/ext/raw" / ds / MAIN)])
        recs = recs[recs.tag.isin(MODELS.values())]
        wide = recs.pivot_table(index=["tag", "user_id", "order_idx"], columns="condition", values=["ndcg@10", "complete"])
        diffs = []
        for m, tag in MODELS.items():
            g = wide.loc[tag]
            for k, (a, b) in CONTRAST.items():
                both = (g[("complete", a)] == 1) & (g[("complete", b)] == 1)
                dc = float((g[("ndcg@10", a)] - g[("ndcg@10", b)])[both].mean())
                put(f"{d}.{m}.{k}.cmp", dc, sg(dc))
                put(f"{d}.{m}.{k}.cmpshare", float(both.mean()), pct(both.mean(), 0))
                diffs.append((m, k, abs(dc - V[f"{d}.{m}.{k}"])))
        for k in CONTRAST:
            dd = [x for m, kk, x in diffs if kk == k]
            dd_no_g4 = [x for m, kk, x in diffs if kk == k and m != "g4"]
            put(f"{d}.{k}.cmpdiff", max(dd), f3(max(dd)))
            put(f"{d}.{k}.cmpdiff.nog4", max(dd_no_g4), f3(max(dd_no_g4)))
    for k in CONTRAST:
        put(f"all.{k}.cmpdiff", max(V[f"{d}.{k}.cmpdiff"] for d in DS), f3(max(V[f"{d}.{k}.cmpdiff"] for d in DS)))
        put(f"all.{k}.cmpdiff.nog4", max(V[f"{d}.{k}.cmpdiff.nog4"] for d in DS), f3(max(V[f"{d}.{k}.cmpdiff.nog4"] for d in DS)))
    flips = [f"{d}.{m}.{k}" for d in DS for m in MODELS for k in CONTRAST
             if V[f"{d}.{m}.{k}.p"] < 0.05 and np.sign(V[f"{d}.{m}.{k}"]) != np.sign(V[f"{d}.{m}.{k}.cmp"])]
    put("aq.signflips", len(flips), str(len(flips)))
    (OUT / "aq_signflips.txt").write_text("\n".join(flips) + "\n")


def lora() -> None:
    sd = read(EXT / "LORA_seeds.csv")
    put("lora.nruns", len(sd), str(len(sd)))
    put("lora.ncollapsed", int(sd.collapsed.sum()), str(int(sd.collapsed.sum())))
    ok = sd[~sd.collapsed]
    for d, ds in DS.items():
        for m, tag in (("q4", "qwen3.5-4b"), ("g4", "gemma4-e4b")):
            s = read(EXT / f"{ds}_{MAIN}_summary.csv")
            w = s.pivot(index="tag", columns="condition", values="ndcg@10")
            for k in ("c0", "c3", "c1", "c4", "c5", "c6"):
                put(f"{d}.{m}ro.{k}", float(w.loc[tag + "-ro", COND[k]]), f3(w.loc[tag + "-ro", COND[k]]))
            for tr in ("real", "masked"):
                g = ok[(ok.dataset == ds) & (ok.model == tag) & (ok.train == tr)]
                put(f"{d}.{m}{tr}.n", len(g), str(len(g)))
                for k in ("c0", "c3", "c1", "c4", "c5", "c6"):
                    vals = g[COND[k]]
                    put(f"{d}.{m}{tr}.{k}", float(vals.mean()), f3(vals.mean()))
                    put(f"{d}.{m}{tr}.{k}.sd", float(vals.std(ddof=1)), f3(vals.std(ddof=1)))
                for kk, (a, b) in (("d01", ("real", "masked")), ("d03", ("real", "described")), ("d04", ("real", "content"))):
                    dv = (g[a] - g[b])
                    put(f"{d}.{m}{tr}.{kk}", float(dv.mean()), sg(dv.mean()))
    sds = [ok[(ok.dataset == ds) & (ok.model == tag) & (ok.train == tr)]["real"].std(ddof=1)
           for ds in DS.values() for tag in ("qwen3.5-4b", "gemma4-e4b") for tr in ("real", "masked")]
    put("lora.sd.max", float(max(sds)), f3(max(sds)))
    c6 = [V[f"{d}.{m}{tr}.c6"] for d in DS for m in ("q4", "g4") for tr in ("real", "masked")]
    put("lora.c6.min", min(c6), f3(min(c6)))
    put("lora.c6.max", max(c6), f3(max(c6)))
    colvals = sd[sd.collapsed][["real", "described", "content", "both", "masked", "realshuf", "maskshuf"]].values
    put("lora.collapse.min", float(colvals.min()), f3(colvals.min()))
    put("lora.collapse.max", float(colvals.max()), f3(colvals.max()))
    col = sd[sd.collapsed].iloc[0]
    put("lora.collapse.real", float(col.real), f3(col.real))
    put("lora.collapse.seed", int(col.seed), str(int(col.seed)))
    fdir = ROOT / "results/failed/amazon-grocery-gemma4-e4b-lora-masked-seed1"
    arch = sum(1 for f in (fdir / "results").glob("*.jsonl") for l in open(f) if l.strip())
    put("lora.collapse.archived.n", arch, thou(arch))
    archived_trainings = int((fdir / "adapter" / "adapter_config.json").exists())
    put("lora.ntrain.all", V["lora.ntrain"] + archived_trainings, str(V["lora.ntrain"] + archived_trainings))
    nconv = V["lora.ntrain"] + archived_trainings - V["lora.ncollapsed"] - archived_trainings
    put("lora.nconverged", nconv, str(nconv))


def revision() -> None:
    h = read(EXT / f"rev_history_{MAIN}.csv")
    hl = h[h.tag != "mostpop"]
    put("hist.ntests", len(hl), str(len(hl)))
    c6 = hl[hl.condition == "realshuf"]
    put("hist.c6.nsig", int((c6.p_holm < 0.05).sum()), str(int((c6.p_holm < 0.05).sum())))
    gr = c6[c6.dataset == "amazon-grocery"]
    put("hist.c6.gr.min", float(gr.delta_vs_random.min()), sg(gr.delta_vs_random.min()))
    put("hist.c6.gr.max", float(gr.delta_vs_random.max()), sg(gr.delta_vs_random.max()))
    put("hist.c6.gr.ciex", int((gr.ci_lo > 0).sum()), str(int((gr.ci_lo > 0).sum())))
    oth = c6[c6.dataset != "amazon-grocery"]
    put("hist.c6.oth.min", float(oth.delta_vs_random.min()), sg(oth.delta_vs_random.min()))
    put("hist.c6.oth.max", float(oth.delta_vs_random.max()), sg(oth.delta_vs_random.max()))
    put("hist.c6.oth.ciex", int(((oth.ci_lo > 0) | (oth.ci_hi < 0)).sum()), str(int(((oth.ci_lo > 0) | (oth.ci_hi < 0)).sum())))
    sig = c6[c6.p_holm < 0.05]
    put("hist.c6.sig.delta", float(sig.delta_vs_random.iloc[0]), sg(sig.delta_vs_random.iloc[0]))
    put("hist.c6.sig.p", float(sig.p_holm.iloc[0]), f"{sig.p_holm.iloc[0]:.2g}")
    put("hist.c6.sig.tag", str(sig.tag.iloc[0]) + "/" + str(sig.dataset.iloc[0]), "")
    inv = {v: k for k, v in MODELS.items()}
    invd = {v: k for k, v in DS.items()}
    for r in hl.itertuples():
        k = f"hist.{invd[r.dataset]}.{inv[r.tag]}.{'c6' if r.condition == 'realshuf' else 'c7'}"
        put(f"{k}.delta", float(r.delta_vs_random), sg(r.delta_vs_random))
        put(f"{k}.lo", float(r.ci_lo), sg(r.ci_lo))
        put(f"{k}.hi", float(r.ci_hi), sg(r.ci_hi))
        put(f"{k}.p", float(r.p_holm), f"{r.p_holm:.2g}")
    c7 = hl[hl.condition == "maskshuf"]
    put("hist.c7.vg.nsig", int(((c7.dataset == "amazon-videogames") & (c7.p_holm < 0.05) & (c7.delta_vs_random < 0)).sum()),
        str(int(((c7.dataset == "amazon-videogames") & (c7.p_holm < 0.05) & (c7.delta_vs_random < 0)).sum())))
    put("hist.c7.oth.nsig", int(((c7.dataset != "amazon-videogames") & (c7.p_holm < 0.05)).sum()),
        str(int(((c7.dataset != "amazon-videogames") & (c7.p_holm < 0.05)).sum())))
    mp = h[h.tag == "mostpop"].set_index("dataset")
    for d, ds in DS.items():
        put(f"hist.mostpop.{d}", float(mp.loc[ds, "delta_vs_random"]), sg(mp.loc[ds, "delta_vs_random"]))
    vr = read(EXT / f"rev_vsref_{MAIN}.csv")
    for d, ds in DS.items():
        for cond, k in (("real", "c0"), ("masked", "c1")):
            s = vr[(vr.dataset == ds) & (vr.reference == "sasrec") & (vr.condition == cond)]
            above = int(((s.delta > 0) & (s.p_holm < 0.05)).sum())
            below = int(((s.delta < 0) & (s.p_holm < 0.05)).sum())
            put(f"vsref.{d}.{k}.above", above, str(above))
            put(f"vsref.{d}.{k}.below", below, str(below))
            put(f"vsref.{d}.{k}.nabovemean", int((s.delta > 0).sum()), str(int((s.delta > 0).sum())))
    q = vr[(vr.dataset == "amazon-videogames") & (vr.reference == "sasrec") & (vr.condition == "real") & (vr.tag == "qwen3.8-27b")].iloc[0]
    put("vg.q38.vssasrec", float(q.delta), sg(q.delta, 4))
    put("vg.q38.vssasrec.p", float(q.p_holm), f"{q.p_holm:.2g}")
    c1b = vr[(vr.reference == "sasrec") & (vr.condition == "masked")]
    put("vsref.c1.allbelow", int(((c1b.delta < 0) & (c1b.p_holm < 0.05)).sum()), str(int(((c1b.delta < 0) & (c1b.p_holm < 0.05)).sum())))
    put("vsref.ntests", len(vr), str(len(vr)))
    ce = read(EXT / "rev_cells.csv")
    for k in ("d53", "d04", "d03"):
        g = ce[ce.effect == k].set_index("analysis")
        fe = g.loc["OLS slope with dataset fixed effects"]
        put(f"cell.fe.{k}", float(fe.estimate), f"{fe.estimate:.2f}")
        put(f"cell.fe.{k}.p", float(fe.p), f"{fe.p:.2g}")
        put(f"cell.fe.{k}.lo", float(fe.ci_lo), mn(fe.ci_lo))
        put(f"cell.fe.{k}.hi", float(fe.ci_hi), mn(fe.ci_hi))
        cen = g.loc["spearman on dataset-centred values"]
        put(f"cell.cen.{k}", float(cen.estimate), mn(cen.estimate))
        put(f"cell.cen.{k}.p", float(cen.p), f"{cen.p:.2g}")
        for d, ds in DS.items():
            w = g.loc[f"spearman within {ds}"]
            put(f"cell.within.{k}.{d}", float(w.estimate), mn(w.estimate))
            put(f"cell.within.{k}.{d}.p", float(w.p), f"{w.p:.2g}")
    for d in DS:
        for m in ("q4", "g4"):
            gap = V[f"{d}.{m}real.c0"] - V[f"{d}.{m}masked.c1"]
            put(f"{d}.{m}.ftgap", gap, sg(gap))
    gaps = [V[f"{d}.{m}.ftgap"] for d in DS for m in ("q4", "g4")]
    put("ftgap.min", min(gaps), sg(min(gaps)))
    put("ftgap.max", max(gaps), sg(max(gaps)))
    zs = [V[f"{d}.{m}ro.d01"] for d in DS for m in ("q4", "g4")]
    put("ftzs.d01.min", min(zs), sg(min(zs)))
    put("ftzs.d01.max", max(zs), sg(max(zs)))
    larger = sum(V[f"{d}.{m}.ftgap"] > V[f"{d}.{m}ro.d01"] for d in DS for m in ("q4", "g4"))
    put("ftgap.nlarger", larger, str(larger))


def revision_runs() -> None:
    A0 = "test_n500_c20_s42_a0"
    amz = ("vg", "be", "gr")
    for d, ds in DS.items():
        s = read(EXT / f"{ds}_{MAIN}_readout_summary.csv")
        c = read(EXT / f"{ds}_{MAIN}_readout_contrasts.csv")
        w = s.pivot(index="tag", columns="condition", values="ndcg@10")
        for m, tag in MODELS.items():
            for k, cond in COND.items():
                put(f"{d}.{m}.lp.{k}", float(w.loc[tag, cond]), f3(w.loc[tag, cond]))
            for k, (a, b) in CONTRAST.items():
                row = c[(c.tag == tag) & (c.contrast == f"{a}-{b}")].iloc[0]
                put(f"{d}.{m}.lp.{k}", float(row.delta), sg(row.delta))
                put(f"{d}.{m}.lp.{k}.p", float(row.p_holm), f"{row.p_holm:.2g}")
            sh = V[f"{d}.{m}.lp.d04"] / V[f"{d}.{m}.lp.d01"]
            put(f"{d}.{m}.lp.idshare", sh, pct(sh, 0))
            gs = V[f"{d}.{m}.d04"] / V[f"{d}.{m}.d01"]
            put(f"{d}.{m}.idshare", gs, pct(gs, 0))
        for k in ("d01", "d03", "d04", "d41", "d53", "d06"):
            vals = [V[f"{d}.{m}.lp.{k}"] for m in MODELS]
            put(f"{d}.lp.{k}.min", min(vals), sg(min(vals)))
            put(f"{d}.lp.{k}.max", max(vals), sg(max(vals)))
            nsig = sum(V[f"{d}.{m}.lp.{k}.p"] < 0.05 for m in MODELS)
            put(f"{d}.lp.{k}.nsig", nsig, str(nsig))
    sh = [V[f"{d}.{m}.lp.idshare"] for d in amz for m in MODELS]
    put("amz.lp.idshare.min", min(sh), pct(min(sh), 0))
    put("amz.lp.idshare.max", max(sh), pct(max(sh), 0))
    put("amz.lp.idshare.median", float(np.median(sh)), pct(float(np.median(sh)), 0))
    put("amz.lp.idshare.nbelowhalf", sum(x < 0.5 for x in sh), str(sum(x < 0.5 for x in sh)))
    gsh = [V[f"{d}.{m}.idshare"] for d in amz for m in MODELS]
    put("amz.idshare.nbelowhalf", sum(x < 0.5 for x in gsh), str(sum(x < 0.5 for x in gsh)))
    mls = [V[f"ml.{m}.lp.idshare"] for m in MODELS]
    put("ml.lp.idshare.min", min(mls), pct(min(mls), 0))
    put("ml.lp.idshare.max", max(mls), pct(max(mls), 0))
    c6 = [V[f"{d}.{m}.lp.c6"] for d in DS for m in MODELS]
    put("all.lp.c6.min", min(c6), f3(min(c6)))
    put("all.lp.c6.max", max(c6), f3(max(c6)))
    flips = sum(1 for d in DS for m in MODELS for k in ("d01", "d03", "d04", "d41", "d53", "d06")
                if V[f"{d}.{m}.{k}.p"] < 0.05 and np.sign(V[f"{d}.{m}.{k}"]) != np.sign(V[f"{d}.{m}.lp.{k}"]))
    put("lp.signflips", flips, str(flips))
    for k in ("d01", "d03", "d04", "d41", "d53"):
        dif = max(abs(V[f"{d}.{m}.lp.{k}"] - V[f"{d}.{m}.{k}"]) for d in DS for m in MODELS)
        put(f"lp.{k}.maxdiff", dif, f3(dif))
    h = read(EXT / f"rev_history_{A0}.csv")
    inv = {v: k for k, v in MODELS.items()}
    invd = {v: k for k, v in DS.items()}
    for d, ds in DS.items():
        s = read(EXT / f"{ds}_{A0}_summary.csv")
        c = read(EXT / f"{ds}_{A0}_contrasts.csv")
        w = s.pivot(index="tag", columns="condition", values="ndcg@10")
        rand = float(s[s.tag == "random"]["ndcg@10"].iloc[0])
        put(f"{d}.a0.mostpop.rel", float(w.loc["mostpop", "idonly"]) - rand, sg(float(w.loc["mostpop", "idonly"]) - rand))
        for m, tag in MODELS.items():
            for k, cond in COND.items():
                put(f"{d}.a0.{m}.{k}", float(w.loc[tag, cond]), f3(w.loc[tag, cond]))
            for k, (a, b) in CONTRAST.items():
                row = c[(c.tag == tag) & (c.contrast == f"{a}-{b}")].iloc[0]
                put(f"{d}.a0.{m}.{k}", float(row.delta), sg(row.delta))
            rec = (V[f"{d}.a0.{m}.c6"] - rand) / (V[f"{d}.a0.{m}.c0"] - rand)
            put(f"{d}.a0.{m}.c6rec", rec, pct(rec, 0))
            hh = h[(h.dataset == ds) & (h.tag == tag) & (h.condition == "realshuf")].iloc[0]
            put(f"{d}.a0.{m}.c6.delta", float(hh.delta_vs_random), sg(hh.delta_vs_random))
            put(f"{d}.a0.{m}.c6.p", float(hh.p_holm), f"{hh.p_holm:.2g}")
            h7 = h[(h.dataset == ds) & (h.tag == tag) & (h.condition == "maskshuf")].iloc[0]
            put(f"{d}.a0.{m}.c7.delta", float(h7.delta_vs_random), sg(h7.delta_vs_random))
            put(f"{d}.a0.{m}.c7.p", float(h7.p_holm), f"{h7.p_holm:.2g}")
        for k in ("c6", "c7"):
            vals = [V[f"{d}.a0.{m}.{k}.delta"] for m in MODELS]
            put(f"{d}.a0.{k}.dmin", min(vals), sg(min(vals)))
            put(f"{d}.a0.{k}.dmax", max(vals), sg(max(vals)))
            ns = sum(V[f"{d}.a0.{m}.{k}.p"] < 0.05 and V[f"{d}.a0.{m}.{k}.delta"] > 0 for m in MODELS)
            put(f"{d}.a0.{k}.nsigpos", ns, str(ns))
            ns = sum(V[f"{d}.a0.{m}.{k}.p"] < 0.05 for m in MODELS)
            put(f"{d}.a0.{k}.nsig", ns, str(ns))
        recs = [V[f"{d}.a0.{m}.c6rec"] for m in MODELS]
        put(f"{d}.a0.c6rec.max", max(recs), pct(max(recs), 0))
        nb = sum(V[f"{d}.a0.{m}.c0"] < float(w.loc["mostpop", "idonly"]) for m in MODELS)
        put(f"{d}.a0.c0.belowmostpop", nb, str(nb))
        if d in amz:
            for m in MODELS:
                shv = V[f"{d}.a0.{m}.d04"] / V[f"{d}.a0.{m}.d01"]
                put(f"{d}.a0.{m}.idshare", shv, pct(shv, 0))
    for d in amz:
        v = [V[f"{d}.a0.{m}.idshare"] for m in MODELS]
        put(f"{d}.a0.idshare.median", float(np.median(v)), pct(float(np.median(v)), 0))
        put(f"{d}.a0.idshare.min", min(v), pct(min(v), 0)), put(f"{d}.a0.idshare.max", max(v), pct(max(v), 0))
    recs = [V[f"{d}.a0.{m}.c6rec"] for d in DS for m in MODELS]
    put("a0.c6rec.max", max(recs), pct(max(recs), 0))
    put("a0.c6rec.median", float(np.median(recs)), pct(float(np.median(recs)), 0))
    mp = [V[f"{d}.a0.mostpop.rel"] for d in DS]
    put("a0.mostpop.rel.min", min(mp), sg(min(mp))), put("a0.mostpop.rel.max", max(mp), sg(max(mp)))
    c7 = [V[f"{d}.a0.{m}.c7.delta"] for d in DS for m in MODELS]
    put("a0.c7.dmin", min(c7), sg(min(c7))), put("a0.c7.dmax", max(c7), sg(max(c7)))
    put("a0.c7.nsigpos", sum(V[f"{d}.a0.c7.nsigpos"] for d in DS), str(sum(V[f"{d}.a0.c7.nsigpos"] for d in DS)))
    put("a0.c6.nsigpos", sum(V[f"{d}.a0.c6.nsigpos"] for d in DS), str(sum(V[f"{d}.a0.c6.nsigpos"] for d in DS)))
    ash = [V[f"{d}.a0.{m}.idshare"] for d in amz for m in MODELS]
    put("amz.a0.idshare.min", min(ash), pct(min(ash), 0)), put("amz.a0.idshare.max", max(ash), pct(max(ash), 0))
    put("amz.a0.idshare.median", float(np.median(ash)), pct(float(np.median(ash)), 0))
    belowmp = sum(V[f"{d}.a0.c0.belowmostpop"] for d in DS)
    put("a0.c0.belowmostpop", belowmp, str(belowmp))
    for d, ds in DS.items():
        s = read(EXT / f"{ds}_{MAIN}_baselines_summary.csv")
        c = read(EXT / f"{ds}_{MAIN}_baselines_contrasts.csv")
        w = s.pivot(index="tag", columns="condition", values="ndcg@10")
        put(f"{d}.sasrec20", float(w.loc["sasrec20", "idonly"]), f3(w.loc["sasrec20", "idonly"]))
        put(f"{d}.ease20", float(w.loc["ease20", "idonly"]), f3(w.loc["ease20", "idonly"]))
        for var, key in (("textsasrec-text", "tst"), ("textsasrec-textid", "tsi")):
            for k, cond in (("c0", "real"), ("c1", "masked"), ("c3", "described"), ("c4", "content"), ("c5", "both")):
                put(f"{d}.{key}.{k}", float(w.loc[var, cond]), f3(w.loc[var, cond]))
            for k, name in (("d01", "real-masked"), ("d03", "real-described"), ("d04", "real-content"),
                            ("d41", "content-masked"), ("d53", "both-described"), ("vss", "real-sasrec")):
                row = c[(c.tag == var) & (c.contrast == name)].iloc[0]
                put(f"{d}.{key}.{k}", float(row.delta), sg(row.delta))
                put(f"{d}.{key}.{k}.p", float(row.p_holm), f"{row.p_holm:.2g}")
            if d in amz:
                shv = V[f"{d}.{key}.d04"] / V[f"{d}.{key}.d01"]
                put(f"{d}.{key}.idshare", shv, pct(shv, 0))
        best = max(V[f"{d}.{m}.c0"] for m in MODELS)
        put(f"{d}.c0.best", best, f3(best))
        put(f"{d}.tsi.c0.vsbest", V[f"{d}.tsi.c0"] - best, sg(V[f"{d}.tsi.c0"] - best))
    nb = sum(V[f"{d}.tsi.c0"] >= V[f"{d}.c0.best"] for d in DS)
    put("tsi.c0.atleastbest", nb, str(nb))
    for key in ("tst", "tsi"):
        vals = [V[f"{d}.{key}.d03"] for d in DS]
        put(f"{key}.d03.min", min(vals), sg(min(vals))), put(f"{key}.d03.max", max(vals), sg(max(vals)))
        ns = sum(V[f"{d}.{key}.d03.p"] < 0.05 for d in DS)
        put(f"{key}.d03.nsig", ns, str(ns))
    na = sum(V[f"{d}.a0.{m}.idshare"] >= 0.5 for d in amz for m in MODELS)
    put("amz.a0.idshare.nabovehalf", na, str(na))
    v = [V[f"{d}.tsi.d01"] for d in amz]
    put("amz.tsi.d01.min", min(v), sg(min(v))), put("amz.tsi.d01.max", max(v), sg(max(v)))
    v = [V[f"{d}.tsi.c1"] - V[f"{d}.sasrec"] for d in amz]
    put("amz.tsi.c1.vss.min", min(v), sg(min(v))), put("amz.tsi.c1.vss.max", max(v), sg(max(v)))
    co = read(EXT / f"rev_catonly_{MAIN}.csv")
    for d, ds in DS.items():
        s = read(EXT / f"{ds}_{MAIN}_baselines_summary.csv")
        w = s.pivot(index="tag", columns="condition", values="ndcg@10")
        put(f"{d}.cat.tid", float(w.loc["catsasrec-textid", "catonly"]), f3(w.loc["catsasrec-textid", "catonly"]))
        put(f"{d}.cat.tt", float(w.loc["catsasrec-text", "catonly"]), f3(w.loc["catsasrec-text", "catonly"]))
        for key, name in (("vss", "catonly-sasrec"), ("d1c", "masked-catonly"), ("d0c", "real-catonly")):
            r = co[(co.dataset == ds) & (co.contrast == name)].iloc[0]
            put(f"{d}.cat.{key}", float(r.delta), sg(r.delta))
            put(f"{d}.cat.{key}.p", float(r.p_holm), f"{r.p_holm:.2g}")
    for key in ("vss", "d1c", "d0c"):
        v = [V[f"{d}.cat.{key}"] for d in amz]
        put(f"amz.cat.{key}.min", min(v), sg(min(v))), put(f"amz.cat.{key}.max", max(v), sg(max(v)))
        put(f"amz.cat.{key}.nsig", sum(V[f"{d}.cat.{key}.p"] < 0.05 for d in amz),
            str(sum(V[f"{d}.cat.{key}.p"] < 0.05 for d in amz)))
    sh = [V[f"{d}.cat.d0c"] / (V[f"{d}.tsi.c0"] - V[f"{d}.sasrec"]) for d in amz]
    put("amz.cat.nameshare.min", min(sh), pct(min(sh), 0)), put("amz.cat.nameshare.max", max(sh), pct(max(sh), 0))
    nb = sum(V[f"{d}.cat.tid"] >= V[f"{d}.c0.best"] for d in DS)
    put("cat.tid.atleastbest", nb, str(nb))
    vb = read(EXT / f"rev_vsbest_{MAIN}.csv")
    for d, ds in DS.items():
        for key, name in (("tsi.c0", "textid-real"), ("cat", "textid-catonly")):
            r = vb[(vb.dataset == ds) & (vb.reference == name)].iloc[0]
            put(f"{d}.{key}.vsbest.p", float(r.p_holm), f"{r.p_holm:.2g}")
            if key == "cat":
                put(f"{d}.cat.vsbest", float(r.delta), sg(r.delta))
    pm = min(V[f"{d}.tsi.c0.vsbest.p"] for d in amz)
    put("amz.tsi.vsbest.pmin", pm, f"{pm:.2g}")
    put("amz.tsi.vsbest.nsig", sum(V[f"{d}.tsi.c0.vsbest.p"] < 0.05 for d in amz),
        str(sum(V[f"{d}.tsi.c0.vsbest.p"] < 0.05 for d in amz)))
    for d in DS:
        for mm in ("min", "max"):
            put(f"{d}.d01.{mm}.u", abs(V[f"{d}.d01.{mm}"]), f3(abs(V[f"{d}.d01.{mm}"])))
    rd = max(abs(V[f"{d}.{m}ro.{k}"] - V[f"{d}.{m}.lp.{k}"]) for d in DS for m in ("q4", "g4")
             for k in ("c0", "c3", "c4", "c5", "c1", "c6"))
    put("ro.lp.maxdiff", rd, f3(rd))
    tsh = [V[f"{d}.tst.idshare"] for d in amz]
    put("amz.tst.idshare.min", min(tsh), pct(min(tsh), 0)), put("amz.tst.idshare.max", max(tsh), pct(max(tsh), 0))


def compute() -> None:
    from datetime import datetime as D
    def t(s):
        return D.strptime(s, "%Y-%m-%d %H:%M:%S")
    def first_last(path):
        stamps = re.findall(r"(?:INFO|WARNING|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d)", Path(path).read_text(errors="ignore"))
        return t("2026-" + stamps[0]), t("2026-" + stamps[-1])
    lanes = {}
    for f in ("grid-gpu1", "grid-gpu3d", "grid-lora-ml", "grid-lora-vg"):
        marks = re.findall(r"######## (\S+ \S+) GPU (\d): (\S+)", (ROOT / "logs" / f"{f}.log").read_text())
        lanes[f] = marks
    iv = {g: [] for g in range(4)}
    for f in sorted((ROOT / "logs").glob("vllm-gpu*-p81*.log")):
        g = int(re.search(r"gpu(\d)", f.name).group(1))
        iv[g].append(first_last(f))
    def lane_iv(name, job_prefix):
        marks = lanes[name]
        a = next(t(s) for s, g, j in marks if j.startswith(job_prefix))
        b = next(t(s) for s, g, j in marks if j == "LANE")
        g = int(next(g for s, g, j in marks if j.startswith(job_prefix)))
        return g, (a, b)
    for name in ("grid-gpu1", "grid-gpu3d", "grid-lora-ml", "grid-lora-vg"):
        g, x = lane_iv(name, "lora")
        iv[g].append(x)
    fix = (ROOT / "logs" / "grid-lora-fix.log").read_text()
    a = re.search(r"^(\d\d:\d\d:\d\d) gemma4-e4b-lora-masked: 10,000", fix, re.M).group(1)
    b = re.search(r"^(\d\d:\d\d:\d\d) done gemma4-e4b-lora-masked", fix, re.M).group(1)
    iv[3].append((t("2026-10-23 " + a), t("2026-10-23 " + b)))
    iv[2].append((t("2026-10-22 09:31:00"), t("2026-10-22 09:32:00")))
    iv[3].append((t("2026-10-22 09:33:30"), t("2026-10-22 09:36:00")))
    iv2 = {g: [] for g in range(4)}
    start = {}
    for line in (ROOT / "logs/ext/scheduler.log").read_text().splitlines():
        m = re.match(r"(\S+ \S+) (START|DONE|FAIL)\s+(\S+)(?: on GPU (\d))?", line)
        if not m:
            continue
        ts, kind, jid, g = m.groups()
        if kind == "START":
            start[jid] = (t(ts), int(g))
        elif jid in start:
            a, gg = start.pop(jid)
            end = t("2026-10-23 16:21:35") if (jid == "lora-check-qwen3.5-4b" and kind == "FAIL") else t(ts)
            iv2[gg].append((a, end))
    iv3 = {g: [] for g in range(4)}
    start, prev, shift = {}, None, 0
    from datetime import timedelta
    for line in (ROOT / "logs/rev/scheduler.log").read_text().splitlines():
        m = re.match(r"(\S+ \S+) (START|DONE|FAIL|waiting|scheduler)", line)
        if not m:
            continue
        ts = t(m.group(1))
        if prev is not None and ts < prev - timedelta(minutes=10):
            shift = 1
        prev = ts
        ts = ts + timedelta(hours=shift)
        m2 = re.match(r"\S+ \S+ (START|DONE|FAIL)\s+(\S+)(?: .*on GPU (\d))?", line)
        if not m2:
            continue
        kind, jid, g = m2.groups()
        if kind == "START":
            start[jid] = (ts, int(g))
        elif jid in start:
            a, gg = start.pop(jid)
            iv3[gg].append((a, ts))
    start2 = {}
    for line in (ROOT / "logs/rev2/scheduler.log").read_text().splitlines():
        m2 = re.match(r"(\S+ \S+) (START|DONE|FAIL)\s+(\S+)(?: .*on GPU (\d))?", line)
        if not m2:
            continue
        ts2, kind, jid, g = m2.groups()
        ts2 = t(ts2) + timedelta(hours=1)
        if kind == "START":
            start2[jid] = (ts2, int(g))
        elif jid in start2:
            a, gg = start2.pop(jid)
            from datetime import datetime as _D
            end = _D.fromtimestamp((ROOT / "logs/rev2/status" / jid).stat().st_mtime).replace(microsecond=0)
            iv3[gg].append((a, min(ts2, end + timedelta(hours=1))))
    iv3[3].append(first_last(ROOT / "logs/vllm-gpu3-p8290.log"))
    iv3[3].append((t("2026-10-24 23:52:15"), t("2026-10-24 23:55:00")))
    iv3[1].append((t("2026-10-25 01:20:00"), first_last(ROOT / "logs/vllm-gpu1-p8299.log")[1]))
    def union(xs):
        xs = sorted(xs)
        tot, cur = 0.0, None
        for a, b in xs:
            if cur and a <= cur[1]:
                cur = (cur[0], max(cur[1], b))
            else:
                if cur:
                    tot += (cur[1] - cur[0]).total_seconds()
                cur = (a, b)
        return (tot + (cur[1] - cur[0]).total_seconds()) / 3600 if cur else 0.0
    h1 = {g: union(iv[g]) for g in range(4)}
    h2 = {g: union(iv2[g]) for g in range(4)}
    h3 = {g: union(iv3[g]) for g in range(4)}
    for g in range(4):
        put(f"gpu{g}.h", h1[g] + h2[g] + h3[g], f"{h1[g] + h2[g] + h3[g]:.1f}")
    put("gpu.h.first", sum(h1.values()), f"{sum(h1.values()):.1f}")
    put("gpu.h.cont", sum(h2.values()), f"{sum(h2.values()):.1f}")
    put("gpu.h.rev", sum(h3.values()), f"{sum(h3.values()):.1f}")
    tot = sum(h1.values()) + sum(h2.values()) + sum(h3.values())
    put("gpu.h.total", tot, f"{tot:.0f}")
    parts = {k: 0 for k in ("main", "readout4", "lora", "c100", "a0", "p2", "pilot", "readout7", "a0ext")}
    n_llm = n_base = 0
    for root in (ROOT / "results/raw", ROOT / "results/ext/raw", ROOT / "results/ext/readout", ROOT / "results/ext/baselines"):
        for f in root.glob("*/*/*/*.jsonl"):
            with open(f, "rb") as fh:
                n = sum(1 for l in fh if l.strip())
            tag, stem = f.parent.name, f.parent.parent.name
            if f.parent.name in ("ease", "mostpop", "sasrec") or root.name == "baselines":
                n_base += n
                continue
            n_llm += n
            if root.name == "readout":
                parts["readout7"] += n
            elif tag.endswith("-ro"):
                parts["readout4"] += n
            elif "-lora-" in tag:
                parts["lora"] += n
            elif tag.endswith("-p2"):
                parts["p2"] += n
            elif stem == MAIN:
                parts["main"] += n
            elif stem.endswith("_a0"):
                parts["a0ext" if root.name == "raw" and root.parent.name == "ext" else "a0"] += n
            elif "_c100_" in stem:
                parts["c100"] += n
            else:
                parts["pilot"] += n
    for k, v in parts.items():
        put(f"n.rank.{k}", v, thou(v))
    assert sum(parts.values()) == n_llm
    put("n.rank.llm", n_llm, thou(n_llm))
    put("n.rank.base", n_base, thou(n_base))
    put("n.rank.all", n_llm + n_base, thou(n_llm + n_base))
    n_probe = 0
    for f in list((ROOT / "results/probes").glob("*/*/*.jsonl")) + list((ROOT / "results/ext/probes").glob("*/*/*.jsonl")):
        with open(f, "rb") as fh:
            n_probe += sum(1 for l in fh if l.strip())
    put("n.probe", n_probe, thou(n_probe))
    n_adapters = len(list((ROOT / "checkpoints").glob("*/*/adapter_config.json"))) + \
        len(list((ROOT / "checkpoints_ext").glob("*/*/adapter_config.json")))
    put("n.adapters", n_adapters, str(n_adapters))


def table_main() -> str:
    lines = []
    for d in DS:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS_NAME[d]}}} \\quad MostPop {T[f'{d}.mostpop']}, "
                     f"EASE {T[f'{d}.ease']}, SASRec {T[f'{d}.sasrec']}, random {T['random']}}}\\\\")
        for m in MODELS:
            cells = [T[f"{d}.{m}.{k}"] for k in ("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7")]
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
        if d != "gr":
            lines.append("\\midrule")
    return "\n".join(lines)


def star(p: float) -> str:
    return "" if p < 0.05 else "$^{\\circ}$"


def table_contrasts() -> str:
    lines = []
    for d in DS:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DS_NAME[d]}}}}}\\\\")
        for m in MODELS:
            cells = []
            for k in ("d01", "d03", "d04", "d41", "d53", "d12", "d06"):
                cells.append(T[f"{d}.{m}.{k}"] + star(V[f"{d}.{m}.{k}.p"]))
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
        if d != "gr":
            lines.append("\\midrule")
    return "\n".join(lines)


def table_ci() -> str:
    lines = []
    for d in DS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{DS_NAME[d]}}}}}\\\\")
        for m in MODELS:
            cells = []
            for k in ("d01", "d03", "d04", "d53", "d06"):
                cells.append(f"{T[f'{d}.{m}.{k}']} [{T[f'{d}.{m}.{k}.lo']}, {T[f'{d}.{m}.{k}.hi']}]")
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
        if d != "gr":
            lines.append("\\midrule")
    return "\n".join(lines)


def table_kdi() -> str:
    lines = []
    for d in DS:
        cells = []
        for m in MODELS:
            k01 = f"{d}.{m}.d01.kdi"
            cells.append(T[k01] if k01 in T else "--")
        lines.append(f"{DS_NAME[d]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_probes() -> str:
    lines = []
    for d in DS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{DS_NAME[d]}}} \\quad ({T[f'{d}.probe.nitems']} target items; "
                     f"year known for {T[f'{d}.probe.nyear']}; title of $\\geq$3 words: {T[f'{d}.probe.ncomp']})}}\\\\")
        for m in MODELS:
            cells = [T[f"{d}.{m}.year"], T[f"{d}.{m}.comp"], T[f"{d}.{m}.comphit"], T[f"{d}.{m}.reid"], T[f"{d}.{m}.reidc"]]
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
        if d != "gr":
            lines.append("\\midrule")
    return "\n".join(lines)


def table_data() -> str:
    rows = [
        ("Users / items", lambda d: f"{T[f'{d}.users']} / {T[f'{d}.items']}"),
        ("Interactions", lambda d: T[f"{d}.inter"]),
        ("Category labels", lambda d: T[f"{d}.ncat"]),
        ("Test users (median history length)", lambda d: f"{T[f'{d}.testusers']} ({T[f'{d}.histmed']})"),
        ("Median title length (characters)", lambda d: T[f"{d}.titlechars"]),
        ("C2: swaps within the category", lambda d: T[f"{d}.swapcat"]),
        ("C2: median popularity-percentile shift", lambda d: T[f"{d}.swappop"]),
        ("C3: unique descriptions", lambda d: T[f"{d}.descuniq"]),
        ("C3: median description length (characters)", lambda d: T[f"{d}.descchars"]),
        ("C3: descriptions with an identifying title word", lambda d: T[f"{d}.descleak"]),
        ("C4: items without generic title words", lambda d: T[f"{d}.contempty"]),
        ("C4: median share of title words kept", lambda d: T[f"{d}.contkept"]),
        ("C4: content views with an identifying title word", lambda d: T[f"{d}.contleak"]),
    ]
    return "\n".join(f"{name} & " + " & ".join(fn(d) for d in DS) + " \\\\" for name, fn in rows)


def table_lora() -> str:
    lines = []
    for d in DS:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DS_NAME[d]}}} \\quad EASE {T[f'{d}.ease']}, SASRec {T[f'{d}.sasrec']}}}\\\\")
        for m in ("q4", "g4"):
            lines.append(f"{MODEL_NAME[m]}, zero-shot & " + " & ".join(T[f"{d}.{m}ro.{k}"] for k in ("c0", "c3", "c4", "c5", "c1", "c6"))
                         + " & -- \\\\")
            for tr, label in (("real", "LoRA on names"), ("masked", "LoRA on codes")):
                cells = [f"{T[f'{d}.{m}{tr}.{k}']}$\\pm${T[f'{d}.{m}{tr}.{k}.sd']}" for k in ("c0", "c3", "c4", "c5", "c1", "c6")]
                lines.append(f"\\quad {label} & " + " & ".join(cells) + f" & {T[f'{d}.{m}{tr}.n']} \\\\")
        if d != "gr":
            lines.append("\\midrule")
    return "\n".join(lines)


def table_robust() -> str:
    rows = []
    for d in DS:
        rows.append(f"{DS_NAME[d]} & "
                    f"{T[f'{d}.c100.d03.min']} to {T[f'{d}.c100.d03.max']} ({T[f'{d}.c100.d03.nsig']}) & "
                    f"{T[f'{d}.c100.d01.min']} to {T[f'{d}.c100.d01.max']} ({T[f'{d}.c100.d01.nsig']}) & "
                    f"{T[f'{d}.a0.d03.min']} to {T[f'{d}.a0.d03.max']} ({T[f'{d}.a0.d03.nsig']}) & "
                    f"{T[f'{d}.a0.d01.min']} to {T[f'{d}.a0.d01.max']} ({T[f'{d}.a0.d01.nsig']}) & "
                    f"{T[f'{d}.p2.d03.min']} to {T[f'{d}.p2.d03.max']} \\\\")
    return "\n".join(rows)


def tex_escape(text: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def table_example() -> str:
    sys_path = str(ROOT)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from recmem.prompts import item_line
    v = pd.read_parquet(ROOT / "data/processed/amazon-grocery/views_s42.parquet").set_index("item_id")
    e = pd.read_parquet(ROOT / "data/processed/amazon-grocery/views_ext_s42.parquet").set_index("item_id")
    item = v.index[v["real"].str.startswith("Crystal Light Peach Tea Drink Mix")][0]
    cat = v.loc[item, "category"]
    rows = [("C0 Real", item_line(v.loc[item, "real"], cat)), ("C1 Masked", item_line(v.loc[item, "masked"], cat)),
            ("C2 Swapped", item_line(v.loc[item, "swapped"], cat)), ("C3 Described", item_line(v.loc[item, "described"], cat)),
            ("C4 Title content", item_line(e.loc[item, "content"], cat)),
            ("C5 Name + description", f"{e.loc[item, 'both']} [{cat}]")]
    put("example.item", item, item)
    return "\n".join(f"{name} & \\texttt{{{tex_escape(text)}}} \\\\" for name, text in rows)


def table_readout() -> str:
    lines = []
    for m in MODELS:
        cells = []
        for d in ("vg", "be", "gr"):
            cells += [T[f"{d}.{m}.c4.complete"], T[f"{d}.{m}.idshare"], T[f"{d}.{m}.lp.idshare"]]
        lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_main_lp() -> str:
    lines = []
    for i, d in enumerate(DS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DS_NAME[d]}}} \\quad random {T['random']}}}\\\\")
        for m in MODELS:
            cells = [T[f"{d}.{m}.lp.{k}"] for k in ("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7")]
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_refs() -> str:
    rows = [("SASRec (ID only)", "sasrec"), ("SASRec-20 (last 20 items)", "sasrec20"), ("EASE (ID only)", "ease"),
            ("EASE-20 (last 20 items)", "ease20")]
    lines = [f"{name} & " + " & ".join(T[f"{d}.{key}"] for d in DS) + " \\\\" for name, key in rows]
    for title, key in (("Text-aware SASRec, text only", "tst"), ("Text-aware SASRec, text + ID", "tsi")):
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for k, name in (("c0", "C0 Real"), ("c1", "C1 Masked"), ("c3", "C3 Descr."), ("c4", "C4 Content"),
                        ("c5", "C5 Name+descr.")):
            lines.append(f"\\quad {name} & " + " & ".join(T[f"{d}.{key}.{k}"] for d in DS) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textit{Ablation: category label as the only text}}\\\\")
    lines.append("\\quad Text only & " + " & ".join(T[f"{d}.cat.tt"] for d in DS) + " \\\\")
    lines.append("\\quad Text + ID & " + " & ".join(T[f"{d}.cat.tid"] for d in DS) + " \\\\")
    lines.append("\\midrule")
    lines.append("Best zero-shot LLM, C0 Real & " + " & ".join(T[f"{d}.c0.best"] for d in DS) + " \\\\")
    return "\n".join(lines)


def table_uniform() -> str:
    lines = []
    for d in DS:
        idsh = (f"{T[f'{d}.a0.idshare.median']} ({T[f'{d}.a0.idshare.min']}--{T[f'{d}.a0.idshare.max']})"
                if d != "ml" else "--")
        lines.append(f"{DS_NAME[d]} & {T[f'{d}.a0.mostpop.rel']} & {T[f'{d}.a0.c6.dmin']} to {T[f'{d}.a0.c6.dmax']} "
                     f"({T[f'{d}.a0.c6.nsigpos']}) & {T[f'{d}.a0.c6rec.max']} & {T[f'{d}.a0.c7.dmin']} to "
                     f"{T[f'{d}.a0.c7.dmax']} & {idsh} \\\\")
    return "\n".join(lines)


def table_history() -> str:
    lines = []
    for i, d in enumerate(DS):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DS_NAME[d]}}} \\quad MostPop {T[f'hist.mostpop.{d}']} "
                     f"from random}}\\\\")
        for m in MODELS:
            cells = []
            for k in ("c6", "c7"):
                h = f"hist.{d}.{m}.{k}"
                cells += [T[f"{d}.{m}.{k}"], f"{T[h + '.delta']} [{T[h + '.lo']}, {T[h + '.hi']}]", T[h + ".p"]]
            lines.append(f"{MODEL_NAME[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_cells() -> str:
    names = {"d53": "\\dd{C5}{C3}", "d04": "\\dd{C0}{C4}", "d03": "\\dd{C0}{C3}"}
    lines = []
    for k in ("d53", "d04", "d03"):
        cells = [f"{T[f'cell.rho.comp.{k}']} ({T[f'cell.rho.comp.{k}.p']})"]
        cells += [T[f"cell.within.{k}.{d}"] for d in DS]
        cells += [f"{T[f'cell.cen.{k}']} ({T[f'cell.cen.{k}.p']})",
                  f"{T[f'cell.fe.{k}']} [{T[f'cell.fe.{k}.lo']}, {T[f'cell.fe.{k}.hi']}] ({T[f'cell.fe.{k}.p']})"]
        lines.append(f"{names[k]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_compute() -> str:
    return "\n".join(f"GPU {g} & {T[f'gpu{g}.h']} \\\\" for g in range(4))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_stats()
    setup_stats()
    main_grid()
    probes()
    strata()
    robustness()
    answer_quality_all()
    lora()
    size_trends()
    revision()
    revision_runs()
    compute()
    tables = {"tab_main": table_main(), "tab_contrasts": table_contrasts(), "tab_ci": table_ci(), "tab_kdi": table_kdi(),
              "tab_probes": table_probes(), "tab_data": table_data(), "tab_lora": table_lora(),
              "tab_robust": table_robust(), "tab_compute": table_compute(),
              "tab_example": table_example(), "tab_history": table_history(), "tab_cells": table_cells(),
              "tab_readout": table_readout(), "tab_main_lp": table_main_lp(), "tab_refs": table_refs(),
              "tab_uniform": table_uniform()}
    for name, body in tables.items():
        body = "\n".join(l for l in body.splitlines() if l.strip())
        (OUT / f"{name}.tex").write_text("% generated by scripts/paper_numbers.py - do not edit\n" + body + "\n")
    with open(OUT / "numbers.tex", "w") as fh:
        fh.write("% generated by scripts/paper_numbers.py - do not edit\n")
        for k in sorted(T):
            fh.write(f"\\nvdef{{{k}}}{{{T[k]}}}\n")
    (OUT / "values.json").write_text(json.dumps({k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                                                 for k, v in V.items()}, indent=0, sort_keys=True))
    print(f"{len(V)} values, {len(tables)} tables -> {OUT}")


if __name__ == "__main__":
    main()
