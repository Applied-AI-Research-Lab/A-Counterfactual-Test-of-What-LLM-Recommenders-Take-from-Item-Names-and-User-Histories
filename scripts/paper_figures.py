#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "paper/generated/values.json").read_text())
OUT = ROOT / "paper/figures"
DS = [("ml", "MovieLens-1M"), ("vg", "Video Games"), ("be", "Beauty"), ("gr", "Grocery")]
MODELS = [("q4", "Qwen3.5-4B"), ("q9", "Qwen3.5-9B"), ("q27", "Qwen3.5-27B"), ("q38", "Qwen3.8-27B"),
          ("g4", "Gemma-4-E4B"), ("g12", "Gemma-4-12B"), ("g31", "Gemma-4-31B")]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0, "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_decomposition():
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.35), sharey=True)
    ys = list(range(len(MODELS)))[::-1]
    xmax = max(V[f"{d}.{m}.d01"] for d, _ in DS for m, _ in MODELS) * 1.08
    for ax, (d, name) in zip(axes, DS):
        for y, (m, _) in zip(ys, MODELS):
            idn, cont = V[f"{d}.{m}.d04"], V[f"{d}.{m}.d41"]
            ax.barh(y, idn, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
            if cont >= 0:
                ax.barh(y, cont, left=max(idn, 0), height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
            else:
                ax.barh(y, cont, left=0, height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlim(min(0, min(V[f"{d}.{m}.d41"] for m, _ in MODELS)) - 0.005, xmax)
        ax.axvline(0, color=AXIS, linewidth=0.6)
        style(ax)
        ax.set_xticks([0, 0.1, 0.2])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([n for _, n in MODELS])
    fig.supxlabel("Difference in NDCG@10 (names minus codes, C0 − C1)", fontsize=7.5, color=INK2, y=0.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["identity: brand and name words (C0 − C4)", "generic title words (C4 − C1)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.07))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_decomposition")


def fig_history():
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.35), sharey=True)
    ys = list(range(len(MODELS)))[::-1]
    for ax, (d, name) in zip(axes, DS):
        for y, (m, _) in zip(ys, MODELS):
            ax.plot([V[f"{d}.{m}.c6"], V[f"{d}.{m}.c0"]], [y + 0.12] * 2, color=GRID, linewidth=1.2, zorder=1)
            ax.plot([V[f"{d}.{m}.c7"], V[f"{d}.{m}.c1"]], [y - 0.12] * 2, color=GRID, linewidth=1.2, zorder=1)
            ax.scatter(V[f"{d}.{m}.c0"], y + 0.12, s=16, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=0.6)
            ax.scatter(V[f"{d}.{m}.c6"], y + 0.12, s=16, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.0, zorder=3)
            ax.scatter(V[f"{d}.{m}.c1"], y - 0.12, s=16, marker="s", color=ORANGE, zorder=3, edgecolor=SURFACE,
                       linewidth=0.6)
            ax.scatter(V[f"{d}.{m}.c7"], y - 0.12, s=16, marker="s", facecolor=SURFACE, edgecolor=ORANGE,
                       linewidth=1.0, zorder=3)
        ax.axvline(V["random"], color=INK2, linewidth=0.7)
        ax.text(V["random"] + 0.006, max(ys) + 0.55, "random", color=INK2, fontsize=6.5, va="bottom")
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlim(0.16, 0.64)
        ax.set_ylim(-0.6, max(ys) + 0.9)
        style(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([n for _, n in MODELS])
    fig.supxlabel("NDCG@10", fontsize=7.5, color=INK2, y=0.0)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=4.5),
         plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=BLUE, markersize=4.5),
         plt.Line2D([], [], marker="s", linestyle="", color=ORANGE, markersize=4.5),
         plt.Line2D([], [], marker="s", linestyle="", markerfacecolor=SURFACE, markeredgecolor=ORANGE, markersize=4.5)]
    fig.legend(h, ["names, own history (C0)", "names, another user's history (C6)",
                   "codes, own history (C1)", "codes, another user's history (C7)"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.07), handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_history")


def fig_knowledge():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75))
    marks = {"ml": (BLUE, "o"), "vg": (ORANGE, "^"), "be": (ORANGE, "s"), "gr": (ORANGE, "D")}
    for ax, (k, ylabel) in zip(axes, (("d53", "Name on top of the description (C5 − C3)"),
                                     ("d04", "Identity: names minus generic words (C0 − C4)"))):
        for d, name in DS:
            c, mk = marks[d]
            xs = [V[f"{d}.{m}.comp"] for m, _ in MODELS]
            ys = [V[f"{d}.{m}.{k}"] for m, _ in MODELS]
            ax.scatter(xs, ys, s=22, marker=mk, color=c, edgecolor=SURFACE, linewidth=0.6, label=name, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        ax.set_xlabel("Title-completion score (mean over target items)")
        ax.set_title(ylabel, loc="left", fontsize=7.5, fontweight="bold")
        ax.set_ylabel("Difference in NDCG@10")
        style(ax, "both")
        ax.set_xlim(0, 0.75)
        rho, p = V[f"cell.rho.comp.{k}"], V[f"cell.rho.comp.{k}.p"]
        ax.text(0.98, 0.03, f"Spearman ρ = {rho:.2f} (p = {p:.2g})", transform=ax.transAxes,
                ha="right", va="bottom", color=INK2, fontsize=6.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_knowledge")


def fig_lora():
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.6), sharey=True)
    conds = [("c0", "names\nC0"), ("c3", "descr.\nC3"), ("c4", "generic\nC4"), ("c1", "codes\nC1")]
    series = [("ro", "zero-shot", AQUA), ("real", "LoRA trained on names", BLUE), ("masked", "LoRA trained on codes", ORANGE)]
    width = 0.26
    for r, (m, mname) in enumerate((("q4", "Qwen3.5-4B"), ("g4", "Gemma-4-E4B"))):
        for c, (d, name) in enumerate(DS):
            ax = axes[r, c]
            for i, (s, label, color) in enumerate(series):
                xs = [j + (i - 1) * width for j in range(len(conds))]
                vals = [V[f"{d}.{m}{s}.{k}"] for k, _ in conds]
                err = [V[f"{d}.{m}{s}.{k}.sd"] if s != "ro" else 0 for k, _ in conds]
                ax.bar(xs, vals, width=width * 0.92, color=color, edgecolor=SURFACE, linewidth=0.6, label=label,
                       yerr=err if s != "ro" else None, error_kw={"elinewidth": 0.7, "ecolor": INK, "capsize": 0})
            ax.axhline(V[f"{d}.sasrec"], color=INK2, linewidth=0.7)
            ax.text(len(conds) - 0.55, V[f"{d}.sasrec"] + 0.008, "SASRec", ha="right", va="bottom", fontsize=6.3,
                    color=INK2)
            ax.axhline(V["random"], color=MUTED, linewidth=0.6)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([n for _, n in conds], fontsize=6.5)
            ax.set_ylim(0.15, 0.75)
            style(ax, "y")
            if r == 0:
                ax.set_title(name, loc="left", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mname}\nNDCG@10")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.04))
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, "fig_lora")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_decomposition()
    fig_history()
    fig_knowledge()
    fig_lora()
    print("figures ->", OUT)
