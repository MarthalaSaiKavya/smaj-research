#!/usr/bin/env python
"""Build the paper's new summary figures from the raw result files.

Usage (repo root):  python analysis/make_paper_figures.py
Writes paper/figures/fig_protocol, fig_layer_profile and fig_goal2_main (.pdf + .png previews).
Palette: reference categorical slots 1-3 in fixed order (validated: all-pairs CVD dE >= 9.2, normal >= 24.0).
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # occlusion, random (same norm), color
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def layer_profile():
    v = json.loads((R / "validate.json").read_text())["check4"]
    g1 = json.loads((R / "status" / "goal1.json").read_text())["n_pass_per_layer"]
    L = np.arange(len(v["resid_swap_pct_mean"]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax)
    ax.plot(L, v["resid_swap_pct_mean"], color=C1, lw=2, marker="o", ms=4, label="full layer-output swap")
    ax.plot(L, v["mlp_swap_pct_mean"], color=C2, lw=2, marker="s", ms=4, label="MLP-output swap (one layer)")
    ax.axhline(90, color=INK2, lw=0.6, ls="--")
    ax.text(17.2, 91.5, "90% gate", ha="right", va="bottom", color=INK2, fontsize=7.5)
    best = int(np.argmax(v["resid_swap_pct_mean"]))
    ax.annotate(f"{v['resid_swap_pct_mean'][best]:.1f}% (L{best})", (best, v["resid_swap_pct_mean"][best]),
                xytext=(best + 2.2, 97), color=INK, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.text(8, v["mlp_swap_pct_mean"][8] + 4, f"max {max(v['mlp_swap_pct_mean']):.1f}%", color=INK, fontsize=7.5, ha="center")
    ax.set_xlabel("LLM layer (base$\\rightarrow$occluded swap at this layer)")
    ax.set_ylabel("gap closed (%)")
    ax.set_ylim(-3, 105)
    ax.set_xticks(range(0, 18, 2))
    ax.legend(loc="center left", bbox_to_anchor=(0.50, 0.58), fontsize=7.5, handlelength=1.6)
    ax.set_title("(a) where the occlusion signal can be swapped", fontsize=8.5, loc="left", color=INK)
    ax = axes[1]
    style(ax)
    counts = [g1[str(i)] for i in L]
    ax.bar(L, counts, width=0.72, color=C1, edgecolor="white", linewidth=0.8)
    for i in (4, 14):
        ax.text(i, counts[i] + 0.2, str(counts[i]), ha="center", va="bottom", color=INK, fontsize=7.5)
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("# selective features")
    ax.set_xticks(range(0, 18, 4))
    ax.set_ylim(0, 9.5)
    ax.set_title(f"(b) selective features ({sum(counts)} total)", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_layer_profile.{ext}", dpi=220)
    plt.close(fig)


def goal2_main():
    rows = list(csv.DictReader(open(R / "goal2_rows.csv")))

    def pf(layer, method, scope="all", direction="inject"):
        d = collections.defaultdict(list)
        for r in rows:
            if r["layer"] == str(layer) and r["method"] == method and r["scope"] == scope and r["direction"] == direction:
                d[int(r["frame"])].append(float(r["closed_pct"]))
        return np.array([np.mean(d[k]) for k in sorted(d)])

    sites = [("L0", 0), ("L4", 4), ("L5", 5), ("all 18\nlayers", -2)]
    methods = [("occlusion features", "occ_features", C1, "o"), ("random, same norm", "random_same_norm", C2, "s"), ("color features", "color_features", C3, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    style(ax)
    w = 0.25
    for mi, (lab, m, col, mk) in enumerate(methods):
        for si, (sname, l) in enumerate(sites):
            vals = pf(l, m)
            x = si + (mi - 1) * w
            ax.bar(x, vals.mean(), width=w * 0.9, color=col, edgecolor="white", linewidth=0.8, label=lab if si == 0 else None)
            ax.scatter(np.full(len(vals), x) + np.linspace(-0.05, 0.05, len(vals)), vals, s=9, color=INK, zorder=3, lw=0, marker=mk)
            if m == "occ_features":
                ax.text(x, max(vals.max(), vals.mean()) + 0.35, f"{vals.mean():.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([s for s, _ in sites])
    ax.set_ylabel("gap closed (%), base$\\rightarrow$occluded")
    ax.set_ylim(-1, 10.5)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.2)
    ax.set_title("(a) patch only the selected features (dots = frames)", fontsize=8.5, loc="left", color=INK)
    ax = axes[1]
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    items = [("whole MLP outputs", pf(-2, "mlp_swap_ceiling")), ("all 9,216 tc features", pf(-2, "all_tc_features")),
             ("72 occlusion features", pf(-2, "occ_features")), ("color features (72)", pf(-2, "color_features")),
             ("random, same norm (72)", pf(-2, "random_same_norm"))]
    cols = [INK2, INK2, C1, C3, C2]
    y = np.arange(len(items))[::-1]
    for yi, (lab, vals), col in zip(y, items, cols):
        ax.barh(yi, vals.mean(), height=0.62, color=col, edgecolor="white", linewidth=0.8)
        ax.text(vals.mean() + 1.5, yi, f"{vals.mean():.1f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in items], fontsize=7.5)
    ax.set_xlim(0, 80)
    ax.set_xlabel("gap closed (%), all 18 layers patched")
    ax.set_title("(b) the effect is distributed", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_goal2_main.{ext}", dpi=220)
    plt.close(fig)


def protocol():
    import pickle

    fr = pickle.load(open(R / "frames.pkl", "rb"))["probe"][0]
    conds = ["base", "recolor", "absent", "occluded", "slab_miss", "occluded_absent"]
    titles = ["base", "recolor", "absent", "occluded", "slab\\_miss", "occluded\\_absent"]
    fig, axes = plt.subplots(2, 6, figsize=(6.6, 2.45))
    for r, (key, cam) in enumerate((("image", "agentview"), ("image2", "wrist"))):
        for c, cond in enumerate(conds):
            ax = axes[r][c]
            ax.imshow(fr["conds"][cond][key][::-1, ::-1])  # as the policy sees it (LIBERO 180-degree flip)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(GRID)
            if r == 0:
                ax.set_title(cond.replace("_", " "), fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(cam, fontsize=8, color=INK)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_protocol.{ext}", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    protocol()
    layer_profile()
    goal2_main()
    print("wrote", sorted(p.name for p in FIG.iterdir()))
