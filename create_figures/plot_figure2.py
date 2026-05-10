#!/usr/bin/env python3
"""Composite Figure 2: alpha sweep and norm-matched control.

The script reads the small, included data bundle under create_figures/data and
writes fig2_combined.{pdf,png} to create_figures/outputs.

Font sizes are all scaled by the single ``FONT_SCALE`` knob below; edit it
to grow / shrink every text element in the figure proportionally.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _figure_helpers import (  # noqa: E402
    _mean_std, PM_COLOR, PRM_COLOR, FIXED_ALPHA_REF,
)

DATA_ROOT = os.path.join(REPO, "create_figures", "data")
OUT_DIR = os.path.join(REPO, "create_figures", "outputs")

# One knob to scale every font in this figure. Bump up to enlarge, down to
# shrink. Applies to rcParams and every explicit ``fontsize=`` below. The
# figure width below is bumped in tandem so longer panel titles still fit
# side-by-side without colliding at this scale.
FONT_SCALE = 1.40
F7C_SWEEP_DIR = os.path.join(DATA_ROOT, "flatness", "F7c", "clip_imagenet_r_T10", "sweep")
F1B_SWEEP_DIR = os.path.join(DATA_ROOT, "flatness", "F1_baselines_clip", "imagenet_r_T10", "sweep")
NORM_MATCHED_DIR = os.path.join(DATA_ROOT, "mechanism", "norm_matched")

PRM_COLOR = PRM_COLOR
FIX_COLOR = "#444444"
LW_REF = 1.5

# Muted hues for non-(P&M, PRM) baselines so the headline P&M/PRM contrast
# stays dominant. (sweep_dir, prefix, color, label, lw, zorder).
BASELINE_OVERLAYS = [
    (F1B_SWEEP_DIR, "lora",           "#7f7f7f", "LoRA (naïve)", 1.4, 2),
    (F1B_SWEEP_DIR, "model_avg_base", "#e377c2", "Model-Avg",    1.4, 2),
    (F1B_SWEEP_DIR, "coma",           "#9467bd", "CoMA",         1.4, 2),
    (F1B_SWEEP_DIR, "cofima",         "#1f77b4", "CoFiMA",       1.4, 2),
    (F1B_SWEEP_DIR, "became",         "#2ca02c", "BECAME",       1.4, 2),
]


# --------------------------------------------------------------------------
# Panel (a): alpha-sweep (PM vs PRM)
# --------------------------------------------------------------------------

def _load_sweep_dir(sweep_dir: str, prefix: str, task_str: str = "05"):
    paths = sorted(glob.glob(os.path.join(
        sweep_dir, f"{prefix}_sweep_task{task_str}_seed*.json"
    )))
    if not paths:
        return None
    alphas = None
    L_seen, Acc_seen, theory = [], [], []
    for p in paths:
        d = json.load(open(p))
        a = np.asarray(d["alpha_grid"], dtype=float)
        if alphas is None:
            alphas = a
        L_seen.append(np.asarray(d["L_seen"], dtype=float))
        Acc_seen.append(np.asarray(d["Acc_seen"], dtype=float))
        theory.append(float(d.get("theory_alpha") or 0.0))
    return {
        "alpha": alphas,
        "L_seen": np.stack(L_seen, axis=0),
        "Acc_seen": np.stack(Acc_seen, axis=0),
        "theory_alpha": np.array(theory),
    }


def _draw_panel_a(ax_loss, ax_acc, base, prox):
    alphas = base["alpha"]
    L_b_m, L_b_s = _mean_std(base["L_seen"])
    L_p_m, L_p_s = _mean_std(prox["L_seen"])
    A_b_m, A_b_s = _mean_std(base["Acc_seen"])
    A_p_m, A_p_s = _mean_std(prox["Acc_seen"])

    base_theory = float(np.mean(base["theory_alpha"]))
    base_acc_emp = float(alphas[int(np.argmax(A_b_m))])
    prox_acc_emp = float(alphas[int(np.argmax(A_p_m))])

    # Overlay other baselines first (in background) so P&M / PRM stay on top.
    baseline_handles = []
    for sd, prefix, color, label, lw, zorder in BASELINE_OVERLAYS:
        d = _load_sweep_dir(sd, prefix, task_str="05")
        if d is None:
            continue
        L_m, L_s = _mean_std(d["L_seen"])
        A_m, A_s = _mean_std(d["Acc_seen"])
        ax_loss.plot(d["alpha"], L_m, color=color, lw=lw, zorder=zorder, alpha=0.85)
        ax_loss.fill_between(d["alpha"], L_m - L_s, L_m + L_s,
                             color=color, alpha=0.10, lw=0, zorder=zorder - 1)
        ax_acc.plot(d["alpha"], A_m, color=color, lw=lw, zorder=zorder, alpha=0.85)
        ax_acc.fill_between(d["alpha"], A_m - A_s, A_m + A_s,
                            color=color, alpha=0.10, lw=0, zorder=zorder - 1)
        baseline_handles.append(Line2D([0], [0], color=color, lw=2.0, label=label))

    # loss subpanel: P&M + PRM (foreground)
    ax_loss.plot(alphas, L_b_m, color=PM_COLOR, lw=2.6, label="P&M", zorder=4)
    ax_loss.fill_between(alphas, L_b_m - L_b_s, L_b_m + L_b_s,
                         color=PM_COLOR, alpha=0.18, lw=0, zorder=3)
    ax_loss.plot(alphas, L_p_m, color=PRM_COLOR, lw=2.8, label="PRM", zorder=5)
    ax_loss.fill_between(alphas, L_p_m - L_p_s, L_p_m + L_p_s,
                         color=PRM_COLOR, alpha=0.18, lw=0, zorder=4)
    ax_loss.set_ylabel(r"seen-task loss $\overline{L}_{\leq t}(\alpha)$")
    ax_loss.set_title("(a) α-sweep: P&M, PRM, and Stage-2 baselines  (CLIP IN-R 10T, task 5)")

    # acc subpanel
    ax_acc.plot(alphas, A_b_m, color=PM_COLOR, lw=2.6, zorder=4)
    ax_acc.fill_between(alphas, A_b_m - A_b_s, A_b_m + A_b_s,
                        color=PM_COLOR, alpha=0.18, lw=0, zorder=3)
    ax_acc.plot(alphas, A_p_m, color=PRM_COLOR, lw=2.8, zorder=5)
    ax_acc.fill_between(alphas, A_p_m - A_p_s, A_p_m + A_p_s,
                        color=PRM_COLOR, alpha=0.18, lw=0, zorder=4)
    ax_acc.set_xlabel(r"merge coefficient $\alpha$")
    ax_acc.set_ylabel(r"seen-task acc. $\overline{\mathrm{Acc}}_{\leq t}(\alpha)$")

    # vertical reference lines on both subpanels
    for ax in (ax_loss, ax_acc):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
        ax.axvline(FIXED_ALPHA_REF, color=FIX_COLOR, ls=":",
                   lw=LW_REF, alpha=0.9, zorder=1)
        ax.axvline(base_theory, color=PM_COLOR, ls="-",
                   lw=LW_REF, alpha=0.95, zorder=2)
        ax.axvline(base_acc_emp, color=PM_COLOR, ls="--",
                   lw=LW_REF, alpha=0.95, zorder=2)
        ax.axvline(prox_acc_emp, color=PRM_COLOR, ls="--",
                   lw=LW_REF, alpha=0.95, zorder=3)

    # legend on top subpanel — headline P&M / PRM first, then baselines
    curve_handles = [
        Line2D([0], [0], color=PM_COLOR,  lw=2.6, label="P&M"),
        Line2D([0], [0], color=PRM_COLOR, lw=2.8, label="PRM"),
    ] + baseline_handles
    ax_loss.legend(handles=curve_handles, loc="upper left",
                   frameon=False, fontsize=8 * FONT_SCALE, ncol=2,
                   handlelength=2.0, columnspacing=0.8)
    style_handles = [
        Line2D([0], [0], color="0.2", lw=LW_REF, ls="-", label=r"theory $\alpha$"),
        Line2D([0], [0], color="0.2", lw=LW_REF, ls="--", label=r"empirical optimum"),
        Line2D([0], [0], color=FIX_COLOR, lw=LW_REF, ls=":", label=r"fixed-$\alpha$ ref."),
    ]
    ax_acc.legend(handles=style_handles, loc="lower left",
                  frameon=False, fontsize=8 * FONT_SCALE)


# --------------------------------------------------------------------------
# Panel (b): radius mechanism
# --------------------------------------------------------------------------

VEC_ORDER_B = ["PM_raw", "PM_to_PRM_norm", "PRM_raw"]
VEC_COLOR_B = {
    "PM_raw":          "#666666",
    "PM_to_PRM_norm": "#666666",
    "PRM_raw":        PRM_COLOR,
}
VEC_LS_B = {
    "PM_raw":          "-",
    "PM_to_PRM_norm": "--",
    "PRM_raw":        "-",
}
VEC_BAR_ALPHA_B = {
    "PM_raw":          0.85,
    "PM_to_PRM_norm": 0.40,
    "PRM_raw":        0.85,
}
VEC_HATCH_B = {
    "PM_raw":          "",
    "PM_to_PRM_norm": "//",
    "PRM_raw":        "",
}
VEC_LABEL_B = {
    "PM_raw":          "P&M raw",
    "PM_to_PRM_norm": "P&M → PRM-norm",
    "PRM_raw":        "PRM raw",
}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _filter_b(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _agg_curves_b(rows, gx, vy):
    g = defaultdict(list)
    for r in rows:
        try:
            g[float(r[gx])].append(float(r[vy]))
        except Exception:
            pass
    xs = sorted(g.keys())
    return (np.array(xs),
            np.array([np.mean(g[x]) for x in xs]),
            np.array([np.std(g[x], ddof=0) for x in xs]))


def _agg_scalar_b(rows, key):
    vs = []
    for r in rows:
        try:
            vs.append(float(r[key]))
        except Exception:
            pass
    if not vs:
        return float("nan"), float("nan")
    arr = np.array(vs)
    return float(arr.mean()), float(arr.std(ddof=0))


def _draw_panel_b(ax_left, per_alpha, per_vec):
    ax_right = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)

    # bars (background, on shared α=[0,1] axis)
    bar_xs = {"PM_raw": 0.18, "PM_to_PRM_norm": 0.50, "PRM_raw": 0.82}
    for v in VEC_ORDER_B:
        sub = _filter_b(per_vec, vector=v)
        if not sub:
            continue
        m, s = _agg_scalar_b(sub, "vec_norm_F")
        ax_right.bar(bar_xs[v], m, yerr=s, width=0.10,
                     color=VEC_COLOR_B[v], hatch=VEC_HATCH_B[v],
                     edgecolor="black", linewidth=0.5, capsize=2,
                     alpha=VEC_BAR_ALPHA_B[v], zorder=1)
        ax_right.text(bar_xs[v], m, f" {m:.2f}", ha="center", va="bottom",
                      fontsize=7.5 * FONT_SCALE, color="#333333", zorder=2)

    # curves (foreground)
    for v in VEC_ORDER_B:
        sub = _filter_b(per_alpha, vector=v)
        if not sub:
            continue
        xs, m, s = _agg_curves_b(sub, "alpha", "Acc_seen")
        ax_left.plot(xs, m, lw=1.8,
                     color=VEC_COLOR_B[v], linestyle=VEC_LS_B[v],
                     label=VEC_LABEL_B[v], zorder=4)
        ax_left.fill_between(xs, m - s, m + s, alpha=0.16,
                             color=VEC_COLOR_B[v], linewidth=0, zorder=3)

    ax_left.set_xlabel(r"merge coefficient $\alpha$")
    ax_left.set_ylabel(r"seen-task acc. $\overline{\mathrm{Acc}}_{\leq t}(\alpha)$")
    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_left.set_ylim(0.55, 0.82)
    ax_left.grid(True, alpha=0.3, axis="y")
    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)
    ax_left.set_title("(b) Norm matched control   (CLIP IN-R 10T, task 5)")

    ax_right.set_ylabel(r"task-vector radius $\|\delta_t\|_F$")
    ax_right.set_ylim(0, max(13.0, ax_right.get_ylim()[1] * 1.05))

    handles, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(handles, labels, loc="lower left",
                   frameon=False, fontsize=8.5 * FONT_SCALE, handlelength=2.6)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 11 * FONT_SCALE,
        "axes.labelsize": 10 * FONT_SCALE,
        "legend.fontsize": 9 * FONT_SCALE,
        "xtick.labelsize": 9 * FONT_SCALE,
        "ytick.labelsize": 9 * FONT_SCALE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Panel (a) data: F7c PM vs PRM at task 5 ----------------------------
    pm  = _load_sweep_dir(F7C_SWEEP_DIR, "pm",        task_str="05")
    prm = _load_sweep_dir(F7C_SWEEP_DIR, "prm_fixed", task_str="05")
    if pm is None or prm is None:
        raise RuntimeError(f"Missing F7c sweeps under {F7C_SWEEP_DIR}")

    # --- Panel (b) data: norm-matched CSVs ----------------------------------
    pa = _read_csv(os.path.join(NORM_MATCHED_DIR, "per_alpha.csv"))
    pv = _read_csv(os.path.join(NORM_MATCHED_DIR, "per_vector.csv"))

    # --- Layout: 2 col, panel-a is 2-row stacked (loss/acc), panel-b spans 2 -
    fig = plt.figure(figsize=(16.0, 5.8))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1.0, 1.05],
                  height_ratios=[1.0, 1.0],
                  hspace=0.30, wspace=0.20)
    ax_la = fig.add_subplot(gs[0, 0])
    ax_aa = fig.add_subplot(gs[1, 0], sharex=ax_la)
    ax_b  = fig.add_subplot(gs[:, 1])

    _draw_panel_a(ax_la, ax_aa, pm, prm)
    _draw_panel_b(ax_b, pa, pv)

    out = os.path.join(OUT_DIR, "fig2_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    out_png = os.path.join(OUT_DIR, "fig2_combined.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2_combined -> {out}")
    print(f"  fig2_combined -> {out_png}")


if __name__ == "__main__":
    main()
