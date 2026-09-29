#!/usr/bin/env python3
"""Beautified paper Figure 1 (a/b/c) for PRM mergeability.

Outputs into paper/figures_draft/fig1{a,b,c}/.

  fig1a: Write-in path on ImageNet-R 10T at task 5.
         Two stacked panels (top=old-task loss L_<t, bottom=seen-task acc
         Acc_seen) with PM (gray) and PRM-fixed (red) curves, three-seed
         mean +/- std shading. Vertical markers for fixed alpha=0.8,
         PM Fisher alpha, PM empirical optimum alpha, PRM Fisher alpha
         (when distinct from PM), PRM empirical optimum alpha.

  fig1b: lambda_prox-by-alpha heatmap of Acc_seen at tasks 2/5/10.
         Per-task adaptive vmin/vmax based on quantiles to highlight the
         widening high-accuracy band.

  fig1c: 3-panel phase diagram (performance / flatness / vector mech).
         Cleaner colors, twin axes, consistent log-x.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(REPO, "create_figures", "data")
F1_SWEEP = os.path.join(DATA_ROOT, "flatness", "F1", "imagenet_r_T10", "sweep")
F2_ROOT = os.path.join(DATA_ROOT, "flatness", "F2")
F2P_ROOT = os.path.join(DATA_ROOT, "flatness", "F2p")
FLAT_CSV = os.path.join(DATA_ROOT, "flatness", "flatness_metrics.csv")
OUT_ROOT = os.path.join(REPO, "create_figures", "outputs")

PM_COLOR = "#5b5b5b"      # neutral gray
PRM_COLOR = "#c8243d"    # PRM red

FIXED_ALPHA_REF = 0.8


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _setup_mpl():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#dcdcdc",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_sweep_seeds(method: str, task_str: str) -> Dict[str, np.ndarray]:
    """Stack seeds for a given method/task.

    Returns dict with alpha (1D), L_seen (S, A), Acc_seen (S, A), theory_alphas (S,).
    """
    paths = sorted(glob.glob(os.path.join(F1_SWEEP, f"{method}_sweep_task{task_str}_seed*.json")))
    if not paths:
        raise FileNotFoundError(f"No sweep files for {method} task {task_str}")
    alphas = None
    L_seen, Acc_seen, theory = [], [], []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        a = np.asarray(d["alpha_grid"], dtype=float)
        if alphas is None:
            alphas = a
        else:
            assert np.allclose(alphas, a), f"alpha grid mismatch in {p}"
        L_seen.append(np.asarray(d["L_seen"], dtype=float))
        Acc_seen.append(np.asarray(d["Acc_seen"], dtype=float))
        theory.append(float(d["theory_alpha"]))
    return {
        "alpha": alphas,
        "L_seen": np.stack(L_seen, axis=0),
        "Acc_seen": np.stack(Acc_seen, axis=0),
        "theory_alpha": np.array(theory),
    }


def _mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0, ddof=0)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    try:
        return float(x) if x not in (None, "", "nan") else float("nan")
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Fig 1a
# ---------------------------------------------------------------------------

def make_fig1a(task_str: str = "05", out_dir: str = None):
    """Write-in path: PM vs PRM-fixed at task 5 on ImageNet-R 10T."""
    out_dir = out_dir or os.path.join(OUT_ROOT, "fig1a")
    os.makedirs(out_dir, exist_ok=True)

    pm = _load_sweep_seeds("pm", task_str)
    prm = _load_sweep_seeds("prm_fixed", task_str)
    alphas = pm["alpha"]

    L_pm_m, L_pm_s = _mean_std(pm["L_seen"])
    L_fm_m, L_fm_s = _mean_std(prm["L_seen"])
    A_pm_m, A_pm_s = _mean_std(pm["Acc_seen"])
    A_fm_m, A_fm_s = _mean_std(prm["Acc_seen"])

    # Theory alphas (mean across seeds)
    pm_theory = float(np.mean(pm["theory_alpha"]))
    prm_theory = float(np.mean(prm["theory_alpha"]))

    # Empirical optima from the (mean) curves (Seen data)
    pm_loss_emp = float(alphas[int(np.argmin(L_pm_m))])
    prm_loss_emp = float(alphas[int(np.argmin(L_fm_m))])
    pm_acc_emp = float(alphas[int(np.argmax(A_pm_m))])
    prm_acc_emp = float(alphas[int(np.argmax(A_fm_m))])

    # Whether to draw the PRM Fisher line independently from the P&M Fisher line.
    show_prm_theory_distinct = abs(prm_theory - pm_theory) > 0.015

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )
    ax_l, ax_a = axes

    # --- Top: old-task loss L_<t(alpha) ---
    ax_l.plot(alphas, L_pm_m, color=PM_COLOR, lw=2.4)
    ax_l.fill_between(alphas, L_pm_m - L_pm_s, L_pm_m + L_pm_s,
                      color=PM_COLOR, alpha=0.18, lw=0)
    ax_l.plot(alphas, L_fm_m, color=PRM_COLOR, lw=2.4)
    ax_l.fill_between(alphas, L_fm_m - L_fm_s, L_fm_m + L_fm_s,
                      color=PRM_COLOR, alpha=0.18, lw=0)
    ax_l.set_ylabel(r"seen-task loss $\overline{L}_{\leq t}(\alpha)$")

    # --- Bottom: seen-task accuracy ---
    ax_a.plot(alphas, A_pm_m, color=PM_COLOR, lw=2.4)
    ax_a.fill_between(alphas, A_pm_m - A_pm_s, A_pm_m + A_pm_s,
                      color=PM_COLOR, alpha=0.18, lw=0)
    ax_a.plot(alphas, A_fm_m, color=PRM_COLOR, lw=2.4)
    ax_a.fill_between(alphas, A_fm_m - A_fm_s, A_fm_m + A_fm_s,
                      color=PRM_COLOR, alpha=0.18, lw=0)
    ax_a.set_xlabel(r"merge coefficient $\alpha$")
    ax_a.set_ylabel(r"seen-task accuracy $\overline{\mathrm{Acc}}_{\leq t}(\alpha)$")

    # --- Vertical alpha markers ---
    # Convention (consistent across PM and PRM):
    #   Fisher alpha  -> solid line
    #   empirical opt -> dashed line
    #   fixed alpha   -> dotted, neutral gray
    # Color encodes method (PM = gray, PRM = blue/red curve color).
    LW_REF = 1.5
    FIX_COLOR = "#444444"

    def _draw(ax, emp_pm, emp_prm):
        # Fixed-alpha reference (method-agnostic, both methods would use this)
        ax.axvline(FIXED_ALPHA_REF, color=FIX_COLOR, linestyle=":",
                   lw=LW_REF, alpha=0.9, zorder=1)
        # PM markers
        ax.axvline(pm_theory, color=PM_COLOR, linestyle="-",
                   lw=LW_REF, alpha=0.95, zorder=2)
        ax.axvline(emp_pm,    color=PM_COLOR, linestyle="--",
                   lw=LW_REF, alpha=0.95, zorder=2)
        # PRM markers
        if show_prm_theory_distinct:
            ax.axvline(prm_theory, color=PRM_COLOR, linestyle="-",
                       lw=LW_REF, alpha=0.95, zorder=2)
        ax.axvline(emp_prm, color=PRM_COLOR, linestyle="--",
                   lw=LW_REF, alpha=0.95, zorder=2)

    _draw(ax_l, pm_loss_emp, prm_loss_emp)
    _draw(ax_a, pm_acc_emp,  prm_acc_emp)

    # ----- Per-line text labels (next to vertical lines) -----
    # We label *which kind* of alpha each line is ("theory" / "optimal" /
    # "fixed"), plus the numeric value. Method (PM vs PRM) is conveyed by
    # color, so we don't repeat it in the label text.
    #
    # Y-banding strategy (avoids overlap when PM/PRM alphas are close):
    #   theory   labels -> upper band
    #   optimal  labels -> mid band
    #   fixed    label  -> lower band
    # PM labels are placed on the LEFT side of the line; PRM on the RIGHT
    # so when both theory lines fall at almost the same alpha, the texts
    # land on opposite sides and remain readable.

    # First, give the top of each panel headroom for labels.
    for ax in (ax_l, ax_a):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.45 * (ymax - ymin))

    def _annot(ax, x, color, text, y_frac, side):
        ymin, ymax = ax.get_ylim()
        y = ymin + y_frac * (ymax - ymin)
        if side == "right":
            ax.text(x + 0.006, y, text, color=color, fontsize=8.0,
                    rotation=90, ha="left", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=0.6))
        else:
            ax.text(x - 0.006, y, text, color=color, fontsize=8.0,
                    rotation=90, ha="right", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=0.6))

    Y_TOP = 0.97
    Y_MID = 0.72   # secondary band when alphas are too close to share row

    def _annotate_panel(ax, emp_pm, emp_prm):
        # Detect collision: when two label x's are within ~0.04 in alpha,
        # push PRM to a secondary y-band so its rotated text doesn't sit
        # on top of the PM label.
        def _collide(a, b):
            return abs(a - b) < 0.04

        prm_thy_y = Y_TOP
        if show_prm_theory_distinct and _collide(pm_theory, prm_theory):
            prm_thy_y = Y_MID
        prm_opt_y = Y_TOP
        if _collide(emp_pm, emp_prm):
            prm_opt_y = Y_MID

        # Draw PRM first, PM last, so PM (gray) labels sit on top of any
        # PRM bbox in case of residual overlap.
        if show_prm_theory_distinct:
            _annot(ax, prm_theory, PRM_COLOR,
                   rf"Fisher $\alpha\!=\!{prm_theory:.2f}$",
                   prm_thy_y, side="right")
        _annot(ax, emp_prm, PRM_COLOR,
               rf"optimal $\alpha\!=\!{emp_prm:.2f}$",
               prm_opt_y, side="right")
        _annot(ax, FIXED_ALPHA_REF, FIX_COLOR,
               rf"fixed $\alpha\!=\!{FIXED_ALPHA_REF:.2f}$",
               Y_TOP, side="left")
        _annot(ax, pm_theory, PM_COLOR,
               rf"Fisher $\alpha\!=\!{pm_theory:.2f}$",
               Y_TOP, side="right")
        _annot(ax, emp_pm, PM_COLOR,
               rf"optimal $\alpha\!=\!{emp_pm:.2f}$",
               Y_TOP, side="left")

    _annotate_panel(ax_l, pm_loss_emp, prm_loss_emp)
    _annotate_panel(ax_a, pm_acc_emp,  prm_acc_emp)

    # ----- Legends -----
    # Top: curves only
    curve_handles = [
        Line2D([0], [0], color=PM_COLOR,   lw=2.4, label="P&M"),
        Line2D([0], [0], color=PRM_COLOR, lw=2.4, label="PRM"),
    ]
    ax_l.legend(handles=curve_handles, loc="upper left", frameon=False,
                fontsize=10)

    # Bottom: line-style legend (linestyle = role; color = method).
    legend_handles = [
        Line2D([0], [0], color="0.2", lw=LW_REF, ls="-",
               label=r"Fisher $\alpha$"),
        Line2D([0], [0], color="0.2", lw=LW_REF, ls="--",
               label=r"empirical optimum"),
        Line2D([0], [0], color=FIX_COLOR, lw=LW_REF, ls=":",
               label=r"fixed-$\alpha$ reference"),
    ]
    ax_a.legend(handles=legend_handles, loc="lower left", frameon=False,
                fontsize=9, ncol=1, handlelength=2.6)

    # Cosmetic
    for ax in (ax_l, ax_a):
        ax.set_xlim(alphas[0], alphas[-1])
        ax.grid(True, which="both", alpha=0.45)

    fig.suptitle(rf"Write-in path on ImageNet-R 10T (task {int(task_str)}, "
                 r"3 seeds, mean$\pm$std)",
                 y=0.995, fontsize=11)

    out_pdf = os.path.join(out_dir, f"fig1a_writein_task{task_str}.pdf")
    out_png = os.path.join(out_dir, f"fig1a_writein_task{task_str}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig1a] wrote {out_pdf} / .png")

    # R_t per-seed: max - min over alpha-grid; aggregate as mean +- std (3 seeds).
    R_pm_per_seed = pm["Acc_seen"].max(axis=1) - pm["Acc_seen"].min(axis=1)
    R_fm_per_seed = prm["Acc_seen"].max(axis=1) - prm["Acc_seen"].min(axis=1)
    R_pm_mean, R_pm_std = float(R_pm_per_seed.mean()), float(R_pm_per_seed.std(ddof=0))
    R_fm_mean, R_fm_std = float(R_fm_per_seed.mean()), float(R_fm_per_seed.std(ddof=0))
    print(f"[fig1a] task {task_str}  R_t  PM = {R_pm_mean:.4f} +/- {R_pm_std:.4f}"
          f"   PRM = {R_fm_mean:.4f} +/- {R_fm_std:.4f}"
          f"   delta = {R_pm_mean - R_fm_mean:+.4f}"
          f"   ratio PRM/PM = {R_fm_mean / max(R_pm_mean, 1e-12):.3f}")

    summary = {
        "task": task_str,
        "seeds": int(pm["L_seen"].shape[0]),
        "pm_theory_alpha": pm_theory,
        "prm_theory_alpha": prm_theory,
        "pm_loss_argmin_alpha": pm_loss_emp,
        "prm_loss_argmin_alpha": prm_loss_emp,
        "pm_acc_argmax_alpha": pm_acc_emp,
        "prm_acc_argmax_alpha": prm_acc_emp,
        "pm_loss_min": float(L_pm_m.min()),
        "prm_loss_min": float(L_fm_m.min()),
        "pm_acc_max": float(A_pm_m.max()),
        "prm_acc_max": float(A_fm_m.max()),
        "pm_R_t_mean": R_pm_mean,
        "pm_R_t_std": R_pm_std,
        "prm_R_t_mean": R_fm_mean,
        "prm_R_t_std": R_fm_std,
        "pm_R_t_per_seed": [float(x) for x in R_pm_per_seed],
        "prm_R_t_per_seed": [float(x) for x in R_fm_per_seed],
    }
    with open(os.path.join(out_dir, f"fig1a_summary_task{task_str}.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Fig 1a -- compact (accuracy-only, wide-short for paper Figure 1 row)
# ---------------------------------------------------------------------------

def make_fig1a_compact(task_str: str = "05", out_dir: str = None,
                       aspect: float = 3.05):
    """Compact accuracy-only write-in path.

    The full make_fig1a stacks a loss panel above an accuracy panel
    (figsize 7.4x6.8, aspect ~1:1). For paper Figure 1 we sit it next to
    fig1b which is a 3-panel heatmap row at aspect ~3.4:1; matching
    heights requires a wider/shorter (a). This compact variant drops the
    loss panel (the figure caption only discusses accuracy) and renders
    a single panel at the requested aspect, so it can be placed at width
    \\(\\approx 0.49\\,\\text{\\textwidth}\\) alongside fig1b at
    \\(\\approx 0.49\\,\\text{\\textwidth}\\) with visually aligned heights.
    """
    out_dir = out_dir or os.path.join(OUT_ROOT, "fig1a")
    os.makedirs(out_dir, exist_ok=True)

    pm = _load_sweep_seeds("pm", task_str)
    prm = _load_sweep_seeds("prm_fixed", task_str)
    alphas = pm["alpha"]

    A_pm_m, A_pm_s = _mean_std(pm["Acc_seen"])
    A_fm_m, A_fm_s = _mean_std(prm["Acc_seen"])

    pm_theory = float(np.mean(pm["theory_alpha"]))
    prm_theory = float(np.mean(prm["theory_alpha"]))
    pm_acc_emp = float(alphas[int(np.argmax(A_pm_m))])
    prm_acc_emp = float(alphas[int(np.argmax(A_fm_m))])
    show_prm_theory_distinct = abs(prm_theory - pm_theory) > 0.015

    # Width fixed; height derived from aspect.
    W = 7.6
    H = W / aspect
    fig, ax = plt.subplots(figsize=(W, H))

    ax.plot(alphas, A_pm_m, color=PM_COLOR, lw=2.0)
    ax.fill_between(alphas, A_pm_m - A_pm_s, A_pm_m + A_pm_s,
                    color=PM_COLOR, alpha=0.18, lw=0)
    ax.plot(alphas, A_fm_m, color=PRM_COLOR, lw=2.0)
    ax.fill_between(alphas, A_fm_m - A_fm_s, A_fm_m + A_fm_s,
                    color=PRM_COLOR, alpha=0.18, lw=0)

    LW_REF = 1.2
    FIX_COLOR = "#444444"
    ax.axvline(FIXED_ALPHA_REF, color=FIX_COLOR, ls=":", lw=LW_REF, alpha=0.9, zorder=1)
    ax.axvline(pm_theory,    color=PM_COLOR,   ls="-",  lw=LW_REF, alpha=0.95, zorder=2)
    ax.axvline(pm_acc_emp,   color=PM_COLOR,   ls="--", lw=LW_REF, alpha=0.95, zorder=2)
    if show_prm_theory_distinct:
        ax.axvline(prm_theory, color=PRM_COLOR, ls="-", lw=LW_REF, alpha=0.95, zorder=2)
    ax.axvline(prm_acc_emp, color=PRM_COLOR, ls="--", lw=LW_REF, alpha=0.95, zorder=2)

    # Make headroom for compact rotated alpha labels.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.30 * (ymax - ymin))

    def _annot(x, color, text, side):
        ymin, ymax = ax.get_ylim()
        y = ymin + 0.97 * (ymax - ymin)
        dx = 0.005
        if side == "right":
            ax.text(x + dx, y, text, color=color, fontsize=7.2,
                    rotation=90, ha="left", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=0.4))
        else:
            ax.text(x - dx, y, text, color=color, fontsize=7.2,
                    rotation=90, ha="right", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=0.4))

    if show_prm_theory_distinct:
        _annot(prm_theory, PRM_COLOR,
               rf"Fisher $\alpha\!=\!{prm_theory:.2f}$", side="right")
    _annot(prm_acc_emp, PRM_COLOR,
           rf"optimal $\alpha\!=\!{prm_acc_emp:.2f}$", side="right")
    _annot(FIXED_ALPHA_REF, FIX_COLOR,
           rf"fixed $\alpha\!=\!{FIXED_ALPHA_REF:.2f}$", side="left")
    _annot(pm_theory, PM_COLOR,
           rf"Fisher $\alpha\!=\!{pm_theory:.2f}$", side="right")
    _annot(pm_acc_emp, PM_COLOR,
           rf"optimal $\alpha\!=\!{pm_acc_emp:.2f}$", side="left")

    ax.set_xlabel(r"merge coefficient $\alpha$")
    ax.set_ylabel(r"$\overline{\mathrm{Acc}}_{\leq t}(\alpha)$")
    ax.set_xlim(alphas[0], alphas[-1])
    ax.grid(True, which="both", alpha=0.45)

    curve_handles = [
        Line2D([0], [0], color=PM_COLOR,   lw=2.0, label="P&M"),
        Line2D([0], [0], color=PRM_COLOR, lw=2.0, label="PRM"),
    ]
    ax.legend(handles=curve_handles, loc="lower left",
              frameon=False, fontsize=8.5)

    fig.tight_layout(pad=0.4)
    out_pdf = os.path.join(out_dir, f"fig1a_writein_task{task_str}_compact.pdf")
    out_png = os.path.join(out_dir, f"fig1a_writein_task{task_str}_compact.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig1a-compact] wrote {out_pdf} / .png")


# ---------------------------------------------------------------------------
# Fig 1b
# ---------------------------------------------------------------------------

def make_fig1b(out_dir: str = None, compact: bool = False):
    """lambda x alpha heatmap of seen-task accuracy at tasks 2 / 5 / 10.

    compact=True drops the suptitle and trims the panel height; the
    LaTeX subcaption already names the figure when used as Fig 1(b).
    Output goes to fig1b_lambda_alpha_heatmap_compact.{pdf,png}.
    """
    out_dir = out_dir or os.path.join(OUT_ROOT, "fig1b")
    os.makedirs(out_dir, exist_ok=True)

    by_task: Dict[int, Dict[float, Dict[float, List[float]]]] = {}
    for p in sorted(glob.glob(os.path.join(F2_ROOT, "**", "*_sweep_task*_seed*.json"),
                              recursive=True)):
        with open(p) as f:
            d = json.load(f)
        lam = _to_float(d.get("lambda_prox"))
        task = int(d.get("task_id_0indexed", -1))
        alphas = d.get("alpha_grid") or []
        accs = d.get("Acc_seen") or []
        if not alphas or not accs:
            continue
        by_task.setdefault(task, {}).setdefault(lam, {})
        for a, acc in zip(alphas, accs):
            by_task[task][lam].setdefault(float(a), []).append(float(acc))
    if not by_task:
        raise RuntimeError("no F2 sweep JSONs found")

    tasks = sorted(by_task.keys())
    # Compact mode is used in paper Figure 1 alongside fig1a (which
    # keeps its loss+accuracy stacked layout). We size each task panel
    # near-square (per-panel aspect ~1:1) and drop the suptitle (the
    # LaTeX subcaption names the figure as Fig 1(b)).  Three near-square
    # task panels in a row -> overall aspect ~3.0:1.
    panel_h = 4.4 if compact else 3.7
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), panel_h),
                             squeeze=False)

    # PRM-themed sequential colormap (cool to PRM-red)
    prm_cmap = LinearSegmentedColormap.from_list(
        "prm_seq",
        ["#1b1c4a", "#3450a4", "#5dadc1", "#f7d488", "#e87b3e", PRM_COLOR],
        N=256,
    )

    for ax_idx, (ax, task) in enumerate(zip(axes[0], tasks)):
        lams = sorted(by_task[task].keys())
        alphas = sorted({a for lam in lams for a in by_task[task][lam]})
        Z = np.full((len(lams), len(alphas)), np.nan)
        for i, lam in enumerate(lams):
            for j, a in enumerate(alphas):
                vals = by_task[task][lam].get(a, [])
                if vals:
                    Z[i, j] = float(np.mean(vals))

        # Use percentile-based clipping so the contrast highlights the band.
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vmin = np.percentile(finite, 10)
            vmax = np.percentile(finite, 99.5)
            # ensure at least a small dynamic range
            if vmax - vmin < 1e-3:
                vmax = vmin + 1e-3

        # Use pcolormesh so x is alpha (continuous) and y is row index (categorical).
        x_edges = np.concatenate([
            [alphas[0] - (alphas[1] - alphas[0]) / 2],
            (np.array(alphas[:-1]) + np.array(alphas[1:])) / 2,
            [alphas[-1] + (alphas[-1] - alphas[-2]) / 2],
        ])
        y_edges = np.arange(len(lams) + 1)
        im = ax.pcolormesh(
            x_edges, y_edges, Z, cmap=prm_cmap,
            vmin=vmin, vmax=vmax, shading="flat",
        )
        ax.set_xlabel(r"merge coefficient $\alpha$")
        if ax_idx == 0:
            ax.set_ylabel(r"$\lambda_{\mathrm{prox}}$")
        ax.set_yticks(np.arange(len(lams)) + 0.5)
        # Clean tick labels (manual mapping for log-spaced lambdas)
        ytls = []
        for l in lams:
            if l == 0:
                ytls.append(r"$0$")
            else:
                e = int(round(np.log10(l)))
                m = l / (10 ** e)
                if abs(m - 1) < 1e-6:
                    ytls.append(rf"$10^{{{e}}}$")
                else:
                    ytls.append(rf"${m:.0f}\!\times\!10^{{{e}}}$")
        ax.set_yticklabels(ytls, fontsize=9)
        ax.set_title(f"task {task + 1}")
        ax.grid(False)
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("seen-task accuracy", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if not compact:
        fig.suptitle(
            r"$\overline{\mathrm{Acc}}_{\leq t}(\alpha)$ over $(\lambda_{\mathrm{prox}}, \alpha)$"
            " (ImageNet-R 10T, PRM-fixed)",
            fontsize=11, y=1.02,
        )
    fig.tight_layout(pad=0.4 if compact else 1.08)
    suffix = "_compact" if compact else ""
    out_pdf = os.path.join(out_dir, f"fig1b_lambda_alpha_heatmap{suffix}.pdf")
    out_png = os.path.join(out_dir, f"fig1b_lambda_alpha_heatmap{suffix}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig1b{suffix}] wrote {out_pdf} / .png")


# ---------------------------------------------------------------------------
# Fig 1c
# ---------------------------------------------------------------------------

def _load_final_runs(dirpath: str, method_prefix: str = "prm_fixed_lam") -> List[Dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        base = os.path.basename(p)
        m = re.search(rf"{method_prefix}(?P<lam>[^_]+)_s(?P<seed>\d+)", base)
        if not m:
            continue
        lam_str = m.group("lam")
        try:
            lam = 0.0 if lam_str == "0" else float(lam_str)
        except ValueError:
            continue
        out.append({
            "lambda_prox": lam,
            "seed": int(m.group("seed")),
            "AAA": _to_float(d.get("AAA")),
            "last_acc": _to_float(d.get("last_acc")),
            "forgetting": _to_float(d.get("forgetting")),
        })
    return out


def _agg_perf(final_runs: List[Dict]):
    by = defaultdict(lambda: defaultdict(list))
    for r in final_runs:
        for k in ("AAA", "last_acc", "forgetting"):
            by[r["lambda_prox"]][k].append(r[k])
    out = {}
    for lam, d in by.items():
        out[lam] = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in d.items()}
    return out


# --- Aggregation method (a): per-seed task-mean, then mean/std across seeds ---
# The (lam, task, seed) -> value tensor is collapsed in two stages:
#   stage 1: average across tasks within each (lam, seed) group
#   stage 2: mean / std across the per-seed scalars
# This way the error bar reflects only seed noise, not the (much larger)
# task-spread of these flatness/vector metrics.

def _per_seed_task_mean(per_lst: Dict[float, Dict[int, Dict[int, float]]]
                        ) -> Dict[float, Tuple[float, float]]:
    """Input shape: {lam: {seed: {task: value}}}.

    For each lam, average across tasks within each seed, then take
    mean / std across seeds.
    """
    out: Dict[float, Tuple[float, float]] = {}
    for lam, seed_dict in per_lst.items():
        per_seed = []
        for _seed, task_dict in seed_dict.items():
            vals = [v for v in task_dict.values() if np.isfinite(v)]
            if vals:
                per_seed.append(float(np.mean(vals)))
        if per_seed:
            arr = np.asarray(per_seed, dtype=float)
            out[lam] = (float(np.mean(arr)), float(np.std(arr, ddof=0)))
        else:
            out[lam] = (float("nan"), float("nan"))
    return out


def _source_matches(path: str, rule: str) -> bool:
    """Match rows generated from the lambda-sweep roots used in Figure 3."""
    if rule == "fixed":
        return "/F2/" in path or path.startswith("results/flatness/F2/")
    if rule == "theory":
        return "/F2p/" in path or path.startswith("results/flatness/F2p/")
    return True


def _agg_flatness(rows: List[Dict[str, str]],
                  method_filter: str = "prm_fixed",
                  rule: str = "fixed"):
    """Per-seed task-mean aggregation for the CSV-stored flatness keys."""
    keys = ["R_t", "W_0p1", "W_0p25", "W_0p5", "q_t", "delta_norm_F"]
    # Bucket rows: {key: {lam: {seed: {task: value}}}}.
    buckets: Dict[str, Dict[float, Dict[int, Dict[int, float]]]] = {
        k: defaultdict(lambda: defaultdict(dict)) for k in keys
    }
    for r in rows:
        if not r.get("method", "").startswith(method_filter):
            continue
        if not _source_matches(r.get("source_path", ""), rule):
            continue
        lam = _to_float(r.get("lambda_prox"))
        seed = int(_to_float(r.get("seed")))
        task = int(_to_float(r.get("task_id")))
        for k in keys:
            v = _to_float(r.get(k))
            if np.isfinite(v):
                # If the same (lam, seed, task) appears more than once
                # (shouldn't, with rule-filtering), keep the first.
                buckets[k][lam][seed].setdefault(task, v)
    out: Dict[float, Dict[str, Tuple[float, float]]] = {}
    for k in keys:
        agg = _per_seed_task_mean(buckets[k])
        for lam, ms in agg.items():
            out.setdefault(lam, {})[k] = ms
    return out


def _compute_W_eps_from_sweep(f2_root: str, eps_list,
                              method_filter: str = "prm_fixed"):
    """For each eps, compute W_{t,eps} per sweep JSON and aggregate by
    lambda using per-seed task-mean (then mean/std across seeds).

    Returns {eps: {lambda: (mean, std)}}.
    """
    # {eps: {lam: {seed: {task: width}}}}
    by_eps: Dict[float, Dict[float, Dict[int, Dict[int, float]]]] = {
        float(e): defaultdict(lambda: defaultdict(dict)) for e in eps_list
    }
    paths = sorted(glob.glob(os.path.join(f2_root, "**",
                                          "*_sweep_task*_seed*.json"),
                             recursive=True))
    for p in paths:
        base = os.path.basename(p)
        if method_filter not in base:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        lam = _to_float(d.get("lambda_prox"))
        # seed / task from filename (canonical: *_task{NN}_seed{N}.json)
        m = re.search(r"task(?P<task>\d+)_seed(?P<seed>\d+)\.json$", base)
        if not m:
            continue
        seed = int(m.group("seed"))
        task = int(m.group("task"))
        alphas = d.get("alpha_grid") or []
        accs = d.get("Acc_seen") or []
        if not alphas or not accs or len(alphas) != len(accs):
            continue
        a_arr = np.asarray(alphas, dtype=float)
        acc_arr = np.asarray(accs, dtype=float)
        a_max = float(np.max(acc_arr))
        if len(a_arr) >= 2:
            d_alpha = (a_arr[-1] - a_arr[0]) / (len(a_arr) - 1)
        else:
            d_alpha = 0.0
        for e in eps_list:
            hits = int(np.sum(acc_arr >= a_max - float(e)))
            by_eps[float(e)][lam][seed].setdefault(task, hits * d_alpha)
    return {e: _per_seed_task_mean(buckets) for e, buckets in by_eps.items()}


def _agg_q_dir(rows, method_filter: str = "prm_fixed",
               rule: str = "fixed",
               eps_div: float = 1e-8):
    """q_t^dir = q_t / (||delta||_F^2 + eps_div); aggregated with method (a).

    Returns {lambda: (mean, std)}.
    """
    buckets: Dict[float, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in rows:
        if not r.get("method", "").startswith(method_filter):
            continue
        if not _source_matches(r.get("source_path", ""), rule):
            continue
        lam = _to_float(r.get("lambda_prox"))
        seed = int(_to_float(r.get("seed")))
        task = int(_to_float(r.get("task_id")))
        q = _to_float(r.get("q_t"))
        d = _to_float(r.get("delta_norm_F"))
        if not (np.isfinite(q) and np.isfinite(d)):
            continue
        buckets[lam][seed].setdefault(task, float(q / (d * d + eps_div)))
    return _per_seed_task_mean(buckets)


def _compute_rule_gap(f2_root: str, rule: str = "fixed",
                      alpha_fixed: float = 0.8):
    """G_t(r) = max_alpha Acc_seen(alpha) - Acc_seen(alpha_r), aggregated by
    lambda using per-seed task-mean.

    rule = "fixed"  -> alpha_r = alpha_fixed (default 0.8)
    rule = "theory" -> alpha_r = each JSON's theory_alpha
    """
    buckets: Dict[float, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict))
    paths = sorted(glob.glob(os.path.join(f2_root, "**",
                                          "*_sweep_task*_seed*.json"),
                             recursive=True))
    for p in paths:
        base = os.path.basename(p)
        m = re.search(r"task(?P<task>\d+)_seed(?P<seed>\d+)\.json$", base)
        if not m:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        lam = _to_float(d.get("lambda_prox"))
        seed = int(m.group("seed"))
        task = int(m.group("task"))
        alphas = d.get("alpha_grid") or []
        accs = d.get("Acc_seen") or []
        if not alphas or not accs or len(alphas) != len(accs):
            continue
        a_arr = np.asarray(alphas, dtype=float)
        acc_arr = np.asarray(accs, dtype=float)
        if rule == "theory":
            target = _to_float(d.get("theory_alpha"))
            if not np.isfinite(target):
                continue
        else:
            target = alpha_fixed
        idx = int(np.argmin(np.abs(a_arr - target)))
        gap = float(np.max(acc_arr) - acc_arr[idx])
        buckets[lam][seed].setdefault(task, gap)
    return _per_seed_task_mean(buckets)


def make_fig1c(out_dir: str = None, rule: str = "fixed",
               data_root: str = None, method_prefix: str = None,
               method_filter: str = None,
               variant_tag: str = None,
               eps_list=(0.10, 0.25, 0.50),
               add_q_dir: bool = False,
               suffix: str = "",
               lambda_max: float = 1.0):
    """lambda_max -- drop final-run lambdas above this from panel (c).

    Use this to exclude in-progress runs (e.g. lambda=10 still sweeping).
    Set to math.inf if everything is done.
    """
    """Phase diagram, redesigned per paper §3.2 / §4.4 alignment.

    Panel 1: accuracy-side flatness/sensitivity/rule-gap (no error bars).
             R_t, G_t(rule), and W_{t,eps} for eps in {0.1, 0.25, 0.5}.
             rule="fixed" uses alpha=0.8; rule="theory" uses each run's
             Fisher-derived theory_alpha.
    Panel 2: vector-level geometry. q_t and ||delta_t||_F on log axes.
    Panel 3: benchmark performance. AAA, Acc_T (left), forgetting (right).

    data_root / method_prefix / method_filter / variant_tag govern which
    Stage-1 method's runs are read:
      "fixed"  variant -> F2,  prm_fixed_lam,  prm_fixed
      "theory" variant -> F2p, prm_theory_lam, prm_theory
    """
    if data_root is None:
        data_root = F2_ROOT if rule == "fixed" else F2P_ROOT
    if method_prefix is None:
        method_prefix = "prm_fixed_lam" if rule == "fixed" else "prm_theory_lam"
    if method_filter is None:
        method_filter = "prm_fixed" if rule == "fixed" else "prm_theory"
    if variant_tag is None:
        variant_tag = rule

    out_dir = out_dir or os.path.join(OUT_ROOT, "fig1c")
    os.makedirs(out_dir, exist_ok=True)

    perf = _agg_perf([r for r in _load_final_runs(data_root, method_prefix)
                      if r["lambda_prox"] <= lambda_max])
    flat = _agg_flatness(_read_csv(FLAT_CSV),
                         method_filter=method_filter, rule=rule)
    rule_gap = _compute_rule_gap(data_root, rule=rule, alpha_fixed=0.8)
    # Drop in-progress lambdas from panel (a) too -- partial task coverage
    # there leaks into per-seed task-mean.
    flat = {l: v for l, v in flat.items() if l <= lambda_max}
    rule_gap = {l: v for l, v in rule_gap.items() if l <= lambda_max}
    if not perf:
        print(f"[fig1c:{variant_tag}] no final-run JSONs at {data_root}; skip")
        return

    perf_lams = sorted(perf.keys())
    flat_lams = sorted(flat.keys())

    def _xform(lams):
        # Render literal lam=0 at the lowest log decade in the available data.
        # If new very-small but nonzero lambdas are added later, this stays
        # sensible.
        nonzero = [l for l in lams if l > 0]
        anchor = (min(nonzero) / 10.0) if nonzero else 1e-7
        return [anchor if l == 0 else l for l in lams]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # ----- Panel 1: accuracy-side flatness / rule-gap (twin axes) -----
    # Left axis : R_t and G_t (small range, ~[0, 0.1])
    # Right axis: W_{t,eps} for eps in {0.1, 0.25, 0.5} (range [0, 1])
    ax = axes[0]
    xs_f = _xform(flat_lams)

    R_m = np.array([flat.get(l, {}).get("R_t", (np.nan, np.nan))[0]
                    for l in flat_lams])
    G_m = np.array([rule_gap.get(l, (np.nan, np.nan))[0]
                    for l in flat_lams])

    SENS_PURPLE = "#8e44ad"
    GAP_RED     = PRM_COLOR

    # Left axis: sensitivity / rule gap (decreasing). No error bars.
    ax.plot(xs_f, R_m, "-o", color=SENS_PURPLE, lw=1.8, ms=5,
            label=r"$R_t$  (sensitivity)")
    gap_label = (r"$G_t$  (fixed-$\alpha{=}0.8$ gap)" if rule == "fixed"
                 else r"$G_t$  (theory-$\alpha$ gap)")
    ax.plot(xs_f, G_m, "-D", color=GAP_RED, lw=1.8, ms=4.8, label=gap_label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{prox}}$")
    ax.set_ylabel(r"$R_t,\ G_t$  (sensitivity / rule gap)", color="#5a2c70")
    ax.tick_params(axis="y", colors="#5a2c70")
    finite_lr = np.concatenate([R_m[np.isfinite(R_m)], G_m[np.isfinite(G_m)]])
    sens_top = float(np.nanmax(finite_lr)) if finite_lr.size else 0.1
    ax.set_ylim(-0.005, max(0.10, sens_top * 1.15))

    # Right axis: plateau widths (increasing) on [0, 1]
    ax2 = ax.twinx()
    W_by_eps = _compute_W_eps_from_sweep(data_root, eps_list,
                                         method_filter=method_filter)
    W_by_eps = {e: {l: v for l, v in inner.items() if l <= lambda_max}
                for e, inner in W_by_eps.items()}
    # Color ramp from light to dark green for sorted eps (small eps = strict).
    eps_sorted = sorted(eps_list)
    green_ramp = ["#74c69d", "#1f9d6f", "#0b3d2e"]
    fmts = ["-s", "-^", "-v", "-D", "-x"]
    for k, e in enumerate(eps_sorted):
        color = green_ramp[k % len(green_ramp)]
        fmt = fmts[k % len(fmts)]
        agg = W_by_eps.get(float(e), {})
        m = np.array([agg.get(l, (np.nan, np.nan))[0] for l in flat_lams])
        ax2.plot(xs_f, m, fmt, color=color, lw=1.6, ms=4.6,
                 label=rf"$W_{{t,{e:.2f}}}$  (plateau width)")
    ax2.set_ylabel(r"$W_{t,\varepsilon}$  (plateau width)", color="#1f9d6f")
    ax2.tick_params(axis="y", colors="#1f9d6f")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#1f9d6f")
    ax2.set_ylim(0.0, 1.02)

    ax.set_title(r"(a) Path flatness & rule gap vs $\lambda_{\mathrm{prox}}$")
    ax.grid(True, which="both", alpha=0.30)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=True,
              framealpha=0.85, edgecolor="#cccccc",
              fontsize=7.8, ncol=1, handlelength=2.4)
    ax.axvline(1e-2, color="0.4", ls=":", lw=1.0, alpha=0.7, zorder=0)

    # ----- Panel 2: vector geometry -----
    ax = axes[1]
    q_m = np.array([flat.get(l, {}).get("q_t", (np.nan, np.nan))[0]
                    for l in flat_lams])
    q_s = np.array([flat.get(l, {}).get("q_t", (np.nan, np.nan))[1]
                    for l in flat_lams])
    d_m = np.array([flat.get(l, {}).get("delta_norm_F", (np.nan, np.nan))[0]
                    for l in flat_lams])
    d_s = np.array([flat.get(l, {}).get("delta_norm_F", (np.nan, np.nan))[1]
                    for l in flat_lams])

    ax.errorbar(xs_f, q_m, yerr=q_s, fmt="-o", color="#5b3aa6", lw=1.7,
                ms=5, capsize=2.5, label=r"$q_t$  (Fisher interference)")
    if add_q_dir:
        qdir = _agg_q_dir(_read_csv(FLAT_CSV),
                          method_filter=method_filter, rule=rule)
        qd_m = np.array([qdir.get(l, (np.nan, np.nan))[0]
                         for l in flat_lams])
        qd_s = np.array([qdir.get(l, (np.nan, np.nan))[1]
                         for l in flat_lams])
        ax.errorbar(xs_f, qd_m, yerr=qd_s, fmt="-D", color="#1faab8",
                    lw=1.7, ms=4.6, capsize=2.5,
                    label=r"$q_t^{\mathrm{dir}} = q_t / (\|\delta_t\|_F^2 + \varepsilon)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{prox}}$")
    ax.set_ylabel(r"$q_t,\ q_t^{\mathrm{dir}}$" if add_q_dir else r"$q_t$",
                  color="#5b3aa6")
    ax.tick_params(axis="y", colors="#5b3aa6")

    ax2 = ax.twinx()
    ax2.errorbar(xs_f, d_m, yerr=d_s, fmt="-s", color="#d96b96", lw=1.7,
                 ms=5, capsize=2.5, label=r"$\|\delta_t\|_F$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\|\delta_t\|_F$", color="#d96b96")
    ax2.tick_params(axis="y", colors="#d96b96")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d96b96")
    ax.set_title(r"(b) Vector geometry vs $\lambda_{\mathrm{prox}}$")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=8.5)
    ax.axvline(1e-2, color="0.4", ls=":", lw=1.0, alpha=0.7, zorder=0)

    # ----- Panel 3: benchmark performance -----
    ax = axes[2]
    aaa_m = np.array([perf[l]["AAA"][0] for l in perf_lams])
    aaa_s = np.array([perf[l]["AAA"][1] for l in perf_lams])
    last_m = np.array([perf[l]["last_acc"][0] for l in perf_lams])
    last_s = np.array([perf[l]["last_acc"][1] for l in perf_lams])
    forg_m = np.array([perf[l]["forgetting"][0] for l in perf_lams])
    forg_s = np.array([perf[l]["forgetting"][1] for l in perf_lams])

    xs_p = _xform(perf_lams)
    ax.errorbar(xs_p, aaa_m, yerr=aaa_s, fmt="-o", color="#2a6fbf", lw=1.7,
                ms=5, capsize=2.5, label=r"AAA")
    ax.errorbar(xs_p, last_m, yerr=last_s, fmt="-s", color="#f08a3f", lw=1.7,
                ms=4.5, capsize=2.5, label=r"$\mathrm{Acc}_T$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{prox}}$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 0.95)
    ax.set_title(r"(c) Benchmark vs $\lambda_{\mathrm{prox}}$")

    ax2 = ax.twinx()
    ax2.errorbar(xs_p, forg_m, yerr=forg_s, fmt="-^", color=PRM_COLOR, lw=1.7,
                 ms=5, capsize=2.5, label=r"forgetting")
    ax2.set_ylabel("forgetting", color=PRM_COLOR)
    ax2.tick_params(axis="y", colors=PRM_COLOR)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PRM_COLOR)
    forg_top = float(np.nanmax(forg_m + forg_s)) if np.isfinite(forg_m).any() else 0.25
    ax2.set_ylim(0.0, max(0.25, forg_top * 1.1))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False, fontsize=8.5)
    ax.axvline(1e-2, color="0.4", ls=":", lw=1.0, alpha=0.7, zorder=0)

    method_name = "PRM-fixed" if variant_tag == "fixed" else "PRM-Fisher"
    fig.suptitle(
        rf"PRM phase diagram on ImageNet-R 10T ({method_name},"
        r" $\lambda_{\mathrm{prox}}$ sweep)",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    out_pdf = os.path.join(out_dir,
                           f"fig1c_phase_diagram_{variant_tag}{suffix}.pdf")
    out_png = os.path.join(out_dir,
                           f"fig1c_phase_diagram_{variant_tag}{suffix}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig1c:{variant_tag}{suffix}] wrote {out_pdf} / .png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _setup_mpl()
    make_fig1a(task_str="05")
    make_fig1a(task_str="10")
    make_fig1a_compact(task_str="05")
    make_fig1b()
    make_fig1b(compact=True)
    # Originals: W at eps in {0.001, 0.0025, 0.005} (= 0.1pct / 0.25pct
    # / 0.5pct of accuracy units, the same scale CSV stored).
    make_fig1c(rule="fixed", eps_list=(0.001, 0.0025, 0.005))
    make_fig1c(rule="theory", eps_list=(0.001, 0.0025, 0.005))
    # v2: looser plateau eps {0.02, 0.05, 0.10} + q_t^dir on panel (b).
    # NOTE: at typical R_t (<<0.02 for most lambdas) these saturate at
    # W=1.  Kept as literal user request; tighter set in _v3 below.
    make_fig1c(rule="fixed",
               eps_list=(0.01, 0.02, 0.05),
               add_q_dir=True, suffix="_v2")
    make_fig1c(rule="theory",
               eps_list=(0.01, 0.02, 0.05),
               add_q_dir=True, suffix="_v2")


if __name__ == "__main__":
    main()
