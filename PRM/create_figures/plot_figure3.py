#!/usr/bin/env python3
"""Paper Figure 3: lambda_prox sensitivity.

The script reads the small, included data bundle under create_figures/data and
writes fig3_lambda_metrics.pdf to create_figures/outputs.

Font sizes are all scaled by the single ``FONT_SCALE`` knob below; edit it
to grow / shrink every text element in the figure proportionally.

  Panel (a): λ → mergeability axes (W_t,ε on left axis,
             q_t and q_t^dir on right log axis).
             Drops R_t and G_t per the new method-section narrative.
  Panel (b): λ → benchmark performance (AAA, Acc_T on left;
             Forg on right). Identifies the interior optimum.

Re-uses local helpers to avoid duplicating sweep-aggregation logic.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _figure_helpers import (  # type: ignore  # noqa: E402
    F2_ROOT, FLAT_CSV,
    _agg_perf, _load_final_runs, _read_csv,
    _agg_q_dir, _compute_W_eps_from_sweep, _agg_flatness,
)

OUT_DIR = os.path.join(REPO, "create_figures", "outputs")

# One knob to scale every font in this figure. Bump up to enlarge, down to
# shrink. Applies to rcParams and every explicit ``fontsize=`` below.
FONT_SCALE = 1.40

# Curve colors (panel a)
W_COLOR_25 = "#1a8a4d"   # darker green
W_COLOR_50 = "#3aa56b"   # lighter green
Q_COLOR    = "#5e2a86"   # deep purple
QDIR_COLOR = "#b47ec7"   # light purple
W_AXIS_COLOR = "#1a6e3f"
Q_AXIS_COLOR = "#5e2a86"

# Panel b colors
ACC_COLOR  = "#1f4f8b"
ACCT_COLOR = "#5290d6"
FORG_COLOR = "#666666"

CHOSEN_LAM = 1e-2
CHOSEN_COLOR = "#c8243d"

LAM_MAX = 1.0  # drop λ=10


def _drop_high_lam(xs: np.ndarray, *arrs: np.ndarray):
    mask = xs <= LAM_MAX
    return (xs[mask],) + tuple(a[mask] for a in arrs)


def _xys(lam_dict, key):
    """{lam: {key: (mean,std)}} -> sorted (xs, ms, ss); drops λ>LAM_MAX."""
    items = sorted([(lam, v[key]) for lam, v in lam_dict.items()
                    if key in v and np.isfinite(v[key][0])])
    if not items:
        return np.array([]), np.array([]), np.array([])
    xs = np.array([k for k, _ in items], dtype=float)
    ms = np.array([v[0] for _, v in items], dtype=float)
    ss = np.array([v[1] for _, v in items], dtype=float)
    return _drop_high_lam(xs, ms, ss)


def _xys_perf(perf_dict, key):
    """{lam: {key: (mean, std)}} -> sorted (xs, ms, ss); drops λ>LAM_MAX."""
    items = sorted([(lam, v[key]) for lam, v in perf_dict.items()
                    if key in v and np.isfinite(v[key][0])])
    xs = np.array([k for k, _ in items], dtype=float)
    ms = np.array([v[0] for _, v in items], dtype=float)
    ss = np.array([v[1] for _, v in items], dtype=float)
    return _drop_high_lam(xs, ms, ss)


def _xys_qdir(qdir_dict):
    items = sorted([(lam, v) for lam, v in qdir_dict.items()
                    if np.isfinite(v[0])])
    xs = np.array([k for k, _ in items], dtype=float)
    ms = np.array([v[0] for _, v in items], dtype=float)
    ss = np.array([v[1] for _, v in items], dtype=float)
    return _drop_high_lam(xs, ms, ss)


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

    # ---- Aggregate ---------------------------------------------------------
    perf = _agg_perf(_load_final_runs(F2_ROOT, method_prefix="prm_fixed_lam"))

    flat_rows = _read_csv(FLAT_CSV)
    flat = _agg_flatness(flat_rows, method_filter="prm_fixed", rule="fixed")
    qdir = _agg_q_dir(flat_rows, method_filter="prm_fixed", rule="fixed")

    # plateau widths at ε ∈ {0.0025, 0.005} (= 0.25 / 0.5 pp)
    W_by = _compute_W_eps_from_sweep(F2_ROOT, [0.0025, 0.005],
                                     method_filter="prm_fixed")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ---- Panel (a): mergeability metrics ----------------------------------
    # left axis: W_{t,0.0025} and W_{t,0.005}  (range 0..1)
    xs_w25, m_w25, s_w25 = _xys({lam: {"W": v} for lam, v in W_by[0.0025].items()}, "W")
    xs_w50, m_w50, s_w50 = _xys({lam: {"W": v} for lam, v in W_by[0.005].items()}, "W")
    # mask λ=0 -> tiny positive (log axis)
    eps = 1e-5
    def _logsafe(x):
        return np.where(x <= 0, eps, x)
    ax_a.plot(_logsafe(xs_w25), m_w25, marker="o", color=W_COLOR_25,
              lw=1.8, label=r"$W_{t,0.25\mathrm{pp}}$")
    ax_a.fill_between(_logsafe(xs_w25), m_w25 - s_w25, m_w25 + s_w25,
                      color=W_COLOR_25, alpha=0.18, lw=0)
    ax_a.plot(_logsafe(xs_w50), m_w50, marker="s", color=W_COLOR_50,
              lw=1.8, linestyle="--", label=r"$W_{t,0.5\mathrm{pp}}$")
    ax_a.fill_between(_logsafe(xs_w50), m_w50 - s_w50, m_w50 + s_w50,
                      color=W_COLOR_50, alpha=0.18, lw=0)
    ax_a.set_xscale("log")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlabel(r"$\lambda_{\mathrm{prox}}$")
    ax_a.set_ylabel(r"plateau width  $W_{t,\varepsilon}$",
                    color=W_AXIS_COLOR)
    ax_a.tick_params(axis="y", labelcolor=W_AXIS_COLOR)

    # right axis: q_t and q_t^dir on log scale, in two purples
    ax_a2 = ax_a.twinx()
    xs_q, m_q, s_q = _xys(flat, "q_t")
    ax_a2.errorbar(_logsafe(xs_q), m_q, yerr=s_q,
                   marker="o", color=Q_COLOR, lw=1.8,
                   label=r"$q_t$")
    xs_qd, m_qd, s_qd = _xys_qdir(qdir)
    ax_a2.errorbar(_logsafe(xs_qd), m_qd, yerr=s_qd,
                   marker="^", color=QDIR_COLOR, lw=1.8, linestyle="--",
                   label=r"$q_t^{\mathrm{dir}}$")
    ax_a2.set_yscale("log")
    ax_a2.set_ylabel(r"Fisher interference $q_t,\,q_t^{\mathrm{dir}}$ (log)",
                     color=Q_AXIS_COLOR)
    ax_a2.tick_params(axis="y", labelcolor=Q_AXIS_COLOR)
    ax_a2.spines["right"].set_visible(True)
    ax_a2.spines["right"].set_color(Q_AXIS_COLOR)
    ax_a.spines["left"].set_color(W_AXIS_COLOR)
    ax_a2.grid(False)

    # chosen-λ vline + label (panel a)
    ax_a.axvline(CHOSEN_LAM, color=CHOSEN_COLOR, linestyle=":",
                 lw=1.4, alpha=0.8, zorder=0)
    ymin_a, ymax_a = ax_a.get_ylim()
    ax_a.text(1.15e-2, ymin_a + 0.05 * (ymax_a - ymin_a),
              r"chosen $\lambda{=}10^{-2}$",
              color=CHOSEN_COLOR, fontsize=8.5 * FONT_SCALE, rotation=90, va="bottom")

    # combined legend at right-center
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="center right",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False, fontsize=8.5 * FONT_SCALE)
    ax_a.set_title(r"(a) $\lambda_{\mathrm{prox}}$ controls mergeability")

    # ---- Panel (b): benchmark performance ----------------------------------
    xs_aaa, m_aaa, s_aaa = _xys_perf(perf, "AAA")
    xs_acc, m_acc, s_acc = _xys_perf(perf, "last_acc")
    xs_f, m_f, s_f = _xys_perf(perf, "forgetting")
    # Scale to percentages (raw JSON values are fractions in [0,1]).
    m_aaa, s_aaa = m_aaa * 100, s_aaa * 100
    m_acc, s_acc = m_acc * 100, s_acc * 100
    m_f, s_f = m_f * 100, s_f * 100
    ax_b.errorbar(_logsafe(xs_aaa), m_aaa, yerr=s_aaa,
                  marker="o", color=ACC_COLOR, lw=1.8, label="AAA")
    ax_b.errorbar(_logsafe(xs_acc), m_acc, yerr=s_acc,
                  marker="s", color=ACCT_COLOR, lw=1.8, linestyle="--",
                  label=r"Acc$_T$")
    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"$\lambda_{\mathrm{prox}}$")
    ax_b.set_ylabel("AAA / Acc$_T$  (%)", color=ACC_COLOR)
    ax_b.tick_params(axis="y", labelcolor=ACC_COLOR)
    ax_b.spines["left"].set_color(ACC_COLOR)

    ax_b2 = ax_b.twinx()
    ax_b2.errorbar(_logsafe(xs_f), m_f, yerr=s_f,
                   marker="^", color=FORG_COLOR, lw=1.8,
                   label="Forgetting")
    ax_b2.set_ylabel("forgetting (%)", color=FORG_COLOR)
    ax_b2.tick_params(axis="y", labelcolor=FORG_COLOR)
    ax_b2.spines["right"].set_visible(True)
    ax_b2.spines["right"].set_color(FORG_COLOR)
    ax_b2.grid(False)

    # mark the chosen lambda
    ax_b.axvline(CHOSEN_LAM, color=CHOSEN_COLOR, linestyle=":",
                 lw=1.4, alpha=0.8, zorder=0)
    ymin_b, ymax_b = ax_b.get_ylim()
    ax_b.text(1.15e-2, ymin_b + 0.05 * (ymax_b - ymin_b),
              r"chosen $\lambda{=}10^{-2}$",
              color=CHOSEN_COLOR, fontsize=8.5 * FONT_SCALE, rotation=90, va="bottom")

    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="lower left",
                frameon=False, fontsize=8.5 * FONT_SCALE)
    ax_b.set_title(r"(b) $\lambda_{\mathrm{prox}}$ controls performance")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_lambda_metrics.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  fig3 -> {out}")


if __name__ == "__main__":
    main()
