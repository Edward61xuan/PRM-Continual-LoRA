"""Phase 4 — 2D shared/residual merge.

Decomposes the per-task task vector delta_t = B_t @ A_t into two components
based on the last-K training-epoch snapshots:

    delta_shared = mean_l  W_l            (stable, temporally averaged)
    delta_res    = delta_final - delta_shared   (volatile residual)

The Stage-2 merge then uses TWO write strengths:

    theta_t = theta_{t-1} + alpha_s * delta_shared + alpha_r * delta_res

with a proxy objective:

    J(alpha_s, alpha_r) = L_t^val(theta_t) + lambda * Q(alpha_s, alpha_r)
    Q(alpha_s, alpha_r) = alpha_s^2 * q_ss + 2 * alpha_s * alpha_r * q_sr + alpha_r^2 * q_rr

where q_** are Fisher-weighted inner products of (delta_shared, delta_res).
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import torch
from torch import nn

from .lora import LoRALayer
from .alpha_sweep import (
    evaluate_task_with_loss,
    _modify_head_for_eval,
    merge_with_alpha,
)


# ---------------------------------------------------------------------------
# Decomposition + application
# ---------------------------------------------------------------------------

def _decompose_one(module, late_snapshots, task_id):
    """Return (delta_shared, delta_res) for one LoRA module."""
    key = f"{module.layer}_{module.pos}"
    delta_final = (
        module.lora_B[task_id].weight @ module.lora_A[task_id].weight
    ).detach().clone()
    snaps = late_snapshots.get(key, []) if late_snapshots else []
    if len(snaps) == 0:
        # No snapshots: treat entire delta as "shared" and zero residual.
        return delta_final, torch.zeros_like(delta_final)
    stacked = torch.stack(snaps)  # (K, out, in)
    delta_shared = stacked.mean(dim=0).clone()
    delta_res = delta_final - delta_shared
    return delta_shared, delta_res


def apply_2d_decomposition(model, task_id, late_snapshots, alpha_s, alpha_r):
    """Decompose + store 2D components with chosen (alpha_s, alpha_r).

    Must be called *after* Stage 1 training (so lora_B[t]@lora_A[t] is final).
    Leaves alphas[task_id] slot filled (dummy value) for bookkeeping invariants
    in LoRALayer.update().
    """
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALayer):
                continue
            shared, res = _decompose_one(module, late_snapshots, task_id)
            module._d2_shared[task_id] = shared.detach()
            module._d2_res[task_id] = res.detach()
            module._d2_alpha_s[task_id] = float(alpha_s)
            module._d2_alpha_r[task_id] = float(alpha_r)
            # Ensure alphas slot exists for len(alphas) == task_id + 1 invariant.
            if (task_id + 1) != len(module.alphas):
                module.cal_alpha(1.0)


def set_2d_alphas(model, task_id, alpha_s, alpha_r):
    """In-place update of (alpha_s, alpha_r) without re-decomposing."""
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALayer):
                continue
            if task_id in module._d2_shared:
                module._d2_alpha_s[task_id] = float(alpha_s)
                module._d2_alpha_r[task_id] = float(alpha_r)


# ---------------------------------------------------------------------------
# Fisher quadratic forms q_ss, q_sr, q_rr
# ---------------------------------------------------------------------------

def compute_2d_q(model, fisher, task_id, embed_dim=768):
    """Return (q_ss, q_sr, q_rr) summed over LoRA modules.

    q_ss = <delta_s, Fbar * delta_s>
    q_sr = <delta_s, Fbar * delta_r>
    q_rr = <delta_r, Fbar * delta_r>

    Fbar = mean of prior-task Fisher diagonals, sliced by K/V/Q position.
    Only valid for task_id >= 1.
    """
    if task_id == 0:
        return 0.0, 0.0, 0.0
    q_ss = 0.0
    q_sr = 0.0
    q_rr = 0.0
    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue
        if task_id not in module._d2_shared:
            continue
        delta_s = module._d2_shared[task_id]
        delta_r = module._d2_res[task_id]
        F_accum = torch.zeros_like(delta_s)
        for i in range(task_id):
            f_i = fisher[f"{module.layer}"][i]
            if module.pos == "k":
                F_accum += f_i[embed_dim: 2 * embed_dim, :].abs()
            elif module.pos == "v":
                F_accum += f_i[2 * embed_dim: 3 * embed_dim, :].abs()
            elif module.pos == "q":
                F_accum += f_i[:embed_dim, :].abs()
        F_accum = F_accum / float(max(1, task_id))
        q_ss += float(torch.sum(F_accum * delta_s * delta_s).item())
        q_sr += float(torch.sum(F_accum * delta_s * delta_r).item())
        q_rr += float(torch.sum(F_accum * delta_r * delta_r).item())
    return q_ss, q_sr, q_rr


# ---------------------------------------------------------------------------
# 2D empirical alpha selection
# ---------------------------------------------------------------------------

def choose_empirical_alpha_2d(
    model, args, cl_manager, pretrained_cfg, label_map, param_cache, fisher,
    task_id, late_snapshots,
    lambda_proxy: float = 1.0,
    alpha_grid=None,
    save_path=None,
    tag="",
):
    """Search (alpha_s, alpha_r) on a 2D grid; minimize J.

    Uses L_t^val (current-task eval only, no full multi-task sweep) for speed.
    Fisher penalty via precomputed (q_ss, q_sr, q_rr). Grid defaults to
    {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}^2 = 36 points.
    """
    if task_id == 0:
        merge_with_alpha(model, task_id, 1.0)
        return 1.0, 1.0

    if alpha_grid is None:
        alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # First, decompose once with dummy alphas (values will be overwritten).
    apply_2d_decomposition(model, task_id, late_snapshots, 0.0, 0.0)
    q_ss, q_sr, q_rr = compute_2d_q(model, fisher, task_id)

    # Freeze grad, rebuild eval head.
    saved_rg = {}
    for n, p in model.named_parameters():
        saved_rg[n] = p.requires_grad
        p.requires_grad_(False)
    _modify_head_for_eval(model, cl_manager, param_cache, task_id)
    model.cuda(); model.eval()

    current_classes = cl_manager.get_classes(task_id)
    criterion = nn.CrossEntropyLoss().cuda()

    records = []
    best = {"J": float("inf"), "alpha_s": 1.0, "alpha_r": 1.0, "L_cur": 0.0, "Acc_cur": 0.0}
    for a_s in alpha_grid:
        for a_r in alpha_grid:
            set_2d_alphas(model, task_id, a_s, a_r)
            L_cur, Acc_cur = evaluate_task_with_loss(
                model, args, current_classes, pretrained_cfg, label_map,
                criterion=criterion,
            )
            Q = a_s * a_s * q_ss + 2 * a_s * a_r * q_sr + a_r * a_r * q_rr
            J = float(L_cur + lambda_proxy * Q)
            records.append({
                "alpha_s": float(a_s), "alpha_r": float(a_r),
                "L_cur": float(L_cur), "Acc_cur": float(Acc_cur),
                "Q": float(Q), "J": J,
            })
            if J < best["J"]:
                best = {"J": J, "alpha_s": float(a_s), "alpha_r": float(a_r),
                        "L_cur": float(L_cur), "Acc_cur": float(Acc_cur)}

    # Finalize
    set_2d_alphas(model, task_id, best["alpha_s"], best["alpha_r"])

    # Restore grad state
    for n, p in model.named_parameters():
        p.requires_grad_(saved_rg.get(n, False))

    print(
        f"  [2d-empirical task {task_id+1}] lambda={lambda_proxy:g} "
        f"q_ss={q_ss:.4g} q_sr={q_sr:.4g} q_rr={q_rr:.4g} "
        f"-> (alpha_s*={best['alpha_s']:.2f}, alpha_r*={best['alpha_r']:.2f}) "
        f"(L_cur={best['L_cur']:.3f}, J={best['J']:.3f})"
    )

    if save_path is not None:
        record = {
            "tag": tag,
            "task_id_0indexed": int(task_id),
            "task_id_1indexed": int(task_id) + 1,
            "lambda_proxy": float(lambda_proxy),
            "alpha_grid": [float(a) for a in alpha_grid],
            "q_ss": float(q_ss), "q_sr": float(q_sr), "q_rr": float(q_rr),
            "records": records,
            "alpha_s_star": best["alpha_s"],
            "alpha_r_star": best["alpha_r"],
            "J_star": best["J"],
            "seed": int(getattr(args, "seed", 2024)),
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(record, f, indent=2)

    return best["alpha_s"], best["alpha_r"]


def apply_2d_fixed(model, task_id, late_snapshots, alpha_s, alpha_r):
    """Non-empirical 2D write-back with externally supplied (alpha_s, alpha_r)."""
    if task_id == 0:
        merge_with_alpha(model, task_id, 1.0)
        return 1.0, 1.0
    apply_2d_decomposition(model, task_id, late_snapshots, alpha_s, alpha_r)
    print(f"  [2d-fixed task {task_id+1}] (alpha_s={alpha_s}, alpha_r={alpha_r})")
    return float(alpha_s), float(alpha_r)
