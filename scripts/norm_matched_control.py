#!/usr/bin/env python3
"""Norm-Matched Control diagnostic (Causal Mechanism, Part B).

For each seed we:
  1. Load the common prefix theta_{t-1} (saved via train_baseline.py
     --save_prefix_at_task t).
  2. Load delta_pm and delta_prm — both Stage-1 outputs of task t starting
     from the SAME prefix, differing only in --prm_prox_weight (0.0 vs 0.01).
  3. Build four candidate task-vectors:
       PM_raw            = delta_pm
       PM_to_PRM_norm   = delta_pm * (||delta_prm|| / ||delta_pm||)
       PRM_raw          = delta_prm
       PRM_to_PM_norm   = delta_prm * (||delta_pm|| / ||delta_prm||)
  4. For each vector × alpha grid, install per-layer offsets via the
     landscape_probe._landscape_offset attribute (consumed by the
     install_offset_hook'd LoRALayer.get_weight) and evaluate
     Acc_seen / Acc_old / Acc_cur + per-task CE + (optional) KL/L2 drift.
  5. Aggregate per (seed, vector) into R_t, W_t, G_t, q_t-norm and dump CSVs.

Usage:
    python scripts/norm_matched_control.py \\
        --prefix_dir outputs/mechanism/prefix \\
        --delta_pm_dir outputs/mechanism/delta/pm \\
        --delta_prm_dir outputs/mechanism/delta/prm \\
        --task 5 --seeds 1 0 2 \\
        --dataset imagenet_r --data_root /path/to/imagenet-r \\
        --num_tasks 10 --model vit_base_patch16_clip_224.openai
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch import nn
import timm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from prm.lora import LoRALayer  # noqa: E402
from prm.prefix_io import (  # noqa: E402
    load_prefix,
    rebuild_model_from_prefix,
)
from prm.alpha_sweep import (  # noqa: E402
    evaluate_task_with_loss,
    _modify_head_for_eval,
    _build_function_probe_loader,
    _collect_function_state,
)
from prm.landscape_probe import (  # noqa: E402
    install_offset_hook,
    uninstall_offset_hook,
    clear_lora_delta_offset,
)
from prm.continual import ClassIncrementalManager  # noqa: E402
from prm.data import (  # noqa: E402
    get_path_dataset,
    make_label_maps,
    ClassIncrementalDataset,
    TargetTransform,
    build_eval_transform,
)
from torch.utils.data import DataLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix_dir", required=True,
                   help="Directory containing prefix_t{t:02d}_seed{s}.pt files.")
    p.add_argument("--delta_pm_dir", required=True,
                   help="Directory containing PM-branch delta_task{t}_seed{s}.pt.")
    p.add_argument("--delta_prm_dir", required=True,
                   help="Directory containing PRM-branch delta_task{t}_seed{s}.pt.")
    p.add_argument("--task", type=int, required=True,
                   help="1-indexed target task whose delta is being studied.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 0, 2])

    p.add_argument("--dataset", required=True,
                   choices=("cifar100", "imagenet_r", "sdomainet", "domainnet", "cub200"))
    p.add_argument("--data_root", required=True)
    p.add_argument("--num_tasks", type=int, required=True)
    p.add_argument("--shuffle_classes", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--model", type=str, default="vit_base_patch16_clip_224.openai")
    p.add_argument("--logit_type", type=str, default="head_out",
                   choices=("head_out", "sim_imgtext"))
    p.add_argument("--logit_scale", type=float, default=4.605170249938965)
    p.add_argument("--rank", type=int, default=10)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--module", type=str, default="kv")
    p.add_argument("--temperature", type=float, default=28.0)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--eval_workers", type=int, default=2)
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction,
                   default=True)

    p.add_argument("--alpha_grid", type=float, nargs="+",
                   default=[round(i * 0.05, 2) for i in range(21)])
    p.add_argument("--width_eps", type=float, default=0.02)
    p.add_argument("--gap_alpha", type=float, default=0.8)

    p.add_argument("--function_probe", action="store_true")
    p.add_argument("--function_probe_samples", type=int, default=500)

    p.add_argument("--out_root", type=str, default="outputs/mechanism/norm_matched")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Vector arithmetic on per-layer deltas
# ---------------------------------------------------------------------------

def _vec_norm(d: Dict[str, torch.Tensor]) -> float:
    """Global Frobenius norm of an effective LoRA delta (sum over layers).

    Each entry of `d` is a (out_dim, in_dim) matrix representing the per-layer
    `delta_l = B_l @ A_l` (the actual ΔW added to the qkv projection).
    Returns sqrt(sum_l ||delta_l||_F^2).
    """
    s = 0.0
    for _, t in d.items():
        s += float((t * t).sum().item())
    return float(s ** 0.5)


def _build_fisher_avg_per_layer(model, fisher: Dict, target_taskid: int,
                                 embed_dim: int = 768
                                 ) -> Dict[str, torch.Tensor]:
    """Build per-LoRALayer averaged Fisher F̄_<t (sliced to the layer's q/k/v
    position), keyed by the LoRALayer's `named_modules` name (matches the
    keys in delta_per_layer dicts).

    Mirrors the slicing in prm.alpha_sweep.compute_q_t.
    """
    out: Dict[str, torch.Tensor] = {}
    if target_taskid <= 0:
        return out
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALayer):
            continue
        block_key = f"{mod.layer}"
        pos = mod.pos
        # Get a reference shape from B@A to allocate F_acc.
        try:
            ref = (mod.lora_B[0].weight @ mod.lora_A[0].weight).detach()
        except Exception:
            continue
        F_acc = torch.zeros_like(ref)
        n_added = 0
        for i in range(target_taskid):
            f_i = fisher.get(block_key, {}).get(i)
            if f_i is None:
                # Try integer key
                f_i = fisher.get(block_key, {}).get(int(i))
            if f_i is None:
                continue
            f_i = f_i.to(ref.device).to(ref.dtype)
            if pos == "k":
                F_acc = F_acc + f_i[embed_dim: 2 * embed_dim, :].abs()
            elif pos == "v":
                F_acc = F_acc + f_i[2 * embed_dim: 3 * embed_dim, :].abs()
            elif pos == "q":
                F_acc = F_acc + f_i[:embed_dim, :].abs()
            n_added += 1
        if n_added > 0:
            F_acc = F_acc / float(n_added)
            out[name] = F_acc
    return out


def _compute_q(vector: Dict[str, torch.Tensor],
               F_avg_per_layer: Dict[str, torch.Tensor]) -> float:
    """q = sum_l <F̄_l, v_l ⊙ v_l>. Vector and Fisher must share layer keys."""
    q = 0.0
    for name, v in vector.items():
        F = F_avg_per_layer.get(name)
        if F is None:
            continue
        v_dev = v.to(F.device).to(F.dtype)
        q += float(torch.sum(F * v_dev * v_dev).item())
    return q


@torch.no_grad()
def _hook_sanity_check(model, vector: Dict[str, torch.Tensor],
                       loader, args) -> Dict:
    """Verify the offset hook implements `theta(alpha) = theta_common + alpha*v`.

    Test 1: alpha=0 must give identical logits to no-offset (max abs diff ≈ 0).
    Test 2: alpha=tiny vs alpha=2*tiny → logit deltas should scale ~2x
            (linearity of the LoRA forward in the offset).
    Returns metrics for logging.
    """
    # Grab a single small batch.
    images = None
    for x, _ in loader:
        images = x.cuda(non_blocking=True)
        break
    if images is None:
        return {"sanity": "no_data"}

    def _logits():
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=args.use_amp):
            return model(images).float()

    # Baseline: no offset.
    clear_lora_delta_offset(model)
    base = _logits()

    # alpha = 0: logits should match base exactly.
    _set_offsets_per_layer(model, vector, scale=0.0)
    a0 = _logits()
    err_zero = float((a0 - base).abs().max().item())

    # alpha = a_small.
    a_small = 1e-3
    _set_offsets_per_layer(model, vector, scale=a_small)
    aS = _logits()
    diff_S = (aS - base)

    # alpha = 2*a_small. Difference from baseline should be ~2x diff_S.
    _set_offsets_per_layer(model, vector, scale=2 * a_small)
    a2S = _logits()
    diff_2S = (a2S - base)

    # Linearity check: (diff_2S / 2) should ≈ diff_S.
    lin_err = float((diff_2S / 2 - diff_S).abs().max().item()) / max(
        float(diff_S.abs().max().item()), 1e-12
    )

    clear_lora_delta_offset(model)
    return {
        "sanity_alpha0_max_err":  err_zero,
        "sanity_linearity_relmax": lin_err,
        "sanity_diff_S_max":      float(diff_S.abs().max().item()),
    }


def _vec_scaled(d: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    return {k: t * factor for k, t in d.items()}


def _build_norm_matched_vectors(delta_pm: Dict, delta_prm: Dict
                                ) -> Dict[str, Dict[str, torch.Tensor]]:
    norm_pm = _vec_norm(delta_pm)
    norm_prm = _vec_norm(delta_prm)
    eps_div = 1e-12
    return {
        "PM_raw":           delta_pm,
        "PM_to_PRM_norm":  _vec_scaled(delta_pm,  norm_prm / max(norm_pm, eps_div)),
        "PRM_raw":         delta_prm,
        "PRM_to_PM_norm":  _vec_scaled(delta_prm, norm_pm  / max(norm_prm, eps_div)),
    }


def _set_offsets_per_layer(model, delta_per_layer: Dict[str, torch.Tensor],
                           scale: float):
    """Install per-LoRALayer offset via the _landscape_offset attribute.

    The offset is added to LoRALayer.get_weight() output by the installed
    monkey-patched `_patched_get_weight` (see install_offset_hook).
    """
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALayer):
            continue
        if name not in delta_per_layer:
            # No delta for this layer => zero offset.
            mod._landscape_offset = torch.zeros_like(
                mod.lora_B[0].weight @ mod.lora_A[0].weight
            )
            continue
        d = delta_per_layer[name]
        try:
            dev = next(mod.parameters()).device
            dtype = mod.lora_B[0].weight.dtype
        except StopIteration:
            dev, dtype = torch.device("cpu"), torch.float32
        mod._landscape_offset = (d.to(dev, dtype) * float(scale))


# ---------------------------------------------------------------------------
# Aggregate metrics from per-alpha rows
# ---------------------------------------------------------------------------

def _summarize_alpha_curve(rows: List[Dict], width_eps: float, gap_alpha: float
                           ) -> Dict:
    """Path-level metrics per Norm-Matched-Control conventions.

    Convention summary:
      R_seen   = max_alpha Acc_seen(alpha) - min_alpha Acc_seen(alpha)
      W_seen   = fraction of alpha where Acc_seen(alpha) >= max - width_eps
                 (Acc plateau width — what we report as "W_t" in the paper)
      W_old_loss = fraction of alpha where L_old(alpha) - L_old(0) <= width_eps
                   (kept as auxiliary metric; do NOT relabel as W_t)
      G_fixed  = max_alpha Acc_seen(alpha) - Acc_seen(alpha=gap_alpha)
                 ("rule gap" — what we report in the paper)
      write_in_gain_at_gap = Acc_seen(alpha=gap_alpha) - Acc_seen(alpha=0)
                             (kept separately; not the rule gap)
    """
    alphas = np.array([r["alpha"] for r in rows], dtype=np.float64)
    L_old = np.array([r["L_old"] for r in rows], dtype=np.float64)
    Acc_seen = np.array([r["Acc_seen"] for r in rows], dtype=np.float64)
    if len(alphas) == 0:
        nan = float("nan")
        return {k: nan for k in ("R_seen","W_seen","W_old_loss","G_fixed",
                                  "Acc_seen_at_gap","L_old_at_alpha0","max_Acc_seen",
                                  "argmax_alpha_seen","write_in_gain_at_gap")}

    max_acc = float(Acc_seen.max())
    R_seen = float(max_acc - Acc_seen.min())

    # Acc plateau width: fraction of grid where Acc_seen is within eps of max.
    W_seen = float(np.mean(Acc_seen >= max_acc - width_eps))

    # Old-loss plateau width (auxiliary — NOT the W_t we report).
    L_old_0 = float(L_old[np.argmin(np.abs(alphas - 0.0))])
    W_old_loss = float(np.mean(L_old - L_old_0 <= width_eps))

    # Rule gap at gap_alpha: max_alpha Acc_seen - Acc_seen(gap_alpha).
    idx_g = int(np.argmin(np.abs(alphas - gap_alpha)))
    Acc_at_gap = float(Acc_seen[idx_g])
    G_fixed = float(max_acc - Acc_at_gap)

    # Auxiliary: write-in gain from alpha=0 to alpha=gap_alpha.
    idx_0 = int(np.argmin(np.abs(alphas - 0.0)))
    write_in_gain = float(Acc_at_gap - Acc_seen[idx_0])

    return {
        "R_seen": R_seen,
        "W_seen": W_seen,
        "W_old_loss": W_old_loss,
        "G_fixed": G_fixed,
        "Acc_seen_at_gap": Acc_at_gap,
        "L_old_at_alpha0": L_old_0,
        "max_Acc_seen": max_acc,
        "argmax_alpha_seen": float(alphas[int(Acc_seen.argmax())]),
        "write_in_gain_at_gap": write_in_gain,
    }


# ---------------------------------------------------------------------------
# Per-(seed) execution
# ---------------------------------------------------------------------------

def _eval_one_seed(args, seed: int):
    prefix_path = os.path.join(
        args.prefix_dir, f"prefix_t{args.task:02d}_seed{seed}.pt"
    )
    delta_pm_path = os.path.join(
        args.delta_pm_dir, f"delta_task{args.task:02d}_seed{seed}.pt"
    )
    delta_prm_path = os.path.join(
        args.delta_prm_dir, f"delta_task{args.task:02d}_seed{seed}.pt"
    )

    print(f"\n=== seed={seed} task={args.task} ===")
    print(f"  prefix: {prefix_path}")
    print(f"  delta_pm: {delta_pm_path}")
    print(f"  delta_prm: {delta_prm_path}")

    for p in (prefix_path, delta_pm_path, delta_prm_path):
        if not os.path.exists(p):
            print(f"  [skip] missing: {p}")
            return [], []

    loaded = load_prefix(prefix_path)
    model, pretrained_cfg, completed = rebuild_model_from_prefix(
        loaded, args, timm.create_model,
    )
    target_taskid = args.task - 1
    assert target_taskid == completed + 1, (
        f"prefix completed_taskid={completed} but target task={args.task} "
        f"=> taskid={target_taskid}; expected {completed+1}"
    )
    delta_pm = torch.load(delta_pm_path, map_location="cpu")["delta_per_layer"]
    delta_prm = torch.load(delta_prm_path, map_location="cpu")["delta_per_layer"]

    # Build cl_manager + label_map for evaluation. label_map is in the prefix;
    # cl_manager is rebuilt deterministically from seed.
    train_dataset = get_path_dataset(args.dataset, args.data_root, train=False)
    cl_manager = ClassIncrementalManager(
        train_dataset.class_list, args.num_tasks, seed,
        shuffle=args.shuffle_classes,
    )
    # Advance _current_taskid to target_taskid - 1 (eval uses sofar through
    # task target_taskid which is the new task being probed).
    label_map = dict(loaded.get("label_map") or {})
    new_classes_seen = []
    for taskid, task_classes in cl_manager:
        if taskid > target_taskid:
            break
        # Augment label_map for the target task too.
        if taskid == target_taskid:
            g2l_map = make_label_maps(taskid, task_classes)
            label_map.update(g2l_map)
            new_classes_seen = task_classes
            break

    # Reconstruct param_cache (head per task) onto GPU.
    pc_cpu = loaded.get("param_cache") or {}
    param_cache = {}
    for k, v in pc_cpu.items():
        if isinstance(v, dict):
            param_cache[k] = {
                kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()
            }
        elif isinstance(v, torch.Tensor):
            param_cache[k] = v.cuda()
        else:
            param_cache[k] = v

    # We need a per-task head for the target task to evaluate its accuracy.
    # The delta payload includes head_weight + head_bias for the target task.
    pm_payload = torch.load(delta_pm_path, map_location="cpu")
    target_head_w = pm_payload["head_weight"].cuda()
    target_head_b = (
        pm_payload["head_bias"].cuda() if pm_payload.get("head_bias") is not None
        else None
    )
    param_cache[f"task_params_{target_taskid}"] = {
        "head.weight": target_head_w,
        "head.bias":   target_head_b if target_head_b is not None else torch.zeros(target_head_w.shape[0]).cuda(),
    }

    # Assemble eval head over tasks 0..target_taskid.
    model = _modify_head_for_eval(model, cl_manager, param_cache, target_taskid)
    model.cuda().eval()

    # Install the offset hook (idempotent).
    install_offset_hook(model)

    criterion = nn.CrossEntropyLoss().cuda()
    vectors = _build_norm_matched_vectors(delta_pm, delta_prm)
    norm_pm = _vec_norm(delta_pm)
    norm_prm = _vec_norm(delta_prm)
    print(f"  ||delta_pm||_F   = {norm_pm:.4f}")
    print(f"  ||delta_prm||_F = {norm_prm:.4f}")

    # Indexing report — explicit so reviewer-style audits are easy.
    print(f"  [indexing] prefix completed_taskid={completed} (0-indexed)")
    print(f"  [indexing] target_taskid={target_taskid} (0-indexed) "
          f"== task {target_taskid + 1} (1-indexed)")
    print(f"  [indexing] eval old tasks: 0..{target_taskid - 1}; "
          f"current task: {target_taskid}")
    print(f"  [indexing] Fisher F_<t aggregates fishers from tasks "
          f"0..{target_taskid - 1}")

    # ----- q_t for each scaled vector (uses common-prefix Fisher F̄_<t) -----
    fisher_loaded = loaded.get("fisher") or {}
    F_avg_per_layer = _build_fisher_avg_per_layer(
        model, fisher_loaded, target_taskid,
    )
    if not F_avg_per_layer:
        print(f"  [warn] no Fisher loaded from prefix; q_t will be NaN")
    else:
        print(f"  Fisher F̄_<t loaded: {len(F_avg_per_layer)} layers, "
              f"averaged over {target_taskid} prior tasks")
    vec_q = {name: _compute_q(vec, F_avg_per_layer) for name, vec in vectors.items()}
    vec_norm_actual = {name: _vec_norm(vec) for name, vec in vectors.items()}
    for name in vectors:
        n2 = vec_norm_actual[name] ** 2 + 1e-12
        print(f"  [{name}] ||v||={vec_norm_actual[name]:.4f}  q={vec_q[name]:.6g}  "
              f"q_dir=q/||v||²={vec_q[name]/n2:.6g}")

    # ----- Hook sanity check (alpha=0 identity, alpha-linearity) -----
    sanity_loader = DataLoader(
        ClassIncrementalDataset(
            get_path_dataset(args.dataset, args.data_root, train=False),
            cl_manager.get_classes(0),
            build_eval_transform(args.dataset, pretrained_cfg),
            TargetTransform(label_map, map_to_local=False),
        ),
        batch_size=min(args.eval_batch_size, 64), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    sanity = _hook_sanity_check(model, vectors["PRM_raw"], sanity_loader, args)
    print(f"  [sanity] alpha0_max_err={sanity.get('sanity_alpha0_max_err'):.2e}  "
          f"linearity_relmax={sanity.get('sanity_linearity_relmax'):.2e}  "
          f"diff_S_max={sanity.get('sanity_diff_S_max'):.4f}")
    if sanity.get('sanity_alpha0_max_err', 1.0) > 1e-3:
        print(f"  [WARN] hook offset at alpha=0 not near-identity; check semantics")
    if sanity.get('sanity_linearity_relmax', 1.0) > 5e-2:
        print(f"  [WARN] hook offset not linear in alpha (rel err > 5%); check")

    # Optional function-state probe.
    probe_loader = None
    ref_logits = None
    if args.function_probe:
        probe_loader = _build_function_probe_loader(
            args, cl_manager, pretrained_cfg, label_map, target_taskid,
            n_samples_total=int(args.function_probe_samples), rng_seed=0,
        )
        # ref = no-offset model.
        clear_lora_delta_offset(model)
        _, ref_logits, _ = _collect_function_state(
            model, probe_loader, use_amp=args.use_amp,
        )

    per_alpha_rows: List[Dict] = []
    per_vec_rows: List[Dict] = []

    for vec_name, vec in vectors.items():
        rows = []
        for alpha in args.alpha_grid:
            _set_offsets_per_layer(model, vec, scale=float(alpha))
            per_task_loss, per_task_acc = [], []
            for eval_tid in range(target_taskid + 1):
                eval_classes = cl_manager.get_classes(eval_tid)
                loss_i, acc_i = evaluate_task_with_loss(
                    model, args, eval_classes, pretrained_cfg, label_map,
                    criterion=criterion,
                )
                per_task_loss.append(float(loss_i))
                per_task_acc.append(float(acc_i))

            L_old = float(np.mean(per_task_loss[:-1])) if target_taskid >= 1 else float("nan")
            Acc_old = float(np.mean(per_task_acc[:-1])) if target_taskid >= 1 else float("nan")
            L_cur = float(per_task_loss[-1])
            Acc_cur = float(per_task_acc[-1])
            L_seen = float(np.mean(per_task_loss))
            Acc_seen = float(np.mean(per_task_acc))

            row = {
                "seed":     int(seed),
                "task":     int(args.task),
                "vector":   vec_name,
                "alpha":    float(alpha),
                "L_old":    L_old,
                "Acc_old":  Acc_old,
                "L_cur":    L_cur,
                "Acc_cur":  Acc_cur,
                "L_seen":   L_seen,
                "Acc_seen": Acc_seen,
                "vec_norm_F": float(_vec_norm(vec)),
            }
            if probe_loader is not None and ref_logits is not None:
                _, cur_logits, _ = _collect_function_state(
                    model, probe_loader, use_amp=args.use_amp,
                )
                p_ref = torch.softmax(ref_logits / args.temperature, dim=-1)
                p_cur = torch.softmax(cur_logits / args.temperature, dim=-1)
                kl = (p_ref * (torch.log(p_ref + 1e-12)
                               - torch.log(p_cur + 1e-12))).sum(dim=-1).mean().item()
                logit_l2 = ((cur_logits - ref_logits) ** 2).sum(dim=-1).mean().sqrt().item()
                row.update({
                    "KL_drift":       float(kl),
                    "logit_drift_L2": float(logit_l2),
                })
            rows.append(row)
            per_alpha_rows.append(row)

        s = _summarize_alpha_curve(rows, args.width_eps, args.gap_alpha)
        v_norm = vec_norm_actual[vec_name]
        v_q = vec_q[vec_name]
        v_qdir = v_q / (v_norm ** 2 + 1e-12)
        per_vec_rows.append({
            "seed":           int(seed),
            "task":           int(args.task),
            "vector":         vec_name,
            "vec_norm_F":     float(v_norm),
            "norm_pm":        float(norm_pm),
            "norm_prm":      float(norm_prm),
            "q_t":            float(v_q),
            "q_t_dir":        float(v_qdir),
            "R_seen":         s["R_seen"],
            "W_seen":         s["W_seen"],
            "W_old_loss":     s["W_old_loss"],
            "G_fixed":        s["G_fixed"],
            "Acc_seen_at_gap":  s["Acc_seen_at_gap"],
            "write_in_gain_at_gap": s["write_in_gain_at_gap"],
            "L_old_at_alpha0":  s["L_old_at_alpha0"],
            "max_Acc_seen":   s["max_Acc_seen"],
            "argmax_alpha_seen": s["argmax_alpha_seen"],
            "sanity_alpha0_max_err":  sanity.get("sanity_alpha0_max_err"),
            "sanity_linearity_relmax": sanity.get("sanity_linearity_relmax"),
        })
        print(f"  {vec_name}: ||v||={v_norm:6.3f}  q_t={v_q:.4g}  "
              f"q_dir={v_qdir:.4g}  R_seen={s['R_seen']:.4f}  "
              f"W_seen={s['W_seen']:.4f}  G_fix={s['G_fixed']:.4f}  "
              f"Acc_seen@α={args.gap_alpha}={s['Acc_seen_at_gap']:.4f}")

    # Cleanup hooks before returning.
    clear_lora_delta_offset(model)
    uninstall_offset_hook(model)
    return per_alpha_rows, per_vec_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.out_root, exist_ok=True)
    all_per_alpha: List[Dict] = []
    all_per_vec: List[Dict] = []
    for seed in args.seeds:
        per_alpha, per_vec = _eval_one_seed(args, seed)
        all_per_alpha.extend(per_alpha)
        all_per_vec.extend(per_vec)

    if all_per_alpha:
        path = os.path.join(args.out_root, "per_alpha.csv")
        keys = list(all_per_alpha[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_per_alpha:
                w.writerow(r)
        print(f"\n[wrote] {path} ({len(all_per_alpha)} rows)")
    if all_per_vec:
        path = os.path.join(args.out_root, "per_vector.csv")
        keys = list(all_per_vec[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_per_vec:
                w.writerow(r)
        print(f"[wrote] {path} ({len(all_per_vec)} rows)")


if __name__ == "__main__":
    main()
