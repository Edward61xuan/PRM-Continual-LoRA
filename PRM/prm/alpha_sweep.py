"""
Alpha sweep and empirical alpha selection for PM/PRM continual merging.

Phase 1 (diagnostic):
    run_alpha_sweep()
        At a selected task t, sweep alpha over a dense grid (e.g. 20 points),
        evaluate per-task loss/acc for all tasks <= t, and save the raw data.
        Interpolates on the line theta(alpha) = theta_{t-1} + alpha * delta_t.

Phase 2 (deployable):
    choose_empirical_alpha()
        At each task t, sweep alpha over a coarser grid (e.g. 11 points),
        minimize the rehearsal-free proxy
            J(alpha) = L_t^val(alpha) + lambda * alpha^2 * q_t,
        and use the argmin as the merge coefficient.

All machinery in this module is a strict add-on to the existing PRM/PM pipeline:
  - does not modify LoRALayer behavior; only mutates alphas[t].data in place
  - does not touch training, Fisher accumulation, or head caching
  - Fisher slicing (K: [D:2D], V: [2D:3D]) matches compute_fisher_alpha()
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .lora import LoRALayer
from .data import (
    get_path_dataset,
    ClassIncrementalDataset,
    TargetTransform,
    build_eval_transform,
)


# ---------------------------------------------------------------------------
# Alpha slot management (in-place mutation of LoRALayer.alphas[t])
# ---------------------------------------------------------------------------

def ensure_alpha_slot(model, task_id, init_alpha: float = 1.0):
    """Make sure each LoRALayer has an alphas[task_id] Parameter.

    Enters with len(alphas) == task_id (no slot for task_id yet).
    Appends one Parameter with value `init_alpha` so that the
    len(alphas) == task_id + 1 invariant is satisfied and subsequent
    in-place mutation works. For task 0, also records old_weight
    (mirroring prm_write_back/merge_pm for first-task bookkeeping).
    """
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALayer):
                continue
            if task_id == 0 and not hasattr(module, "old_weight"):
                module.old_weight = (
                    module.lora_B[0].weight @ module.lora_A[0].weight
                ).detach()
            if (task_id + 1) != len(module.alphas):
                module.cal_alpha(float(init_alpha))


def set_task_alpha(model, task_id, alpha: float):
    """In-place set alphas[task_id].data across all LoRALayers.

    Assumes ensure_alpha_slot() or cal_alpha() has already been called for
    task_id. Does NOT append; safe to call repeatedly inside a sweep loop.
    """
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALayer):
                continue
            module.alphas[task_id].data.fill_(float(alpha))


def merge_with_alpha(model, task_id, alpha: float):
    """Fresh write-back using an externally-provided alpha.

    Mirrors the tail of merge_pm/prm_write_back without computing alpha
    internally. Only call this on the non-sweep path (sweep path uses
    ensure_alpha_slot + set_task_alpha instead).
    """
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALayer):
                continue
            if task_id == 0:
                module.old_weight = (
                    module.lora_B[0].weight @ module.lora_A[0].weight
                ).detach()
            module.cal_alpha(float(alpha))


# ---------------------------------------------------------------------------
# Fisher-weighted interference q_t = <delta_t, Fbar_{<t} o delta_t>
# ---------------------------------------------------------------------------

def compute_q_t(model, fisher, task_id, embed_dim: int = 768,
                return_delta_norms: bool = False):
    """Quadratic interference strength of the current task vector.

        q_t = sum_layers sum_j  Fbar_{<t}_j * delta_t_j^2

    Fbar_{<t} is the per-element mean of Fisher diagonals from tasks 0..t-1,
    sliced to match each LoRA module's K/V/Q position (same slicing as
    compute_fisher_alpha). For task_id == 0 returns 0.0.

    Args:
        return_delta_norms: if True, also compute and return the Frobenius
            norm of delta_t = B_t @ A_t (global and per-layer). Captured
            before any write-back or direction-gate overwrite, so the
            returned norm is the raw Stage-1 output.

    Returns:
        If return_delta_norms is False (default, backward compatible):
            (q_total: float, q_per_layer: dict[str, float])
        If return_delta_norms is True:
            (q_total, q_per_layer, delta_norm_F, delta_norm_per_layer)
    """
    if task_id == 0:
        # delta_0 is still meaningful (it's the first LoRA adapter).
        if return_delta_norms:
            dn, dnp = compute_delta_norm(model, task_id)
            return 0.0, {}, dn, dnp
        return 0.0, {}

    q_total = 0.0
    q_per_layer = {}
    delta_sq_total = 0.0
    delta_norm_per_layer = {}

    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue

        layer_name = f"{module.layer}"
        pos = module.pos

        delta = (
            module.lora_B[task_id].weight @ module.lora_A[task_id].weight
        ).detach()

        if return_delta_norms:
            d_sq = float(torch.sum(delta * delta).detach().cpu().item())
            delta_sq_total += d_sq
            delta_norm_per_layer[f"L{layer_name}/{pos}"] = d_sq ** 0.5

        F_accum = torch.zeros_like(delta)
        for i in range(task_id):
            f_i = fisher[layer_name][i]
            if pos == "k":
                F_accum += f_i[embed_dim: 2 * embed_dim, :].abs()
            elif pos == "v":
                F_accum += f_i[2 * embed_dim: 3 * embed_dim, :].abs()
            elif pos == "q":
                F_accum += f_i[:embed_dim, :].abs()
        F_accum = F_accum / float(max(1, task_id))

        q_layer = float(torch.sum(F_accum * delta * delta).detach().cpu().item())
        key = f"L{layer_name}/{pos}"
        q_per_layer[key] = q_layer
        q_total += q_layer

    if return_delta_norms:
        return q_total, q_per_layer, delta_sq_total ** 0.5, delta_norm_per_layer
    return q_total, q_per_layer


def append_per_task_summary(results_file: str, header: dict, task_entry: dict):
    """Append a per-task summary entry to {results_file}.per_task.json.

    The sidecar file is separate from the main results JSON so aggregation
    scripts can read per-task progressions without parsing stdout logs.

    Schema:
        {header fields...,
         "per_task": [{task_id, q_t, delta_norm_F, alpha_theory,
                       alpha_chosen, alpha_method, train_loss_final,
                       val_acc, ...}, ...]}

    On re-entry for the same task_id, the old entry is replaced (keeps the
    sidecar idempotent if a run is resumed).
    """
    if not results_file:
        return
    sidecar_path = results_file + ".per_task.json"
    try:
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                data = json.load(f)
        else:
            data = dict(header)
            data["per_task"] = []
    except Exception:
        data = dict(header)
        data["per_task"] = []
    if "per_task" not in data or not isinstance(data["per_task"], list):
        data["per_task"] = []
    tid = task_entry.get("task_id")
    data["per_task"] = [e for e in data["per_task"] if e.get("task_id") != tid]
    data["per_task"].append(task_entry)
    data["per_task"].sort(key=lambda e: e.get("task_id", 0))
    os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_delta_norm(model, task_id):
    """Frobenius norm of delta_t = B_t @ A_t (global and per-layer).

    Independent of Fisher; usable at task 0 where q_t is trivially zero.

    Returns:
        (delta_norm_F: float, delta_norm_per_layer: dict[str, float])
    """
    total_sq = 0.0
    per_layer = {}
    for module in model.modules():
        if not isinstance(module, LoRALayer):
            continue
        delta = (
            module.lora_B[task_id].weight @ module.lora_A[task_id].weight
        ).detach()
        d_sq = float(torch.sum(delta * delta).detach().cpu().item())
        per_layer[f"L{module.layer}/{module.pos}"] = d_sq ** 0.5
        total_sq += d_sq
    return total_sq ** 0.5, per_layer


# ---------------------------------------------------------------------------
# Evaluation helpers with loss
# ---------------------------------------------------------------------------

def evaluate_task_with_loss(
    model, args, task_classes, pretrained_cfg, label_map, criterion=None
):
    """Like evaluate_task(), but also returns mean cross-entropy loss.

    Identical data pipeline, transforms, temperature scaling, and AMP as
    existing evaluate_task() in train_baseline.py / train_prm.py.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss().cuda()

    eval_transform = build_eval_transform(args.dataset, pretrained_cfg)
    target_transform = TargetTransform(label_map, map_to_local=False)
    eval_dataset = get_path_dataset(args.dataset, args.data_root, train=False)
    dataset = ClassIncrementalDataset(
        eval_dataset, task_classes, eval_transform, target_transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.eval_workers, pin_memory=True,
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for images, target in dataloader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.use_amp
            ):
                logits = model(images)
            # Loss uses the same temperature-scaled logits as training (CE)
            loss = criterion(logits / args.temperature, target)
            preds = logits.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)
            total_loss += float(loss.item()) * target.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Core sweep (shared by diagnostic Phase 1 and empirical Phase 2)
# ---------------------------------------------------------------------------

def _modify_head_for_eval(model, cl_manager, param_cache, task_id):
    """Inline copy of modify_head() from the training scripts.

    We copy rather than import to avoid adding cross-script imports. The
    behavior is exactly identical: assemble the concatenated evaluation head
    from the per-task cached head weights for tasks 0..task_id.
    """
    target_classes = cl_manager.sofar_task_classes

    if model.head.out_features == len(target_classes):
        return model

    old_head = deepcopy(model.head)
    dev = old_head.weight.device
    dtype = old_head.weight.dtype
    model.head = nn.Linear(
        old_head.in_features, len(target_classes),
        old_head.bias is not None, dev, dtype,
    )
    model.head.requires_grad_(old_head.weight.requires_grad)

    hw = torch.cat([
        param_cache[f"task_params_{t}"]["head.weight"].data.to(dev, dtype)
        for t in range(task_id + 1)
    ])
    model.head.weight.data = hw
    if old_head.bias is not None:
        hb = torch.cat([
            param_cache[f"task_params_{t}"]["head.bias"].data.to(dev, dtype)
            for t in range(task_id + 1)
        ])
        model.head.bias.data = hb
    return model


def _find_last_block(model):
    """Locate the last transformer block of a timm ViT for the F4-full feature hook.
    Works for both augreg ViT-B (.blocks) and CLIP ViT-B (.blocks)."""
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return model.blocks[-1]
    raise RuntimeError("function probe: cannot locate model.blocks[-1]")


def _build_function_probe_loader(
    args, cl_manager, pretrained_cfg, label_map, task_id,
    n_samples_total: int = 500, rng_seed: int = 0,
):
    """Fixed, reproducible eval subset over all seen tasks for function-space probes."""
    from torch.utils.data import Subset

    eval_transform = build_eval_transform(args.dataset, pretrained_cfg)
    target_transform = TargetTransform(label_map, map_to_local=False)
    eval_dataset = get_path_dataset(args.dataset, args.data_root, train=False)

    seen_classes = []
    for t in range(task_id + 1):
        seen_classes.extend(cl_manager.get_classes(t))

    wrapped = ClassIncrementalDataset(
        eval_dataset, seen_classes, eval_transform, target_transform,
    )
    n_total = len(wrapped)
    if n_total == 0:
        return None
    rng = np.random.default_rng(rng_seed)
    n_pick = int(min(n_samples_total, n_total))
    indices = rng.permutation(n_total)[:n_pick].tolist()
    subset = Subset(wrapped, indices)
    loader = DataLoader(
        subset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.eval_workers, pin_memory=True,
    )
    return loader


def _collect_function_state(model, loader, use_amp: bool):
    """One forward pass: capture CLS features, logits, argmax preds."""
    last_block = _find_last_block(model)
    buf = {}

    def _hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        buf["feat"] = (t[:, 0] if t.ndim == 3 else t).detach()

    handle = last_block.register_forward_hook(_hook)

    features_chunks, logits_chunks, preds_chunks = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for images, target in loader:
                images = images.cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    logits = model(images)
                features_chunks.append(buf["feat"].cpu().half())
                logits_chunks.append(logits.detach().cpu().half())
                preds_chunks.append(logits.argmax(dim=1).cpu().to(torch.int32))
    finally:
        handle.remove()

    return (
        torch.cat(features_chunks, dim=0),
        torch.cat(logits_chunks, dim=0),
        torch.cat(preds_chunks, dim=0),
    )


def _sweep_alphas_core(
    model, args, cl_manager, pretrained_cfg, label_map, param_cache,
    task_id, alpha_grid, function_probe_loader=None,
):
    """Sweep alpha along the interpolation line; return per-alpha metrics.

    Preconditions:
      - model has finished Stage 1 for task_id (cache_task_params already
        called, Fisher already accumulated if needed by caller)
      - ensure_alpha_slot(model, task_id) has been called so alphas[task_id]
        exists. Each sweep step only mutates alphas[task_id].data.

    Returns a dict with:
      alpha_grid        : list[float], the alphas probed
      task_loss         : list[list[float]], shape [len(grid)][task_id+1]
                          task_loss[k][i] = val CE loss on task i at alpha_k
      task_acc          : list[list[float]], same shape; per-task val acc

    When `function_probe_loader` is provided, also returns:
      function_features : np.ndarray [K, N, D] fp16 — CLS features per α
      function_logits   : np.ndarray [K, N, C] fp16 — logits per α
      function_preds    : np.ndarray [K, N]    int32 — argmax preds per α
    """
    # Freeze params for evaluation (does not modify trainable_mask permanently)
    saved_requires_grad = {}
    for name, p in model.named_parameters():
        saved_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)
    # LoRA modules' training flag (just for consistency)
    saved_lora_training = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            saved_lora_training.append((module, module.training))
            module.training = False

    # Assemble evaluation head for tasks 0..task_id
    model = _modify_head_for_eval(model, cl_manager, param_cache, task_id)
    model.cuda()
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    task_loss = []
    task_acc = []
    func_features, func_logits, func_preds = [], [], []

    for alpha in alpha_grid:
        set_task_alpha(model, task_id, alpha)
        per_task_loss = []
        per_task_acc = []
        for eval_tid in range(task_id + 1):
            eval_classes = cl_manager.get_classes(eval_tid)
            loss_i, acc_i = evaluate_task_with_loss(
                model, args, eval_classes, pretrained_cfg, label_map,
                criterion=criterion,
            )
            per_task_loss.append(float(loss_i))
            per_task_acc.append(float(acc_i))
        task_loss.append(per_task_loss)
        task_acc.append(per_task_acc)

        if function_probe_loader is not None:
            feats, logits, preds = _collect_function_state(
                model, function_probe_loader, use_amp=getattr(args, "use_amp", True),
            )
            func_features.append(feats.numpy())
            func_logits.append(logits.numpy())
            func_preds.append(preds.numpy())

    # Restore requires_grad state
    for name, p in model.named_parameters():
        p.requires_grad_(saved_requires_grad.get(name, False))
    for module, tflag in saved_lora_training:
        module.training = tflag

    try:
        model.remove_text_features()
    except Exception:
        pass

    out = {
        "alpha_grid": [float(a) for a in alpha_grid],
        "task_loss": task_loss,
        "task_acc": task_acc,
    }
    if function_probe_loader is not None:
        out["function_features"] = np.stack(func_features, axis=0)
        out["function_logits"] = np.stack(func_logits, axis=0)
        out["function_preds"] = np.stack(func_preds, axis=0)
    return out


# ---------------------------------------------------------------------------
# Aggregates + proxy objective
# ---------------------------------------------------------------------------

def _summarize_sweep(sweep_data, task_id):
    """Compute derived aggregates: current / old / seen averages per alpha."""
    task_loss = np.array(sweep_data["task_loss"])   # [N, task_id+1]
    task_acc = np.array(sweep_data["task_acc"])     # [N, task_id+1]

    L_cur = task_loss[:, task_id].tolist()
    Acc_cur = task_acc[:, task_id].tolist()

    if task_id >= 1:
        L_old = task_loss[:, :task_id].mean(axis=1).tolist()
        Acc_old = task_acc[:, :task_id].mean(axis=1).tolist()
    else:
        L_old = [None] * len(sweep_data["alpha_grid"])
        Acc_old = [None] * len(sweep_data["alpha_grid"])

    L_seen = task_loss.mean(axis=1).tolist()
    Acc_seen = task_acc.mean(axis=1).tolist()

    return {
        "L_cur": L_cur, "Acc_cur": Acc_cur,
        "L_old": L_old, "Acc_old": Acc_old,
        "L_seen": L_seen, "Acc_seen": Acc_seen,
    }


def _proxy_J(L_cur, alpha_grid, q_t, lam):
    """J(alpha) = L_t^val(alpha) + lambda * alpha^2 * q_t"""
    L_cur = np.asarray(L_cur, dtype=float)
    alphas = np.asarray(alpha_grid, dtype=float)
    return (L_cur + lam * (alphas ** 2) * float(q_t)).tolist()


def _default_alpha_grid(n: int = 20):
    """Evenly spaced alpha grid. Default plan: {0.00, 0.05, ..., 0.95} (n=20)."""
    if n <= 1:
        return [0.0]
    step = 1.0 / n
    return [round(i * step, 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Public API: Phase 1 diagnostic sweep
# ---------------------------------------------------------------------------

def run_alpha_sweep(
    model, args, cl_manager, pretrained_cfg, label_map, param_cache, fisher,
    task_id, save_path,
    alpha_grid=None, theory_alpha=None, chosen_alpha=None,
    tag="",
    function_probe: bool = False,
    function_probe_samples: int = 500,
    function_probe_save_path: str = "",
):
    """Phase 1 diagnostic alpha sweep.

    When `function_probe=True`, also collect CLS features / logits / argmax
    preds on a fixed seed-0 subset of the seen-so-far eval data at every α,
    and save them to `function_probe_save_path` as a compressed .npz.
    """
    if alpha_grid is None:
        alpha_grid = _default_alpha_grid(20)

    # 1) Create alpha slot (once)
    ensure_alpha_slot(model, task_id, init_alpha=float(alpha_grid[0]))

    # Build function-probe loader if requested (before sweep mutates model).
    probe_loader = None
    if function_probe:
        probe_loader = _build_function_probe_loader(
            args, cl_manager, pretrained_cfg, label_map, task_id,
            n_samples_total=int(function_probe_samples),
            rng_seed=0,
        )

    # 2) Core sweep
    raw = _sweep_alphas_core(
        model, args, cl_manager, pretrained_cfg, label_map, param_cache,
        task_id, alpha_grid, function_probe_loader=probe_loader,
    )

    # 3) q_t + ||delta_t||_F
    q_total, q_per_layer, delta_norm_F, delta_norm_per_layer = compute_q_t(
        model, fisher, task_id, return_delta_norms=True,
    )

    # 4) JSON dump
    summary = _summarize_sweep(raw, task_id)
    # Resolve chosen alpha for bookkeeping
    resolved_chosen = chosen_alpha
    if resolved_chosen is None:
        resolved_chosen = theory_alpha if theory_alpha is not None else alpha_grid[-1]
    record = {
        "tag": tag,
        "task_id_1indexed": int(task_id) + 1,
        "task_id_0indexed": int(task_id),
        "num_prev_tasks": int(task_id),
        "alpha_grid": raw["alpha_grid"],
        "per_task_loss": raw["task_loss"],
        "per_task_acc": raw["task_acc"],
        "L_cur": summary["L_cur"],
        "Acc_cur": summary["Acc_cur"],
        "L_old": summary["L_old"],
        "Acc_old": summary["Acc_old"],
        "L_seen": summary["L_seen"],
        "Acc_seen": summary["Acc_seen"],
        "q_t": float(q_total),
        "q_per_layer": {k: float(v) for k, v in q_per_layer.items()},
        "delta_norm_F": float(delta_norm_F),
        "delta_norm_per_layer": {k: float(v) for k, v in delta_norm_per_layer.items()},
        "lambda_prox": float(getattr(args, "prm_prox_weight", 0.0) or 0.0),
        "perturbation_eps": float(getattr(args, "eps", 0.0) or 0.0),
        "alpha_chosen": float(resolved_chosen),
        "theory_alpha": (float(theory_alpha) if theory_alpha is not None else None),
        "fixed_alpha_ref": 0.8,
        "seed": int(getattr(args, "seed", 2024)),
        "dataset": str(getattr(args, "dataset", "")),
        "model": str(getattr(args, "model", "")),
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  [alpha_sweep] saved {save_path}  (q_t={q_total:.4g})")

    if function_probe and "function_features" in raw:
        npz_path = function_probe_save_path or save_path.replace(".json", "_function.npz")
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez_compressed(
            npz_path,
            alpha_grid=np.asarray(raw["alpha_grid"], dtype=np.float32),
            features=raw["function_features"],
            logits=raw["function_logits"],
            preds=raw["function_preds"],
            alpha_chosen=np.float32(resolved_chosen),
            task_id=np.int32(task_id),
            tag=np.array(tag, dtype=object),
        )
        print(f"  [alpha_sweep] saved function probe {npz_path}"
              f"  (features {raw['function_features'].shape}, logits {raw['function_logits'].shape})")

    # 5) Leave alphas[task_id] at chosen value
    if chosen_alpha is None:
        chosen_alpha = theory_alpha if theory_alpha is not None else alpha_grid[-1]
    set_task_alpha(model, task_id, float(chosen_alpha))

    return record


# ---------------------------------------------------------------------------
# Public API: Phase 2 empirical alpha selection
# ---------------------------------------------------------------------------

def choose_empirical_alpha(
    model, args, cl_manager, pretrained_cfg, label_map, param_cache, fisher,
    task_id,
    lambda_proxy: float = 1.0,
    alpha_grid=None,
    save_path=None,
    tag="",
):
    """Pick alpha at task_id by minimizing J(alpha) over a 1D grid.

        J(alpha) = L_t^val(alpha) + lambda_proxy * alpha^2 * q_t

    L_t^val is evaluated on the current task's validation split (no
    rehearsal); q_t is the accumulated-Fisher interference strength.

    Flow mirrors run_alpha_sweep() but:
      - uses a coarser grid by default (11 points)
      - picks argmin_J(alpha) and leaves alphas[task_id] there
      - optionally dumps a JSON for later inspection

    Returns:
        alpha_star: float, the selected merge coefficient.
    """
    if task_id == 0:
        # Task 0 always writes alpha=1 and records old_weight; use merge helper.
        merge_with_alpha(model, task_id, 1.0)
        return 1.0

    if alpha_grid is None:
        # default coarse 11-point grid {0.0, 0.1, ..., 1.0}
        alpha_grid = [round(0.1 * i, 4) for i in range(11)]

    ensure_alpha_slot(model, task_id, init_alpha=float(alpha_grid[0]))

    raw = _sweep_alphas_core(
        model, args, cl_manager, pretrained_cfg, label_map, param_cache,
        task_id, alpha_grid,
    )

    q_total, q_per_layer, delta_norm_F, delta_norm_per_layer = compute_q_t(
        model, fisher, task_id, return_delta_norms=True,
    )
    L_cur = [row[task_id] for row in raw["task_loss"]]
    J = _proxy_J(L_cur, raw["alpha_grid"], q_total, lambda_proxy)
    k_star = int(np.argmin(J))
    alpha_star = float(raw["alpha_grid"][k_star])

    # Log
    print(
        f"  [empirical-alpha task {task_id+1}] "
        f"lambda={lambda_proxy:g} q_t={q_total:.4g}"
        f" -> alpha*={alpha_star:.4f} (J*={J[k_star]:.4f})"
    )

    # Leave alphas[task_id] at the chosen value
    set_task_alpha(model, task_id, alpha_star)

    if save_path is not None:
        summary = _summarize_sweep(raw, task_id)
        record = {
            "tag": tag,
            "task_id_1indexed": int(task_id) + 1,
            "task_id_0indexed": int(task_id),
            "num_prev_tasks": int(task_id),
            "alpha_grid": raw["alpha_grid"],
            "per_task_loss": raw["task_loss"],
            "per_task_acc": raw["task_acc"],
            "L_cur": summary["L_cur"],
            "Acc_cur": summary["Acc_cur"],
            "L_old": summary["L_old"],
            "Acc_old": summary["Acc_old"],
            "L_seen": summary["L_seen"],
            "Acc_seen": summary["Acc_seen"],
            "q_t": float(q_total),
            "q_per_layer": {k: float(v) for k, v in q_per_layer.items()},
            "delta_norm_F": float(delta_norm_F),
            "delta_norm_per_layer": {k: float(v) for k, v in delta_norm_per_layer.items()},
            "lambda_prox": float(getattr(args, "prm_prox_weight", 0.0) or 0.0),
            "perturbation_eps": float(getattr(args, "eps", 0.0) or 0.0),
            "lambda_proxy": float(lambda_proxy),
            "J_alpha": J,
            "alpha_star": alpha_star,
            "alpha_chosen": alpha_star,
            "J_star": float(J[k_star]),
            "seed": int(getattr(args, "seed", 2024)),
            "dataset": str(getattr(args, "dataset", "")),
            "model": str(getattr(args, "model", "")),
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(record, f, indent=2)

    return alpha_star
