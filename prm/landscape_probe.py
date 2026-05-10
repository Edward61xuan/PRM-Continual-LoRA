"""F8-landscape: save/load utilities for post-hoc loss-surface visualization.

Wave-A training did not persist any model state. For the 3D
`loss-landscape + trajectory` figure we need to reconstruct θ(α) offline:

    θ(α) = θ_{t-1} + α · (B_t @ A_t)

This module provides:

  save_task_checkpoint(path, model, param_cache, task_id, method, seed)
      After each task's write-back, dump model.state_dict() + param_cache +
      LoRALayer.old_weight attributes (those are NOT in state_dict()).

  load_task_checkpoint(path) -> dict
      Inverse; returns a dict ready for model.load_state_dict(...).

  flatten_lora_delta(model) -> np.ndarray
      Flatten the current LoRA delta weight of every (block, k/v) LoRALayer
      into one 1D numpy fp32 vector. Used as per-sample input to PCA.

  apply_lora_delta_offset(model, direction_flat, scale)
      In-place mutate each LoRALayer's last lora_A to add `scale * dir` of
      the flat direction, restricted to the LoRA-delta subspace. For PC-grid
      landscape eval.

"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

from .lora import LoRALayer


def _iter_lora_layers(model):
    """Yield (dotted_name, LoRALayer) for every LoRALayer in the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            yield name, module


def save_task_checkpoint(path: str, model, param_cache: Dict, task_id: int,
                         method: str, seed: int, extra: Dict = None):
    """Persist enough state to reconstruct θ(α) at any α for this task."""
    old_weights = {}
    for name, lora in _iter_lora_layers(model):
        if hasattr(lora, "old_weight"):
            old_weights[name] = lora.old_weight.detach().cpu().clone()

    # Materialize param_cache onto CPU to keep checkpoint portable.
    cpu_param_cache = {}
    for k, v in param_cache.items():
        if isinstance(v, dict):
            cpu_param_cache[k] = {kk: vv.detach().cpu().clone()
                                  if isinstance(vv, torch.Tensor) else vv
                                  for kk, vv in v.items()}
        elif isinstance(v, torch.Tensor):
            cpu_param_cache[k] = v.detach().cpu().clone()
        else:
            cpu_param_cache[k] = v

    payload = {
        "task_id": int(task_id),
        "method": str(method),
        "seed": int(seed),
        "model_state_dict": {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()},
        "param_cache": cpu_param_cache,
        "old_weights": old_weights,
    }
    if extra:
        payload.update(extra)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"  [landscape_probe] saved ckpt task={task_id} method={method} "
          f"seed={seed} -> {path} (state_keys={len(payload['model_state_dict'])}, "
          f"old_weights={len(old_weights)})")


def load_task_checkpoint(path: str) -> Dict:
    """Inverse of save_task_checkpoint. Returns a dict with the same keys."""
    return torch.load(path, map_location="cpu")


def restore_old_weights(model, old_weights: Dict[str, torch.Tensor]):
    """Re-attach `old_weight` attributes to each LoRALayer from a loaded ckpt."""
    for name, lora in _iter_lora_layers(model):
        if name in old_weights:
            lora.old_weight = old_weights[name].to(
                next(lora.parameters()).device
            )


def flatten_lora_delta(model) -> np.ndarray:
    """Concat every LoRALayer's B_t @ A_t (last-task delta) into a 1D np vector.

    Used as the per-sample PCA input for the landscape viz. Only the
    CURRENT task's (B_t, A_t) is collected; previous tasks' information is
    implicit in the merged weight (recoverable from alphas × B_i @ A_i).
    """
    vecs = []
    with torch.no_grad():
        for _, lora in _iter_lora_layers(model):
            t = lora.task_id
            delta = (lora.lora_B[t].weight @ lora.lora_A[t].weight)  # (out, in)
            vecs.append(delta.detach().cpu().float().reshape(-1).numpy())
    if not vecs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(vecs, axis=0).astype(np.float32, copy=False)


def flatten_net_lora_delta(model) -> np.ndarray:
    """Concat every LoRALayer's TOTAL effective delta
    (sum_i alpha_i * B_i A_i + current-task contribution) into a 1D np vector.

    This is the "net perturbation from pretrained" at the current state —
    the right quantity for landscape PCA.
    """
    vecs = []
    with torch.no_grad():
        for _, lora in _iter_lora_layers(model):
            delta = lora.get_weight(perturbation=0)  # (out, in) total effective
            vecs.append(delta.detach().cpu().float().reshape(-1).numpy())
    if not vecs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(vecs, axis=0).astype(np.float32, copy=False)


def collect_flat_split_sizes(model) -> List[int]:
    """Per-LoRALayer element counts, so we can split a flat vector back."""
    sizes = []
    for _, lora in _iter_lora_layers(model):
        sizes.append(lora.out_dim * lora.in_dim)
    return sizes


def apply_lora_delta_offset(model, flat_direction: np.ndarray, scale: float):
    """Shift each LoRALayer's B·A delta by `scale * flat_direction` (restricted
    to the LoRA delta subspace) by overwriting lora_B[t].weight. A[t] is left
    identity-ish so the product equals the target.

    Implementation: since we can't guarantee a valid B·A factorization of an
    arbitrary offset delta, we instead TEMPORARILY OVERWRITE each LoRALayer's
    effective last-task weight by monkey-patching get_weight() to add the
    offset. Cleaner alternative: replace lora_B[t].weight with
    (offset @ A_pinv), but rank-10 A pseudo-inverse is ill-conditioned.

    We take the monkey-patch route — set a per-layer attribute
    `_landscape_offset` that get_weight picks up. Also save the original
    get_weight so we can restore.
    """
    sizes = collect_flat_split_sizes(model)
    total = sum(sizes)
    if flat_direction.size != total:
        raise ValueError(
            f"flat_direction size {flat_direction.size} != expected {total}"
        )
    offset_parts = np.split(flat_direction, np.cumsum(sizes)[:-1])

    for (_, lora), off in zip(_iter_lora_layers(model), offset_parts):
        off_tensor = torch.from_numpy(off.reshape(lora.out_dim, lora.in_dim)) \
            .to(next(lora.parameters()).device) \
            .to(lora.lora_B[lora.task_id].weight.dtype) * float(scale)
        lora._landscape_offset = off_tensor


def clear_lora_delta_offset(model):
    for _, lora in _iter_lora_layers(model):
        if hasattr(lora, "_landscape_offset"):
            del lora._landscape_offset


def _patched_get_weight(orig_get_weight, lora_self):
    """Return a function that adds `_landscape_offset` to orig weight."""
    def wrapped(perturbation=0):
        w = orig_get_weight(perturbation=perturbation)
        if hasattr(lora_self, "_landscape_offset"):
            w = w + lora_self._landscape_offset
        return w
    return wrapped


def install_offset_hook(model):
    """Monkey-patch every LoRALayer.get_weight to honor `_landscape_offset`.

    Idempotent (re-installs cleanly). Call `uninstall_offset_hook` to reverse.
    """
    for _, lora in _iter_lora_layers(model):
        if not hasattr(lora, "_orig_get_weight"):
            lora._orig_get_weight = lora.get_weight
        lora.get_weight = _patched_get_weight(lora._orig_get_weight, lora)


def uninstall_offset_hook(model):
    for _, lora in _iter_lora_layers(model):
        if hasattr(lora, "_orig_get_weight"):
            lora.get_weight = lora._orig_get_weight
            del lora._orig_get_weight
        if hasattr(lora, "_landscape_offset"):
            del lora._landscape_offset
