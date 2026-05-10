"""Prefix save/load + Stage-1-only delta extraction for twin-train experiments.

Used by the Norm-Matched Control diagnostic (Causal Mechanism Diagnostics):
  1) Train a baseline (P&M) for tasks 0..t-1, then save a "prefix" — model
     state + param_cache + fisher + label_map + RNG state — at the boundary
     before task t starts.
  2) Resume from that prefix and run task t's Stage-1 training with two
     different prox weights (e.g. 0.0 = PM, 0.01 = PRM). Skip Stage-2
     write-back and dump the raw delta_t = B_t @ A_t per layer plus head[t]
     so they can be replayed later under arbitrary norm-matching transforms.

Both train_baseline.py and train_prm.py share these helpers.
"""
from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .lora import LoRALayer, apply_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gather_old_weights(model):
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer) and hasattr(mod, "old_weight"):
            out[name] = mod.old_weight.detach().cpu().clone()
    return out


def _restore_old_weights(model, old_weights):
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer) and name in old_weights:
            ow = old_weights[name]
            try:
                dev = next(mod.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
            mod.old_weight = ow.to(dev)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Prefix save / load / rebuild
# ---------------------------------------------------------------------------

def save_prefix(
    path: str,
    model,
    param_cache: Dict,
    fisher: Dict,
    label_map: Dict,
    pretrained_cfg,
    completed_taskid: int,
    args,
    extra: Optional[Dict] = None,
):
    """Persist enough state to resume training of task `completed_taskid + 1`.

    Should be called BEFORE any apply_lora / training is done for the next
    task (i.e., right at the top of the loop iteration that would train
    task = completed_taskid + 1).
    """
    payload = {
        "completed_taskid": int(completed_taskid),
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "old_weights": _gather_old_weights(model),
        "param_cache": _to_cpu(param_cache),
        "fisher": _to_cpu(fisher),
        "label_map": dict(label_map),
        "pretrained_cfg": pretrained_cfg,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(
        f"[prefix_io] saved prefix completed_taskid={completed_taskid} -> {path} "
        f"(state_keys={len(payload['model_state_dict'])}, "
        f"fisher_layers={len(payload['fisher'])}, "
        f"old_weights={len(payload['old_weights'])}, "
        f"param_cache_tasks={sum(1 for k in param_cache if k.startswith('task_params_'))})"
    )


def load_prefix(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def rebuild_model_from_prefix(loaded: Dict, args, timm_create_model_fn):
    """Reconstruct model + state ready to start Stage-1 of task completed+1.

    Steps:
      1. Build a fresh pretrained model.
      2. apply_lora for tasks 0..completed (so per-task lora_A/lora_B lists
         are sized correctly).
      3. Pre-allocate alphas/gates ParameterList entries (placeholder=1.0)
         so state_dict load matches.
      4. Resize model.head to the saved shape.
      5. Load model_state_dict (strict=False to tolerate buffer renames).
      6. Restore old_weight LoRA attributes.

    Returns: (model, pretrained_cfg, completed_taskid).
    """
    completed = int(loaded["completed_taskid"])

    other_args = {
        "logit_type": args.logit_type,
        "logit_scale": args.logit_scale,
        "logit_scale_trainable": False,
    }
    model = timm_create_model_fn(
        args.model,
        pretrained=True,
        pretrained_strict=False,
        num_classes=1,  # placeholder, will be resized below
        other_args_dict=other_args,
    )
    pretrained_cfg = (
        deepcopy(loaded.get("pretrained_cfg")) if loaded.get("pretrained_cfg") is not None
        else deepcopy(model.pretrained_cfg)
    )

    # Apply LoRA for tasks 0..completed AND grow alphas in lockstep so that
    # LoRALayer.update()'s `task_id+1 == len(alphas)` invariant holds across
    # the chain. (Otherwise apply_lora for tid>=1 silently no-ops.)
    for tid in range(completed + 1):
        apply_lora(
            model, tid,
            rank=args.rank, eps=args.eps,
            num_tasks=args.num_tasks, module=args.module,
        )
        # Append one alpha (placeholder=1.0; load_state_dict will overwrite).
        for mod in model.modules():
            if isinstance(mod, LoRALayer):
                mod.cal_alpha(1.0)

    # Resize head to saved shape.
    sd = loaded["model_state_dict"]
    head_w = sd["head.weight"]
    n_out, in_feat = head_w.shape[0], head_w.shape[1]
    has_bias = "head.bias" in sd
    model.head = nn.Linear(in_feat, n_out, bias=has_bias)

    # Load weights (strict=False to ignore buffers like text_features that
    # may have been removed via remove_text_features() before saving).
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[prefix_io] load_state_dict missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        if missing[:3]:
            print(f"  missing[:3]: {missing[:3]}")
        if unexpected[:3]:
            print(f"  unexpected[:3]: {unexpected[:3]}")

    # Re-attach old_weight attrs (not in state_dict).
    _restore_old_weights(model, loaded["old_weights"])

    model = model.cuda()
    return model, pretrained_cfg, completed


def restore_rng(loaded: Dict):
    """Restore deterministic RNG state captured in the prefix payload."""
    rng = loaded.get("rng") or {}
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("torch_cuda_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "random" in rng:
        random.setstate(rng["random"])


# ---------------------------------------------------------------------------
# Stage-1 delta extraction
# ---------------------------------------------------------------------------

def save_stage1_delta(
    path: str,
    model,
    taskid: int,
    seed: int,
    method: str,
    prox_weight: float = 0.0,
    train_loss_final: Optional[float] = None,
    q_t: Optional[float] = None,
    delta_norm_F: Optional[float] = None,
    delta_norm_per_layer: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    """Dump per-layer delta_t = B_t @ A_t and the per-task head for replay.

    `model` is expected to have just finished Stage-1 of task `taskid` (the
    LoRA factors are populated but no Stage-2 write-back has been applied).
    """
    deltas = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALayer):
                if taskid >= len(mod.lora_A) or taskid >= len(mod.lora_B):
                    continue
                w = mod.lora_B[taskid].weight @ mod.lora_A[taskid].weight
                deltas[name] = w.detach().cpu().clone()

    head_w = model.head.weight.detach().cpu().clone()
    head_b = (
        model.head.bias.detach().cpu().clone()
        if model.head.bias is not None else None
    )

    payload = {
        "taskid": int(taskid),
        "method": str(method),
        "seed": int(seed),
        "prox_weight": float(prox_weight),
        "delta_per_layer": deltas,
        "head_weight": head_w,
        "head_bias": head_b,
        "train_loss_final": (
            float(train_loss_final) if train_loss_final is not None else None
        ),
        "q_t": float(q_t) if q_t is not None else None,
        "delta_norm_F": float(delta_norm_F) if delta_norm_F is not None else None,
        "delta_norm_per_layer": (
            {k: float(v) for k, v in delta_norm_per_layer.items()}
            if delta_norm_per_layer else None
        ),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(
        f"[prefix_io] saved Stage-1 delta task={taskid} method={method} "
        f"seed={seed} prox={prox_weight} -> {path} "
        f"(layers={len(deltas)}, head_out={head_w.shape[0]}, "
        f"delta_norm_F={delta_norm_F})"
    )
