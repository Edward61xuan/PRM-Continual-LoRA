#!/usr/bin/env python3
"""
PRM: Proximity-Regularized Merging
==================================

Train PRM-fixed or PRM-Fisher on class-incremental image benchmarks.

Algorithm Overview
------------------
PRM decomposes rehearsal-free continual learning into 3 stages per task t:

  Stage 1 -- Fast Proposal:
      Train a proximity-regularized LoRA adapter (the "fast branch"):
          (A_t, B_t) = argmin L_CE(x; theta_{t-1} + B_t A_t)
                         + lambda_w * (||A_t - A_t^(0)||^2
                                       + ||B_t - B_t^(0)||^2)
      A_t^(0), B_t^(0) are the freshly allocated task-t initializations.
      Perturbation-aware training samples perturbation in {0, +1, -1} to
      produce the noise-injected weight: (1 + perturbation * eps) * B_t @ A_t

  Stage 2 -- Write-Back:
      Merge the fast branch into the shared model:
          theta_t = theta_{t-1} + alpha_t * (B_t @ A_t)
      PRM-fixed uses a constant alpha (default 0.8).
      PRM-Fisher uses P&M's Fisher-derived per-task coefficient.

Architecture: ViT-B/16 + per-task LoRA (rank 10) on K and V.

Usage
-----
    python train_prm.py \\
        --dataset imagenet_r \\
        --data_root /path/to/imagenet-r \\
        --num_tasks 10 \\
        --epochs 10 \\
        --prm_alpha 0.8 \\
        --prm_prox_weight 0.01

References
----------
    PRM: Proximity-Regularized Merging for rehearsal-free continual learning.
"""

import os
import sys
import argparse
import random
import json
import platform
import subprocess
import tempfile
from importlib import metadata as importlib_metadata
from time import time as ttime
from copy import deepcopy
from collections import OrderedDict

os.environ.setdefault("TIMM_FUSED_ATTN", "0")

import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import tqdm
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from prm.lora import LoRALayer, QkvWithLoRA, apply_lora
from prm.landscape_probe import save_task_checkpoint
from prm.prefix_io import (
    save_prefix,
    load_prefix,
    rebuild_model_from_prefix,
    restore_rng,
    save_stage1_delta,
)
from prm.prm_ops import (
    proximity_loss,
    get_lora_params_snapshot,
    prm_write_back,
    compute_prm_fisher_alpha,
    compute_prm_budgeted_alpha,
    compute_writeability_gate,
    prm_write_back_gated,
    apply_direction_gate,
    consolidate,
)
from prm.fisher import init_fisher_storage, accumulate_fisher
from prm.alpha_sweep import (
    run_alpha_sweep,
    choose_empirical_alpha,
    merge_with_alpha,
    _default_alpha_grid,
    compute_q_t,
    append_per_task_summary,
)
from prm.phase4 import (
    choose_empirical_alpha_2d,
    apply_2d_fixed,
)
from prm.data import (
    get_path_dataset,
    ClassIncrementalDataset,
    make_label_maps,
    TargetTransform,
    build_train_transform,
    build_eval_transform,
)
from prm.continual import ClassIncrementalManager
from prm.metrics import (
    calc_accuracy,
    calc_aaa,
    calc_bwt,
    calc_forgetting,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args(argv=None):
    p = argparse.ArgumentParser(description="PRM-fixed / PRM-Fisher continual learning")

    # Dataset
    p.add_argument("-d", "--dataset", type=str, required=True,
                   choices=("cifar100", "imagenet_r", "sdomainet", "cub200"))
    p.add_argument("-dr", "--data_root", type=str, required=True,
                   help="Root dir with train/ and val/ subdirectories")
    p.add_argument("-t", "--num_tasks", type=int, default=10)
    p.add_argument("--shuffle_classes", action=argparse.BooleanOptionalAction,
                   default=True, help="Shuffle classes with PCG64 (disable with --no-shuffle_classes)")
    p.add_argument("--seed", type=int, default=2024)

    # Model
    p.add_argument("-m", "--model", type=str,
                   default="vit_base_patch16_clip_quickgelu_224.openai")
    p.add_argument("--logit_type", type=str, default="head_out",
                   choices=("head_out", "sim_imgtext"))
    p.add_argument("--logit_scale", type=float, default=4.605170249938965)
    p.add_argument("--pretrained_checkpoint", type=str, default="",
                   help="Optional local timm checkpoint; bypasses network lookup")

    # LoRA
    p.add_argument("--rank", type=int, default=10, help="LoRA rank")
    p.add_argument("--eps", type=float, default=0.5, help="Perturbation magnitude")
    p.add_argument("--p0", type=float, default=0.33,
                   help="Probability weight for +1/-1 perturbation sampling")
    p.add_argument("--module", type=str, default="kv",
                   help="Which projections to adapt: kv, qkv, etc.")

    # Training
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_scale", type=float, default=10.0,
                   help="Head LR multiplier")
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--lr_sch", type=str, default="multistep",
                   choices=("cosine", "step", "multistep"))
    p.add_argument("--decay_milestones", type=int, nargs="+", default=[5, 8])
    p.add_argument("--decay_rate", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--temperature", type=float, default=28.0)
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use CUDA automatic mixed precision (disable with --no-use_amp)")
    p.add_argument("--expand_times", type=int, default=10)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eval_workers", type=int, default=2)
    p.add_argument("--show_bar", action="store_true")

    # PRM parameters
    p.add_argument("--prm_method", type=str, default="fixed",
                   choices=("fixed", "fisher"),
                   help="Published variant: PRM-fixed or PRM-Fisher")
    p.add_argument("--prm_alpha", type=float, default=0.8,
                   help="Fixed merge coefficient for PRM-fixed write-back")
    p.add_argument("--prm_prox_weight", type=float, default=0.01,
                   help="Proximity regularization weight (lambda_w)")

    # Output
    p.add_argument("--results_file", type=str, default="",
                   help="Path to save JSON results")
    p.add_argument("--fisher_diagnostics", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Compute and retain per-task Fisher/q_t for fixed runs. "
                        "Fisher coefficients and alpha sweeps always require Fisher.")

    # Diagnostic alpha sweep for the published coefficient rules.
    p.add_argument("--alpha_sweep_tasks", type=int, nargs="+", default=[],
                   help="1-indexed task IDs at which to run the diagnostic "
                        "alpha sweep (e.g. 2 5 10). Empty = no sweep.")
    p.add_argument("--alpha_grid_n", type=int, default=20,
                   help="Number of alpha points in the diagnostic sweep grid "
                        "(default 20 => {0.00, 0.05, ..., 0.95}).")
    p.add_argument("--alpha_sweep_out_dir", type=str, default="",
                   help="Directory to dump per-task sweep JSONs.")
    p.add_argument("--function_probe", action="store_true",
                   help="F4-full: save CLS features + logits + argmax preds at "
                        "each alpha to a companion .npz beside each sweep JSON.")
    p.add_argument("--function_probe_samples", type=int, default=500,
                   help="Number of samples in the F4-full function probe subset.")
    p.add_argument("--save_landscape_ckpt_dir", type=str, default="",
                   help="F8-landscape: if set, dump model+param_cache+old_weights"
                        " to <dir>/ckpt_task{t}_seed{s}.pt after each task.")
    p.add_argument("--save_landscape_per_epoch_task", type=int, default=-1,
                   help="F8-landscape: dump per-epoch Stage-1 snapshot at "
                        "<dir>/ckpt_task{t}_epoch{k}_seed{s}.pt for the listed "
                        "1-indexed task id.")
    p.add_argument("--save_landscape_only_task", type=int, default=-1,
                   help="If >0 (1-indexed), only save the post-task ckpt for "
                        "this task; skip all others. Saves disk for mechanism "
                        "diagnostics that only need one task's checkpoint.")

    # Twin-train infrastructure (Norm-Matched Control diagnostic)
    p.add_argument("--save_prefix_at_task", type=int, default=-1,
                   help="If >0 (1-indexed), abort training right BEFORE this "
                        "task starts and dump the model+param_cache+fisher+"
                        "label_map+RNG state to <save_prefix_dir>/prefix_t{t}.pt"
                        " for later resume.")
    p.add_argument("--save_prefix_dir", type=str, default="",
                   help="Directory in which to save the prefix payload.")
    p.add_argument("--resume_prefix", type=str, default="",
                   help="Path to a prefix payload from --save_prefix_at_task. "
                        "Loads model+state, skips already-completed tasks, and "
                        "trains only --train_task.")
    p.add_argument("--train_task", type=int, default=-1,
                   help="When --resume_prefix is set, the 1-indexed task to "
                        "train (must equal completed_taskid+2).")
    p.add_argument("--stage1_only_save_delta", type=str, default="",
                   help="If set (resume mode only), skip Stage-2 write-back "
                        "and Stage-3 consolidation, then dump (delta_t per "
                        "layer, head_t) to <dir>/delta_task{t}_seed{s}.pt and"
                        " exit.")

    # Defaults retained for internal diagnostic code paths that are not part of
    # the public release CLI. Published runs only accept fixed or fisher above.
    p.set_defaults(
        drift_budget=5.0, gate_alpha_base="fisher", gate_a=1.0, gate_b=1.0,
        gate_c=1.0, gate_late_k=3, gate_no_rescale=False, gate_mode="element",
        consolidation=False, consolidation_epochs=2, consolidation_lr=1e-4,
        empirical_lambda=1.0, empirical_grid_n=11, empirical_sweep_out_dir="",
        d2_mode="empirical", d2_alpha_s=1.0, d2_alpha_r=1.0,
        d2_alpha_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], d2_late_k=3,
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def needs_fisher(args):
    """Sweeps require the full history, including tasks before the first sweep."""
    return (
        args.prm_method in ("fisher", "gated", "empirical", "2d")
        or args.fisher_diagnostics
        or bool(args.alpha_sweep_tasks)
    )


def validate_fisher_history(model, fisher, taskid):
    """A fixed-only prefix cannot supply missing historical Fisher statistics."""
    for module in model.modules():
        if isinstance(module, QkvWithLoRA):
            missing = [i for i in range(taskid) if i not in fisher.get(str(module.layer), {})]
            if missing:
                raise ValueError(
                    f"Missing Fisher history for layer {module.layer}, tasks {missing}. "
                    "Regenerate the prefix with --fisher_diagnostics or --prm_method fisher."
                )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    """Atomically write JSON after creating the destination's parent directory."""
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent,
            prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def local_checkpoint_overlay(path):
    """Build a timm file overlay while preserving a symlink's file suffix."""
    if not path:
        return None
    checkpoint_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {checkpoint_path}")
    return {"file": checkpoint_path}


def _git_metadata():
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _cached_hf_checkpoint(repo_id, filename=None):
    """Return a locally cached checkpoint path/revision without network I/O."""
    if not repo_id:
        return None, None
    try:
        from huggingface_hub import try_to_load_from_cache
        candidates = [filename] if filename else ["model.safetensors", "pytorch_model.bin"]
        for candidate in candidates:
            cached = try_to_load_from_cache(repo_id, candidate)
            if isinstance(cached, str) and os.path.isfile(cached):
                parts = os.path.abspath(cached).split(os.sep)
                revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
                return os.path.realpath(cached), revision
    except (ImportError, OSError, ValueError):
        pass
    return None, None


def collect_run_metadata(args, cl_manager, pretrained_cfg):
    """Collect provenance without changing the training or evaluation path."""
    commit, dirty = _git_metadata()
    split_metadata = {}
    for train, split in ((True, "train"), (False, "val")):
        dataset = get_path_dataset(args.dataset, args.data_root, train=train)
        per_class = {str(cls): len(paths) for cls, paths in dataset.class_imgpath_dict.items()}
        split_metadata[split] = {
            "images": int(sum(per_class.values())),
            "classes": int(len(per_class)),
            "images_per_class": per_class,
        }

    cached_path, cached_revision = _cached_hf_checkpoint(
        pretrained_cfg.get("hf_hub_id"), pretrained_cfg.get("hf_hub_filename")
    )
    checkpoint = {
        "model": str(args.model),
        "local_path": (
            os.path.realpath(os.path.expanduser(args.pretrained_checkpoint))
            if args.pretrained_checkpoint else None
        ),
        "hf_hub_id": pretrained_cfg.get("hf_hub_id"),
        "hf_hub_filename": pretrained_cfg.get("hf_hub_filename"),
        "cached_path": cached_path,
        "cached_revision": cached_revision,
        "url": pretrained_cfg.get("url"),
    }
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count_visible": int(torch.cuda.device_count()),
        }
    return {
        "config": dict(vars(args)),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "pillow": importlib_metadata.version("Pillow"),
            "huggingface_hub": importlib_metadata.version("huggingface-hub"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": gpu,
        },
        "dataset_metadata": {
            "root": os.path.realpath(os.path.expanduser(args.data_root)),
            **split_metadata,
        },
        "class_order": [int(x) for x in cl_manager.all_classes],
        "task_classes": [[int(x) for x in task] for task in cl_manager.task_class_list],
        "checkpoint": checkpoint,
    }


def set_model_mode(model, training, taskid=0):
    """Configure which parameters are trainable and which modules are in train mode."""
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if training:
            if "logit_scale" in name:
                p.requires_grad_(True)
            if "head" in name:
                p.requires_grad_(True)
            elif "lora_A" in name:
                if str(taskid) in name.split("lora_A.")[1]:
                    p.requires_grad_(True)
            elif "lora_B" in name:
                if str(taskid) in name.split("lora_B.")[1]:
                    p.requires_grad_(True)

    model.eval()
    if training:
        for n, m in model.named_modules():
            if any(s in n for s in ("lora", "head")) and not isinstance(m, nn.Identity):
                m.train()

    model.cuda()
    return model


def set_learning_rates(model, base_lr, lr_scale):
    """Create parameter groups with scaled LR for the head."""
    groups = [
        {"params": [], "lr": base_lr},
        {"params": [], "lr": lr_scale * base_lr},
    ]
    for name, p in model.named_parameters():
        if p.requires_grad:
            idx = 1 if "head" in name else 0
            groups[idx]["params"].append(p)
    return groups


def modify_head(model, target_classes, param_cache, training, current_taskid):
    """Resize the classification head for training (per-task) or evaluation (all seen)."""
    if model.head.out_features != len(target_classes):
        old_head = deepcopy(model.head)
        dev = old_head.weight.device
        dtype = old_head.weight.dtype
        model.head = nn.Linear(
            old_head.in_features, len(target_classes),
            old_head.bias is not None, dev, dtype,
        )
        model.head.requires_grad_(old_head.weight.requires_grad)

        if not training:
            # Assemble evaluation head from cached per-task heads
            hw = torch.cat([
                param_cache[f"task_params_{t}"]["head.weight"].data.to(dev, dtype)
                for t in range(current_taskid + 1)
            ])
            model.head.weight.data = hw
            if old_head.bias is not None:
                hb = torch.cat([
                    param_cache[f"task_params_{t}"]["head.bias"].data.to(dev, dtype)
                    for t in range(current_taskid + 1)
                ])
                model.head.bias.data = hb
    return model


def cache_task_params(model, taskid, param_cache):
    """Save trainable parameters for the current task."""
    task_params = OrderedDict()
    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
    param_cache[f"task_params_{taskid}"] = task_params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, dataloader, criterion, optimizer, args, taskid, prox_snapshot=None
):
    """Run one epoch of PRM-fixed training with perturbation sampling."""
    amp_scaler = GradScaler(enabled=args.use_amp)
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    for images, target in tqdm.tqdm(
        dataloader, total=len(dataloader), dynamic_ncols=True,
        disable=not args.show_bar,
    ):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            # Perturbation sampling: p(0) = 1-2*p0, p(+1) = p0, p(-1) = p0
            probs = (1 - args.p0 * 2, args.p0, args.p0)
            perturbation = random.choices([0, 1, -1], weights=probs, k=1)[0]
            logits = model(images, perturbation=perturbation)

        ce_loss = criterion(logits / args.temperature, target)

        # Stage 1: Proximity regularization
        if args.prm_prox_weight > 0 and prox_snapshot is not None:
            prox_loss = proximity_loss(model, prox_snapshot)
            total_loss = ce_loss + args.prm_prox_weight * prox_loss
        else:
            total_loss = ce_loss

        optimizer.zero_grad()
        amp_scaler.scale(total_loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        preds = logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += images.size(0)
        total_loss_sum += ce_loss.item() * images.size(0)

    avg_loss = total_loss_sum / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_one_task(model, args, cl_manager, label_map, pretrained_cfg, param_cache, fisher, taskid, task_classes, per_task_meta=None):
    """Stage 1 + 2: Fast Proposal then Write-Back for one task.

    If per_task_meta is a dict, populates per_task_meta[taskid] with
    {q_t, delta_norm_F, alpha_chosen, alpha_method, train_loss_final}.
    """
    print(f"\n{'='*80}")
    print(f"Task [{taskid+1}/{cl_manager.num_tasks}] -- Classes: {task_classes}")
    print(f"{'='*80}")
    use_fisher = needs_fisher(args)
    if use_fisher:
        validate_fisher_history(model, fisher, taskid)

    # Update label maps
    g2l_map = make_label_maps(taskid, task_classes)
    label_map.update(g2l_map)

    # Configure model
    model = set_model_mode(model, training=True, taskid=taskid)
    model = modify_head(model, task_classes, param_cache, training=True, current_taskid=taskid)

    # Build dataset with per-task label mapping
    train_transform = build_train_transform(args.dataset, pretrained_cfg)
    target_transform = TargetTransform(label_map, map_to_local=True)
    train_dataset = get_path_dataset(args.dataset, args.data_root, train=True)
    dataset = ClassIncrementalDataset(
        train_dataset, task_classes, train_transform, target_transform,
        expand_times=args.expand_times,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    criterion = nn.CrossEntropyLoss().cuda()

    # Ordinary fixed runs allocate no Fisher tensors.
    if use_fisher:
        for module in model.modules():
            if isinstance(module, QkvWithLoRA):
                name = f"{module.layer}"
                fisher[name][taskid] = torch.zeros_like(
                    module.qkv.weight.data
                ).to(module.qkv.weight.device)

    # Set up optimizer and scheduler
    param_groups = set_learning_rates(model, args.lr, args.lr_scale)
    optimizer = create_optimizer_v2(
        param_groups, opt="adamW", lr=args.lr,
        weight_decay=args.weight_decay, foreach=True,
    )
    scheduler, num_epochs = create_scheduler_v2(
        optimizer, sched=args.lr_sch, num_epochs=args.epochs,
        decay_epochs=1000, decay_milestones=args.decay_milestones,
        decay_rate=args.decay_rate, min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.4%})")

    # Stage 1: Fast Proposal with proximity regularization
    prox_snapshot = get_lora_params_snapshot(model)

    # For GatedPRM and Phase 4 2D: collect (B_t @ A_t) snapshots during the
    # last K epochs. GatedPRM uses them for the Stability signal; 2D uses them
    # to build delta_shared = mean(snapshots).
    late_snapshots = {}
    _use_snaps = args.prm_method in ("gated", "2d")
    _late_k = args.d2_late_k if args.prm_method == "2d" else args.gate_late_k
    gate_late_start = (args.epochs - _late_k + 1) if _use_snaps else None

    train_loss_final = None
    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)
        loss, acc = train_one_epoch(
            model, dataloader, criterion, optimizer, args, taskid,
            prox_snapshot=prox_snapshot,
        )
        print(f"  Epoch [{epoch}/{args.epochs}] loss={loss:.4f} acc={acc:.2%}")
        train_loss_final = float(loss)

        if (args.save_landscape_ckpt_dir
                and args.save_landscape_per_epoch_task == taskid + 1):
            ep_path = os.path.join(
                args.save_landscape_ckpt_dir,
                f"ckpt_task{taskid+1:02d}_epoch{epoch:02d}_seed{args.seed}.pt",
            )
            save_task_checkpoint(
                ep_path, model, param_cache, taskid,
                method=f"prm_{args.prm_method}", seed=args.seed,
                extra={"stage": "stage1_mid_epoch", "epoch": int(epoch),
                       "train_loss": train_loss_final, "train_acc": float(acc)},
            )

        if gate_late_start is not None and epoch >= gate_late_start:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, LoRALayer):
                        key = f"{module.layer}_{module.pos}"
                        w_snap = (
                            module.lora_B[taskid].weight
                            @ module.lora_A[taskid].weight
                        ).detach().clone()
                        late_snapshots.setdefault(key, []).append(w_snap)

    # Cache per-task parameters (head weights, LoRA params)
    cache_task_params(model, taskid, param_cache)

    # This extra data pass also consumes RNG (shuffle/augmentation). Skipping
    # it changes subsequent RNG state relative to historical diagnostic runs.
    if use_fisher:
        accumulate_fisher(model, dataloader, criterion, fisher, taskid)
    else:
        # Fisher used to switch the model to eval mode as a side effect.
        # Keep write-in/evaluation frozen even when that pass is skipped.
        set_model_mode(model, training=False)

    # Stage 2: Write-Back
    # Capture raw Stage-1 interference and update magnitude BEFORE any
    # write-back / gate overwrite so the logged values reflect the
    # pre-merge state. compute_q_t only reads fisher[0..taskid-1] which
    # is populated above, and delta_t = B_t @ A_t from the just-trained
    # LoRA block.
    if use_fisher:
        _q_t, _q_per_layer, _delta_norm_F, _delta_norm_per_layer = compute_q_t(
            model, fisher, taskid, return_delta_norms=True,
        )
    else:
        _q_t, _q_per_layer = None, {}
        _delta_norm_F, _delta_norm_per_layer = None, {}
    _alpha_chosen = None
    _alpha_method = args.prm_method

    # Twin-train Stage-1-only path: dump delta + head and skip Stage 2/3.
    if args.stage1_only_save_delta:
        os.makedirs(args.stage1_only_save_delta, exist_ok=True)
        delta_path = os.path.join(
            args.stage1_only_save_delta,
            f"delta_task{taskid+1:02d}_seed{args.seed}.pt",
        )
        method_tag = (
            f"prm_{args.prm_method}_prox{args.prm_prox_weight}"
        )
        save_stage1_delta(
            delta_path, model, taskid,
            seed=args.seed, method=method_tag,
            prox_weight=float(args.prm_prox_weight or 0.0),
            train_loss_final=train_loss_final,
            q_t=_q_t,
            delta_norm_F=_delta_norm_F,
            delta_norm_per_layer=_delta_norm_per_layer,
            extra={
                "dataset": args.dataset,
                "num_tasks": args.num_tasks,
                "model": args.model,
                "eps": args.eps,
            },
        )
        if isinstance(per_task_meta, dict):
            per_task_meta[int(taskid)] = {
                "task_id": int(taskid),
                "q_t": float(_q_t) if _q_t is not None else None,
                "delta_norm_F": float(_delta_norm_F) if _delta_norm_F is not None else None,
                "delta_norm_per_layer": {
                    k: float(v) for k, v in _delta_norm_per_layer.items()
                },
                "alpha_chosen": None,
                "alpha_method": "stage1_only",
                "train_loss_final": (
                    float(train_loss_final) if train_loss_final is not None else None
                ),
            }
        return model

    if args.prm_method == "gated":
        if taskid == 0:
            alpha = 1.0
            print(f"  Write-back [PRM-gated/{args.gate_mode}]: alpha=1.0 (first task, no gating)")
        elif args.gate_alpha_base == "fixed":
            alpha = args.prm_alpha
            print(f"  Write-back [PRM-gated/{args.gate_mode}/fixed]: alpha={alpha}")
        elif args.gate_alpha_base == "fisher":
            alpha = compute_prm_fisher_alpha(model, fisher, taskid)
            print(f"  Write-back [PRM-gated/{args.gate_mode}/fisher]: alpha={alpha:.6f} (Fisher-derived)")
        elif args.gate_alpha_base == "budgeted":
            alpha = compute_prm_budgeted_alpha(
                model, taskid, args.drift_budget, base_alpha=args.prm_alpha,
            )
            print(f"  Write-back [PRM-gated/{args.gate_mode}/budgeted]: alpha={alpha:.6f}")
        else:
            raise ValueError(f"Unknown gate_alpha_base: {args.gate_alpha_base}")
        _alpha_chosen = float(alpha)
        _alpha_method = f"gated-{args.gate_mode}-{args.gate_alpha_base}"

        if args.gate_mode == "direction":
            apply_direction_gate(
                model, fisher, taskid,
                late_snapshots=late_snapshots if late_snapshots else None,
                a=args.gate_a, b=args.gate_b, c=args.gate_c,
                rescale_mean=(not args.gate_no_rescale),
            )
            with torch.no_grad():
                prm_write_back(model, alpha, taskid)
        else:
            gates = compute_writeability_gate(
                model, fisher, taskid,
                late_snapshots=late_snapshots if late_snapshots else None,
                a=args.gate_a, b=args.gate_b, c=args.gate_c,
                rescale_mean=(not args.gate_no_rescale),
            )
            with torch.no_grad():
                prm_write_back_gated(model, alpha, taskid, gates)
    else:
        sweep_requested = (
            taskid >= 1
            and (taskid + 1) in args.alpha_sweep_tasks
            and args.prm_method not in ("empirical", "2d")
        )

        if taskid == 0:
            alpha = 1.0
            merge_with_alpha(model, taskid, alpha)
            print(f"  Write-back: alpha=1.0 (first task)")
            _alpha_chosen = 1.0
            _alpha_method = f"{args.prm_method}-task0"
        elif args.prm_method == "2d":
            # Phase 4: decompose delta_t into (shared, residual); pick
            # (alpha_s, alpha_r) either fixed or via argmin J proxy.
            if args.d2_mode == "fixed":
                apply_2d_fixed(
                    model, taskid, late_snapshots,
                    args.d2_alpha_s, args.d2_alpha_r,
                )
            else:
                sweep_save_path = None
                if args.empirical_sweep_out_dir:
                    sweep_save_path = os.path.join(
                        args.empirical_sweep_out_dir,
                        f"prm_2d_task{taskid+1:02d}_seed{args.seed}.json",
                    )
                # Note: decomposition is also created internally.
                choose_empirical_alpha_2d(
                    model, args, cl_manager, pretrained_cfg, label_map,
                    param_cache, fisher, taskid, late_snapshots,
                    lambda_proxy=args.empirical_lambda,
                    alpha_grid=args.d2_alpha_grid,
                    save_path=sweep_save_path,
                    tag=f"prm-2d_seed{args.seed}",
                )
            _alpha_method = f"2d-{args.d2_mode}"
            # 2d uses two scalars (shared, residual); leave alpha_chosen None.
        elif args.prm_method == "empirical":
            # Phase 2: pick alpha via argmin J(alpha) = L_t^val + lam * a^2 * q_t.
            grid_n = max(2, args.empirical_grid_n)
            empirical_grid = [round(i / (grid_n - 1), 4) for i in range(grid_n)]
            sweep_save_path = None
            if args.empirical_sweep_out_dir:
                sweep_save_path = os.path.join(
                    args.empirical_sweep_out_dir,
                    f"prm_empirical_task{taskid+1:02d}_seed{args.seed}.json",
                )
            alpha = choose_empirical_alpha(
                model, args, cl_manager, pretrained_cfg, label_map,
                param_cache, fisher, taskid,
                lambda_proxy=args.empirical_lambda,
                alpha_grid=empirical_grid,
                save_path=sweep_save_path,
                tag=f"prm-empirical_seed{args.seed}",
            )
            print(f"  Write-back [PRM-empirical]: alpha={alpha:.6f} (lambda={args.empirical_lambda:g})")
            _alpha_chosen = float(alpha)
            _alpha_method = "empirical"
        elif sweep_requested:
            # Phase 1 diagnostic: dense sweep; finalize with method's own alpha.
            fisher_alpha_ref = compute_prm_fisher_alpha(model, fisher, taskid)
            if args.prm_method == "fixed":
                alpha = args.prm_alpha
            elif args.prm_method == "fisher":
                alpha = fisher_alpha_ref
            elif args.prm_method == "budgeted":
                alpha = compute_prm_budgeted_alpha(
                    model, taskid, args.drift_budget, base_alpha=args.prm_alpha,
                )
            else:
                raise ValueError(f"Unknown prm_method for sweep: {args.prm_method}")

            sweep_save_path = os.path.join(
                args.alpha_sweep_out_dir or "results",
                f"prm_{args.prm_method}_sweep_task{taskid+1:02d}_seed{args.seed}.json",
            )
            sweep_grid = _default_alpha_grid(args.alpha_grid_n)
            run_alpha_sweep(
                model, args, cl_manager, pretrained_cfg, label_map,
                param_cache, fisher, taskid,
                save_path=sweep_save_path,
                alpha_grid=sweep_grid,
                theory_alpha=fisher_alpha_ref,
                chosen_alpha=alpha,
                tag=f"prm-{args.prm_method}_seed{args.seed}",
                function_probe=args.function_probe,
                function_probe_samples=args.function_probe_samples,
            )
            print(f"  Write-back [PRM-{args.prm_method}]: alpha={alpha:.6f} (after sweep)")
            _alpha_chosen = float(alpha)
            _alpha_method = f"{args.prm_method}-sweep"
        else:
            if args.prm_method == "fixed":
                alpha = args.prm_alpha
                print(f"  Write-back [PRM-fixed]: alpha={alpha}")
            elif args.prm_method == "fisher":
                alpha = compute_prm_fisher_alpha(model, fisher, taskid)
                print(f"  Write-back [PRM-Fisher]: alpha={alpha:.6f} (Fisher-derived)")
            elif args.prm_method == "budgeted":
                alpha = compute_prm_budgeted_alpha(
                    model, taskid, args.drift_budget, base_alpha=args.prm_alpha,
                )
                print(f"  Write-back [PRM-budgeted]: alpha={alpha:.6f} (drift-constrained, cap={args.prm_alpha})")
            else:
                raise ValueError(f"Unknown prm_method: {args.prm_method}")

            with torch.no_grad():
                prm_write_back(model, alpha, taskid)
            _alpha_chosen = float(alpha)
            _alpha_method = args.prm_method

    # Stage 3: Consolidation (optional)
    if args.consolidation:
        print(f"  Stage 3: Consolidation ({args.consolidation_epochs} epochs)")
        consolidate(
            model, dataloader, criterion, args, taskid,
            consolidation_epochs=args.consolidation_epochs,
            consolidation_lr=args.consolidation_lr,
        )
        # Re-cache params: consolidation changes head + LoRA weights,
        # so the eval head must reflect post-consolidation state
        cache_task_params(model, taskid, param_cache)

    # Clean up text features buffer
    model.remove_text_features()

    # Set LoRA modules to inference mode
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.training = False

    if isinstance(per_task_meta, dict):
        per_task_meta[int(taskid)] = {
            "task_id": int(taskid),
            "q_t": float(_q_t) if _q_t is not None else None,
            "delta_norm_F": float(_delta_norm_F) if _delta_norm_F is not None else None,
            "delta_norm_per_layer": {k: float(v) for k, v in _delta_norm_per_layer.items()},
            "alpha_chosen": (float(_alpha_chosen) if _alpha_chosen is not None else None),
            "alpha_method": str(_alpha_method),
            "train_loss_final": (float(train_loss_final) if train_loss_final is not None else None),
        }

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, args, task_classes, pretrained_cfg, label_map):
    """Evaluate accuracy on a single task's classes."""
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

    total_correct = 0
    total_samples = 0
    model.eval()
    torch.cuda.empty_cache()

    for images, target in dataloader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            with torch.no_grad():
                logits = model(images)
        preds = logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)

    return total_correct / total_samples if total_samples > 0 else 0.0


def evaluate_all_tasks(model, args, cl_manager, pretrained_cfg, label_map, param_cache, acc_mat, acc_list, taskid):
    """Evaluate on all tasks seen so far after training task `taskid`."""
    model = modify_head(
        model, cl_manager.sofar_task_classes, param_cache,
        training=False, current_taskid=taskid,
    )
    set_model_mode(model, training=False)

    for eval_tid in range(taskid + 1):
        eval_classes = cl_manager.get_classes(eval_tid)
        acc = evaluate_task(model, args, eval_classes, pretrained_cfg, label_map)
        acc_mat[taskid, eval_tid] = acc
        print(f"  Eval task [{eval_tid+1}/{taskid+1}]: {acc:.2%}")

    avg_acc = np.mean(acc_mat[taskid, : taskid + 1])
    acc_list[taskid] = avg_acc

    model.remove_text_features()
    return avg_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PRM paper runs require a CUDA GPU. CPU-only utilities and tests "
            "remain available; run training on a CUDA host."
        )
    seed_everything(args.seed)
    method_tag = f"PRM-{args.prm_method}"
    print(f"[{method_tag}] dataset={args.dataset}, tasks={args.num_tasks}, "
          f"alpha={args.prm_alpha}, prox_weight={args.prm_prox_weight}, "
          f"rank={args.rank}, eps={args.eps}")

    # Initialize continual learning manager
    train_dataset = get_path_dataset(args.dataset, args.data_root, train=False)
    cl_manager = ClassIncrementalManager(
        train_dataset.class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes,
    )

    # Accuracy tracking
    nt = cl_manager.num_tasks
    acc_mat = np.zeros([nt, nt])
    acc_list = np.zeros([nt])

    # State
    label_map = {}
    param_cache = {}
    fisher = init_fisher_storage()
    model = None
    pretrained_cfg = None
    run_metadata = None
    start_time = ttime()
    per_task_meta = {}

    # Twin-train: resume from a saved prefix and train only --train_task.
    completed_taskid = -1  # last task whose Stage 1+2 are baked in via prefix
    target_taskid = -1     # 0-indexed task to train (set when resuming)
    if args.resume_prefix:
        assert args.train_task >= 1, "--resume_prefix requires --train_task >= 1"
        loaded = load_prefix(args.resume_prefix)
        model, pretrained_cfg, completed_taskid = rebuild_model_from_prefix(
            loaded, args, timm.create_model,
        )
        run_metadata = collect_run_metadata(args, cl_manager, pretrained_cfg)
        # Restore cached parameters; restore Fisher below only when needed.
        label_map = dict(loaded.get("label_map") or {})
        for k, v in (loaded.get("param_cache") or {}).items():
            if isinstance(v, dict):
                param_cache[k] = {
                    kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, torch.Tensor):
                param_cache[k] = v.cuda()
            else:
                param_cache[k] = v
        # Drop historical diagnostic tensors when resuming an ordinary fixed
        # run; only Fisher-dependent runs need to materialize them on the GPU.
        loaded_fisher = loaded.pop("fisher", None)
        if not needs_fisher(args):
            loaded_fisher = None
        for k, v in (loaded_fisher or {}).items():
            fisher[k] = {
                int(tk) if isinstance(tk, (int, str)) and str(tk).lstrip('-').isdigit() else tk:
                    (tv.cuda() if isinstance(tv, torch.Tensor) else tv)
                for tk, tv in (v.items() if isinstance(v, dict) else [])
            }
        del loaded_fisher
        target_taskid = args.train_task - 1
        assert target_taskid == completed_taskid + 1, (
            f"--train_task={args.train_task} (taskid={target_taskid}) must "
            f"equal completed_taskid+1={completed_taskid+1}"
        )
        restore_rng(loaded)
        print(f"[Resume] prefix completed_taskid={completed_taskid}; "
              f"will train only target taskid={target_taskid} "
              f"(1-indexed task {target_taskid+1})")

    for taskid, task_classes in cl_manager:
        # Resume mode: skip tasks already covered by the prefix.
        if args.resume_prefix:
            if taskid < target_taskid:
                continue
            if taskid > target_taskid:
                break

        # Save-prefix mode: dump state at the boundary BEFORE training task t.
        if (not args.resume_prefix
                and args.save_prefix_at_task > 0
                and (taskid + 1) == args.save_prefix_at_task):
            assert args.save_prefix_dir, (
                "--save_prefix_at_task requires --save_prefix_dir"
            )
            prefix_path = os.path.join(
                args.save_prefix_dir,
                f"prefix_t{args.save_prefix_at_task:02d}_seed{args.seed}.pt",
            )
            save_prefix(
                prefix_path, model, param_cache, fisher, label_map,
                pretrained_cfg, completed_taskid=taskid - 1, args=args,
                extra={"upstream_method": f"prm_{args.prm_method}"},
            )
            print(f"[Prefix] saved -> {prefix_path}; exiting before task "
                  f"{args.save_prefix_at_task}.")
            return

        # Create or reuse model
        if model is None and taskid == 0:
            other_args = {
                "logit_type": args.logit_type,
                "logit_scale": args.logit_scale,
                "logit_scale_trainable": False,
            }
            pretrained_overlay = local_checkpoint_overlay(args.pretrained_checkpoint)
            model = timm.create_model(
                args.model, pretrained=True, pretrained_strict=False,
                num_classes=512 if args.logit_type == "sim_imgtext" else len(task_classes),
                other_args_dict=other_args,
                pretrained_cfg_overlay=pretrained_overlay,
            )
            pretrained_cfg = deepcopy(model.pretrained_cfg)
            run_metadata = collect_run_metadata(args, cl_manager, pretrained_cfg)

        # Apply LoRA adapters
        apply_lora(model, taskid, rank=args.rank, eps=args.eps,
                   num_tasks=args.num_tasks, module=args.module)

        # Enable LoRA training mode
        for module in model.modules():
            if isinstance(module, LoRALayer):
                module.training = True

        # Train
        model = train_one_task(
            model, args, cl_manager, label_map, pretrained_cfg,
            param_cache, fisher, taskid, task_classes,
            per_task_meta=per_task_meta,
        )

        # Stage-1-only twin-train mode: delta+head already dumped inside
        # train_one_task; skip eval/Stage-2/Stage-3 and exit cleanly.
        if args.stage1_only_save_delta:
            print(f"[Stage1Only] dumped delta for task {taskid+1} -> "
                  f"{args.stage1_only_save_delta}; exiting.")
            return

        if args.save_landscape_ckpt_dir and (
                args.save_landscape_only_task <= 0
                or args.save_landscape_only_task == taskid + 1):
            ckpt_path = os.path.join(
                args.save_landscape_ckpt_dir,
                f"ckpt_task{taskid+1:02d}_seed{args.seed}.pt",
            )
            save_task_checkpoint(
                ckpt_path, model, param_cache, taskid,
                method=f"prm_{args.prm_method}", seed=args.seed,
            )

        # Evaluate
        avg_acc = evaluate_all_tasks(
            model, args, cl_manager, pretrained_cfg, label_map,
            param_cache, acc_mat, acc_list, taskid,
        )

        # Report metrics
        forgetting = calc_forgetting(acc_mat, taskid)
        aaa = calc_aaa(acc_list, taskid)
        bwt = calc_bwt(acc_mat, taskid)
        elapsed = ttime() - start_time

        print(f"\n  >> Task [{taskid+1}/{nt}] Results:")
        print(f"     Last Acc:   {avg_acc:.2%}")
        print(f"     Forgetting: {forgetting:.2%}")
        print(f"     AAA:        {aaa:.2%}")
        print(f"     BWT:        {bwt:+.2%}")
        print(f"     Time:       {elapsed:.0f}s")

        # Save results incrementally
        if args.results_file:
            results = {
                "method": f"prm_{args.prm_method}",
                "task": taskid + 1,
                "AAA": float(aaa),
                "last_acc": float(avg_acc),
                "forgetting": float(forgetting),
                "BWT": float(bwt),
                "prm_alpha": args.prm_alpha,
                "prm_method": args.prm_method,
                "prm_prox_weight": args.prm_prox_weight,
                "consolidation": args.consolidation,
                "model": args.model,
                "seed": args.seed,
                **(run_metadata or {}),
                "acc_matrix": acc_mat[:taskid + 1, :taskid + 1].tolist(),
                "average_accuracy_by_task": acc_list[:taskid + 1].tolist(),
                "elapsed_seconds": float(elapsed),
            }
            if args.prm_method == "gated":
                results.update({
                    "gate_mode": args.gate_mode,
                    "gate_alpha_base": args.gate_alpha_base,
                    "gate_a": args.gate_a,
                    "gate_b": args.gate_b,
                    "gate_c": args.gate_c,
                    "gate_late_k": args.gate_late_k,
                    "gate_rescale_mean": not args.gate_no_rescale,
                })
            write_json(args.results_file, results)

            # Per-task sidecar for flatness-mechanism analysis.
            meta = per_task_meta.get(taskid, {})
            task_entry = dict(meta)
            task_entry.update({
                "task_id": int(taskid),
                "val_acc": float(avg_acc),
                "aaa_so_far": float(aaa),
                "forgetting_so_far": float(forgetting),
                "acc_matrix": acc_mat[:taskid + 1, :taskid + 1].tolist(),
            })
            header = {
                "seed": int(args.seed),
                "dataset": str(args.dataset),
                "method": f"prm_{args.prm_method}",
                "lambda_prox": float(args.prm_prox_weight or 0.0),
                "perturbation_eps": float(args.eps or 0.0),
                "model": str(args.model),
                "num_tasks": int(args.num_tasks),
                **(run_metadata or {}),
            }
            append_per_task_summary(args.results_file, header, task_entry)

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL [{method_tag}]: AAA={calc_aaa(acc_list, nt-1):.2%}, "
          f"BWT={calc_bwt(acc_mat, nt-1):+.2%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
