"""
Fisher Information accumulation and optimal alpha computation.

Used by the P&M (Propose & Merge) method and PRM-Fisher to compute
curvature-aware merge coefficients. The Fisher diagonal approximates
the importance of each weight dimension for each task.

The optimal alpha formula minimizes expected loss increase across all tasks:
    alpha* = -sum(F_i * delta * (W_{t-1} - W_i)) / sum(F_i * delta^2)

where:
    F_i    = Fisher diagonal for task i
    delta  = B_t @ A_t  (current task's LoRA adapter)
    W_{t-1}= accumulated merged weight before current task
    W_i    = weight when task i was first learned
"""

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .lora import LoRALayer, QkvWithLoRA


def init_fisher_storage(num_layers=12):
    """Initialize Fisher storage dict for all transformer layers.

    Returns:
        dict: {layer_idx_str: {task_id: fisher_tensor}}
    """
    fisher = {}
    for i in range(num_layers):
        fisher[str(i)] = {}
    return fisher


def accumulate_fisher(
    model, dataloader, criterion, fisher, task_id, num_layers=12
):
    """Accumulate Fisher Information diagonal for the current task.

    Computes F_t = E[(d L / d W_qkv)^2] over the training data,
    where the expectation is approximated by averaging over all samples.

    The Fisher is stored per-layer in the `fisher` dict and is used
    later to compute the optimal merge alpha.

    Args:
        model:      The ViT model (set to eval mode before calling).
        dataloader:  Training data loader for the current task.
        criterion:   Loss function (CrossEntropyLoss).
        fisher:      Fisher storage dict from init_fisher_storage().
        task_id:     Current task index.
        num_layers:  Number of transformer layers (default 12).
    """
    # Enable grad only on the frozen QKV weights for Fisher computation
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if "qkv.qkv" in name:
            p.requires_grad_(True)
    model.eval()

    # Initialize Fisher accumulators
    for module in model.modules():
        if isinstance(module, QkvWithLoRA):
            name = f"{module.layer}"
            fisher[name][task_id] = torch.zeros_like(
                module.qkv.weight.data
            ).to(module.qkv.weight.device)

    num_samples = 0
    for images, target in dataloader:
        model.zero_grad()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()

        for module in model.modules():
            if isinstance(module, QkvWithLoRA):
                name = f"{module.layer}"
                grad = module.qkv.weight.grad.detach() ** 2
                fisher[name][task_id] += grad * images.size(0)
        num_samples += images.size(0)

    # Normalize by number of samples
    for layer_idx in range(num_layers):
        fisher[str(layer_idx)][task_id] /= num_samples


def compute_fisher_alpha(model, fisher, task_id, embed_dim=768):
    """Compute the Fisher-optimal global alpha for P&M write-back.

    Implements the closed-form optimal merge coefficient:
        alpha* = -sum_layers sum_tasks (F * delta * (W_{t-1} - W_accum - lora_i))
                 / sum_layers sum_tasks (F * delta^2)

    The Fisher slices for K and V projections are taken from the
    appropriate rows of the QKV weight matrix:
        K: rows [dim : 2*dim]
        V: rows [2*dim : 3*dim]

    Args:
        model:     The ViT model with LoRA adapters.
        fisher:    Fisher storage dict with accumulated diagonal.
        task_id:   Current task index.
        embed_dim: Model embedding dimension (default 768 for ViT-B).

    Returns:
        float: The globally optimal merge alpha.
    """
    ups = 0.0
    downs = 0.0

    for module in model.modules():
        if isinstance(module, LoRALayer):
            name = f"{module.layer}"
            delta = (
                module.lora_B[task_id].weight @ module.lora_A[task_id].weight
            )
            # Reconstruct W_{t-1}: accumulated weight before this task
            W_t_1 = 0.0
            for i in range(task_id):
                W_t_1 += (
                    module.alphas[i]
                    * module.lora_B[i].weight
                    @ module.lora_A[i].weight
                )

            weight = 0.0
            for i in range(task_id + 1):
                lora_i = module.lora_B[i].weight @ module.lora_A[i].weight

                if "k" in module.pos:
                    f_k = fisher[name][i][
                        embed_dim : embed_dim * 2, :
                    ].abs()
                    up = (W_t_1 - weight - lora_i) * f_k * delta
                    down = delta * f_k * delta
                    ups += up
                    downs += down

                if "v" in module.pos:
                    f_v = fisher[name][i][
                        embed_dim * 2 : embed_dim * 3, :
                    ].abs()
                    up = (W_t_1 - weight - lora_i) * f_v * delta
                    down = delta * f_v * delta
                    ups += up
                    downs += down

                if i < task_id:
                    weight += module.alphas[i] * lora_i

    alpha_tensor = (-torch.sum(ups) / torch.sum(downs)).detach()
    return float(alpha_tensor.cpu().item())
