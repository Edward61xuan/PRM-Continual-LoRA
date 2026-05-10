"""
LoRA (Low-Rank Adaptation) for continual learning with per-task adapters.

This module implements:
  - LoRALayer: Per-task LoRA adapter with perturbation-aware weight formula.
  - QkvWithLoRA: Wraps a frozen QKV projection with LoRA adapters on K/V.
  - apply_lora: Applies LoRA adapters to all transformer blocks in a model.

Key design:
  Each task t gets its own (A_t, B_t) pair. After training, the merged weight is:
    W = sum_{i<t} alpha_i * (B_i @ A_i)  +  (1 + perturbation * eps) * B_t @ A_t

  where perturbation in {0, +1, -1} is sampled stochastically during training,
  and alpha_i is the write-back coefficient computed after task i.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Per-task LoRA adapter with sequential proposal and merging support.

    Maintains a list of (A_i, B_i) adapter pairs and alpha_i merge coefficients
    across tasks. During training on task t, only (A_t, B_t) are active.

    Args:
        in_dim:  Input dimension of the adapted projection.
        out_dim: Output dimension of the adapted projection.
        rank:    LoRA rank (default 10).
        eps:     Perturbation magnitude for perturbation-aware training.
        num_tasks: Maximum number of tasks.
        pos:     Position identifier ('k' or 'v').
        layer:   Transformer block index (0..11).
    """

    def __init__(self, in_dim, out_dim, rank, eps, num_tasks, pos="", layer=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_tasks = num_tasks
        self.eps = eps
        self.pos = pos
        self.layer = layer

        # Per-task LoRA matrices: task 0 initialized here, others added via update()
        self.lora_A = nn.ModuleList([nn.Linear(in_dim, rank, bias=False)])
        self.lora_B = nn.ModuleList([nn.Linear(rank, out_dim, bias=False)])
        self.alphas = nn.ParameterList([])
        # Per-task writeability gates for GatedPRM. Each entry is either a
        # (out_dim, in_dim) tensor (element-wise gate applied to B_i @ A_i)
        # or None (no gating, equivalent to a gate of ones).
        self.gates = []
        # Phase 4: per-task 2D shared/residual decomposition. Keyed by task id,
        # only populated when --prm_method 2d is active for that task.
        #   _d2_shared[i]: tensor (out_dim, in_dim) representing delta_shared
        #   _d2_res[i]:    tensor (out_dim, in_dim) representing delta_residual
        #   _d2_alpha_s[i], _d2_alpha_r[i]: scalar alphas for shared / residual.
        self._d2_shared = {}
        self._d2_res = {}
        self._d2_alpha_s = {}
        self._d2_alpha_r = {}

        nn.init.kaiming_uniform_(self.lora_A[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[0].weight)

        self.task_id = 0

    def update(self, task_id):
        """Add a new (A, B) adapter pair for a new task."""
        if (self.task_id + 1) == len(self.alphas):
            self.lora_A.append(nn.Linear(self.in_dim, self.rank, bias=False))
            nn.init.kaiming_uniform_(self.lora_A[task_id].weight, a=math.sqrt(5))
            self.lora_B.append(nn.Linear(self.rank, self.out_dim, bias=False))
            nn.init.zeros_(self.lora_B[task_id].weight)
            self.task_id = task_id

    def cal_alpha(self, alpha=1):
        """Record the write-back alpha for the current task."""
        self.alphas.append(
            nn.Parameter(torch.tensor(alpha), requires_grad=False)
        )
        # Keep gates list aligned with alphas: append None for ungated tasks.
        while len(self.gates) < len(self.alphas):
            self.gates.append(None)

    def cal_alpha_gated(self, alpha, gate):
        """Record alpha and a per-element gate tensor for the current task.

        The gate is applied element-wise to (B_t @ A_t) at weight assembly time:
            W = sum_i alpha_i * gate_i * (B_i @ A_i)

        Args:
            alpha: Scalar merge coefficient (same semantics as cal_alpha).
            gate:  (out_dim, in_dim) tensor with values typically in (0, 1)
                   or near 1 (rescaled). None disables gating for this task.
        """
        self.cal_alpha(alpha)
        if gate is not None:
            self.gates[self.task_id] = gate.detach()

    def get_weight(self, perturbation=0):
        """Compute the effective LoRA weight across all tasks.

        The perturbation-aware formula for the current (un-merged) task is:
            (1 + perturbation * eps) * B_t @ A_t

        For already-merged tasks:
            sum_{i<t} alpha_i * B_i @ A_i                  (1D merge)
            sum_{i<t} alpha_s_i * delta_s_i + alpha_r_i * delta_r_i   (2D merge, if set)

        Args:
            perturbation: One of {0, +1, -1}. Non-zero only during training.

        Returns:
            Tensor of shape (out_dim, in_dim): the effective LoRA delta weight.
        """
        weight = 0.0
        for i in range(self.task_id):
            if i in self._d2_shared:
                # Phase 4 2D merge for this past task
                weight = weight + (
                    self._d2_alpha_s[i] * self._d2_shared[i]
                    + self._d2_alpha_r[i] * self._d2_res[i]
                )
            else:
                base = self.lora_B[i].weight @ self.lora_A[i].weight
                gate_i = self.gates[i] if i < len(self.gates) and self.gates[i] is not None else 1.0
                weight = weight + self.alphas[i] * gate_i * base

        if (self.task_id + 1) != len(self.alphas):
            # Training phase: current task not yet merged
            weight = weight + (1 + perturbation * self.eps) * (
                self.lora_B[self.task_id].weight @ self.lora_A[self.task_id].weight
            )
        elif self.task_id in self._d2_shared:
            # Inference / post-merge with 2D components
            weight = weight + (
                self._d2_alpha_s[self.task_id] * self._d2_shared[self.task_id]
                + self._d2_alpha_r[self.task_id] * self._d2_res[self.task_id]
            )
        else:
            # Inference / post-merge: current task already has alpha (and optional gate)
            base_t = self.lora_B[self.task_id].weight @ self.lora_A[self.task_id].weight
            gate_t = (
                self.gates[self.task_id]
                if self.task_id < len(self.gates) and self.gates[self.task_id] is not None
                else 1.0
            )
            weight = weight + self.alphas[self.task_id] * gate_t * base_t

        return weight

    def forward(self, x, perturbation=0):
        """Apply LoRA as a linear transform: result = x @ W^T."""
        return F.linear(x, self.get_weight(perturbation=perturbation))


class QkvWithLoRA(nn.Module):
    """Wraps a frozen QKV linear layer with per-task LoRA on K and/or V projections.

    The original qkv linear produces [Q, K, V] concatenated. This module adds
    LoRA deltas to the K and V portions of the output.

    Args:
        qkv:       The original nn.Linear(dim, 3*dim) layer.
        rank:      LoRA rank.
        task_id:   Initial task id (0).
        eps:       Perturbation magnitude.
        num_tasks: Maximum number of tasks.
        layer:     Transformer block index.
        module:    Which projections to adapt ('kv', 'qkv', etc.).
    """

    def __init__(self, qkv, rank, task_id, eps, num_tasks, layer, module="kv"):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.layer = layer
        self.module = module
        self.num_tasks = num_tasks

        if "q" in module:
            self.lora_q = LoRALayer(
                self.dim, self.dim, rank, eps, num_tasks, pos="q", layer=layer
            )
        if "k" in module:
            self.lora_k = LoRALayer(
                self.dim, self.dim, rank, eps, num_tasks, pos="k", layer=layer
            )
        if "v" in module:
            self.lora_v = LoRALayer(
                self.dim, self.dim, rank, eps, num_tasks, pos="v", layer=layer
            )

    def update(self, task_id):
        """Add new LoRA adapters for a new task."""
        if "q" in self.module:
            self.lora_q.update(task_id)
        if "k" in self.module:
            self.lora_k.update(task_id)
        if "v" in self.module:
            self.lora_v.update(task_id)

    def forward(self, x, perturbation=0):
        """Forward pass: frozen QKV + LoRA deltas on selected projections."""
        qkv = self.qkv(x)
        if "q" in self.module:
            qkv[:, :, : self.dim] += self.lora_q(x, perturbation=perturbation)
        if "k" in self.module:
            qkv[:, :, self.dim : 2 * self.dim] += self.lora_k(
                x, perturbation=perturbation
            )
        if "v" in self.module:
            qkv[:, :, -self.dim :] += self.lora_v(x, perturbation=perturbation)
        return qkv


def apply_lora(model, task_id, rank=10, eps=0.5, num_tasks=10, module="kv"):
    """Apply LoRA adapters to all transformer blocks in the model.

    On task 0, wraps each block's attn.qkv with QkvWithLoRA.
    On subsequent tasks, adds new (A, B) pairs via update().

    Args:
        model:     VisionTransformer model.
        task_id:   Current task index.
        rank:      LoRA rank.
        eps:       Perturbation magnitude.
        num_tasks: Total number of tasks.
        module:    Projection targets (default 'kv' for K and V only).
    """
    if task_id == 0:
        assign_lora = partial(
            QkvWithLoRA,
            rank=rank,
            task_id=task_id,
            eps=eps,
            num_tasks=num_tasks,
            module=module,
        )
        for i, block in enumerate(model.blocks):
            block.attn.qkv = assign_lora(
                block.attn.qkv, task_id=task_id, layer=i, module=module
            )
    else:
        for block in model.blocks:
            block.attn.qkv.update(task_id=task_id)
