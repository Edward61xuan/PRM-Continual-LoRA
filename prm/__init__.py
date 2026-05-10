"""
PRM: Proximity-Regularized Merging for rehearsal-free continual learning.

This package implements the PRM algorithm which decomposes continual learning
into three stages per task:
  1. Fast Proposal: Train a proximity-regularized fast branch
  2. Write-Back: Merge fast branch into shared model
  3. Consolidation: (Optional) Further stabilize with conservative re-training
"""

from .lora import LoRALayer, QkvWithLoRA, apply_lora
from .prm_ops import (
    proximity_loss,
    get_lora_params_snapshot,
    compute_prm_theory_alpha,
    compute_prm_budgeted_alpha,
    consolidate,
)
from .fisher import accumulate_fisher, compute_fisher_alpha
from .metrics import calc_aaa, calc_bwt, calc_fwt, calc_forgetting

# Import vit module to register custom VisionTransformer with timm's model registry.
# The @register_model decorators fire at import time, overriding timm's built-in
# ViT entries with our perturbation-aware + CLIP logit-aware VisionTransformer.
from . import vit  # noqa: F401

__all__ = [
    "LoRALayer",
    "QkvWithLoRA",
    "apply_lora",
    "proximity_loss",
    "get_lora_params_snapshot",
    "compute_prm_theory_alpha",
    "compute_prm_budgeted_alpha",
    "consolidate",
    "accumulate_fisher",
    "compute_fisher_alpha",
    "calc_aaa",
    "calc_bwt",
    "calc_fwt",
    "calc_forgetting",
]
