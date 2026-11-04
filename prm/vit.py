# SPDX-License-Identifier: Apache-2.0
# Copyright 2020, Ross Wightman
# Modified for PRM; see THIRD_PARTY_NOTICES.md and licenses/timm-APACHE-2.0.txt.
"""
Vision Transformer (ViT) with perturbation-aware forward pass.

Trimmed from timm 0.9.12 VisionTransformer to include only the components
needed for PRM: Attention, Block, and VisionTransformer with LoRA support.

The key modification is that the forward pass accepts a `perturbation` argument
that propagates through all attention blocks to the QkvWithLoRA layer,
enabling perturbation-aware training.

Based on:
  'An Image Is Worth 16x16 Words' - https://arxiv.org/abs/2010.11929
  timm 0.9.12 by Ross Wightman

Upstream source:
  https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/models/vision_transformer.py
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Upstream acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Sequence, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    DropPath,
    AttentionPoolLatent,
    PatchDropout,
    trunc_normal_,
    lecun_normal_,
    resample_abs_pos_embed,
    resample_patch_embed,
    use_fused_attn,
    get_act_layer,
    get_norm_layer,
    LayerType,
)
from timm.models import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models import generate_default_cfgs, register_model

__all__ = ["VisionTransformer"]


class Attention(nn.Module):
    """Multi-head self-attention with perturbation support.

    When perturbation != 0, the qkv call passes perturbation through to
    QkvWithLoRA, which applies the perturbation-aware weight formula.
    """
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, perturbation=0) -> Tensor:
        B, N, C = x.shape
        if perturbation != 0:
            qkv = self.qkv(x, perturbation=perturbation)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Manual attention (no fused_attn to allow perturbation propagation)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """Transformer block with perturbation-aware attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, perturbation=0) -> Tensor:
        x_a = self.attn(self.norm1(x), perturbation=perturbation)
        x = x + self.drop_path1(self.ls1(x_a))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with CLIP-compatible logit output.

    Supports perturbation-aware forward pass for PRM training.
    Uses text feature similarity (sim_imgtext) or linear head (head_out) for logits.
    """
    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        other_args_dict: dict = None,
        **kwargs,
    ):
        super().__init__()
        if other_args_dict is None:
            other_args_dict = {
                "logit_type": "sim_imgtext",
                "logit_scale": 4.605170249938965,
                "logit_scale_trainable": False,
            }

        assert global_pool in ("", "avg", "token", "map")
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches
            if no_embed_class
            else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.patch_drop = (
            PatchDropout(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
            if patch_drop_rate > 0
            else nn.Identity()
        )
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Logit scaling (CLIP-style)
        _logit_scale = other_args_dict.get("logit_scale", 4.605170249938965)
        if other_args_dict.get("logit_scale_trainable", False):
            self.logit_scale = nn.Parameter(torch.tensor(_logit_scale))
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(_logit_scale), persistent=False
            )

        self.logit_type = other_args_dict.get("logit_type", "sim_imgtext")

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(_get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    def _pos_embed(self, x: Tensor) -> Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=(
                    0 if self.no_embed_class else self.num_prefix_tokens
                ),
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: Tensor, perturbation=0) -> Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if perturbation:
            for block in self.blocks:
                x = block(x, perturbation=perturbation)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # CLS token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def cache_text_features(self, text_features: Tensor):
        _p = next(self.parameters())
        self.register_buffer(
            "text_features",
            text_features.to(device=_p.device, dtype=_p.dtype),
            persistent=False,
        )

    def remove_text_features(self):
        self.register_buffer("text_features", None, persistent=False)

    def encode_image(self, x, pre_logits=False, perturbation=0):
        x = self.forward_features(x, perturbation=perturbation)
        x = self.forward_head(x, pre_logits)
        return x

    def forward_logits(self, image_features: Tensor):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits = (
            image_features
            @ self.get_buffer("text_features").t()
            * self.logit_scale.exp()
        )
        return logits

    def forward(self, x: Tensor, perturbation=0) -> Tensor:
        x = self.encode_image(x, perturbation=perturbation)
        if self.logit_type == "head_out":
            pass
        elif self.logit_type == "sim_imgtext":
            x = self.forward_logits(x)
        else:
            raise ValueError(f"Unknown logit_type: {self.logit_type}")
        return x


# ---------------------------------------------------------------------------
# Weight initialization helpers (from timm)
# ---------------------------------------------------------------------------

def _init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def _init_weights_vit_jax(module: nn.Module, name: str = "", head_bias: float = 0.0):
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if "mlp" in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def _get_init_weights_vit(mode: str = "jax", head_bias: float = 0.0):
    if "jax" in mode:
        return partial(_init_weights_vit_jax, head_bias=head_bias)
    return _init_weights_vit_timm


# ---------------------------------------------------------------------------
# Checkpoint conversion helpers (from timm, needed for pretrained weight loading)
# ---------------------------------------------------------------------------

def _convert_openai_clip(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    prefix: str = "visual.",
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ("conv1", "patch_embed.proj"),
        ("positional_embedding", "pos_embed"),
        ("transformer.resblocks.", "blocks."),
        ("ln_pre", "norm_pre"),
        ("ln_post", "norm"),
        ("ln_", "norm"),
        ("in_proj_", "qkv."),
        ("out_proj", "proj"),
        ("mlp.c_fc", "mlp.fc1"),
        ("mlp.c_proj", "mlp.fc2"),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, "")
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k == "proj":
            k = "head.weight"
            v = v.transpose(0, 1)
            out_dict["head.bias"] = torch.zeros(v.shape[0])
        elif k == "class_embedding":
            k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == "pos_embed":
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                v = resample_abs_pos_embed(
                    v,
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=(
                        0
                        if getattr(model, "no_embed_class", False)
                        else getattr(model, "num_prefix_tokens", 1)
                    ),
                )
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    adapt_layer_scale: bool = False,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert checkpoint state_dict to match our custom VisionTransformer."""
    import re

    out_dict = {}
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = ""

    if "visual.class_embedding" in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif "module.visual.class_embedding" in state_dict:
        return _convert_openai_clip(state_dict, model, prefix="module.visual.")

    if "encoder" in state_dict:
        state_dict = state_dict["encoder"]
        prefix = "module."

    if "visual.trunk.pos_embed" in state_dict:
        prefix = "visual.trunk."

    if prefix:
        state_dict = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v, (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and "gamma_" in k:
            k = re.sub(r"gamma_([0-9])", r"ls\1.gamma", k)
        elif "pre_logits" in k:
            continue
        out_dict[k] = v
    return out_dict


# ---------------------------------------------------------------------------
# Model configuration and registration with timm
# ---------------------------------------------------------------------------

def _cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": _cfg(hf_hub_id="timm/"),
    "vit_base_patch16_clip_224.openai": _cfg(
        hf_hub_id="timm/vit_base_patch16_clip_224.openai",
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512,
    ),
    "vit_base_patch32_clip_224.openai": _cfg(
        hf_hub_id="timm/vit_base_patch32_clip_224.openai",
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512,
    ),
    "vit_large_patch14_clip_224.openai": _cfg(
        hf_hub_id="timm/vit_large_patch14_clip_224.openai",
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768,
    ),
}

# QuickGELU variants share the same pretrained configs
_quick_gelu_cfgs = [
    "vit_base_patch32_clip_224.openai",
    "vit_base_patch16_clip_224.openai",
    "vit_large_patch14_clip_224.openai",
]
default_cfgs.update({
    n.replace("_clip_", "_clip_quickgelu_"): default_cfgs[n]
    for n in _quick_gelu_cfgs
})
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(
    variant: str, pretrained: bool = False, **kwargs
) -> VisionTransformer:
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    _filter_fn = checkpoint_filter_fn
    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


def _register_prm_override(fn):
    """Register an intentional timm factory override without hiding other warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=rf"Overwriting {fn.__name__} in registry.*",
            category=UserWarning,
        )
        return register_model(fn)


@_register_prm_override
def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    )
    return _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs),
    )


@_register_prm_override
def vit_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm,
    )
    return _create_vision_transformer(
        "vit_base_patch16_clip_224", pretrained=pretrained, **dict(model_args, **kwargs),
    )


@_register_prm_override
def vit_base_patch32_clip_quickgelu_224(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer="quick_gelu",
    )
    return _create_vision_transformer(
        "vit_base_patch32_clip_quickgelu_224", pretrained=pretrained,
        **dict(model_args, **kwargs),
    )


@_register_prm_override
def vit_base_patch16_clip_quickgelu_224(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer="quick_gelu",
    )
    return _create_vision_transformer(
        "vit_base_patch16_clip_quickgelu_224", pretrained=pretrained,
        **dict(model_args, **kwargs),
    )


@_register_prm_override
def vit_large_patch14_clip_quickgelu_224(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer="quick_gelu",
    )
    return _create_vision_transformer(
        "vit_large_patch14_clip_quickgelu_224", pretrained=pretrained,
        **dict(model_args, **kwargs),
    )
