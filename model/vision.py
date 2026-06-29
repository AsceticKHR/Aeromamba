"""
Vision Encoder for AeroMamba-VLA.

Supports two modes:
  1. Single encoder   — SigLIP-L-384 or DINOv2-L alone.
  2. DinoSigLIP fusion — DINOv2-L + SigLIP-SO400M, channel-wise patch concat.
     Mirrors UAV-OpenVLA `prismatic/models/backbones/vision/dinosiglip_vit.py`.

DinoSigLIP fusion design (UAV-OpenVLA reference):
  ┌─ SigLIP-SO400M/14 @ 384px  →  [B, N, D_sig=1152]  ─┐
  │                                                      ├─ cat dim=-1 → [B, N, D_sig+D_dino]
  └─ DINOv2-L/14 @ 384px       →  [B, N, D_dino=1024] ─┘

  Both featurizers use penultimate-block features (n={-2}),
  same resolution ⇒ same N = (384 // 14)² = 27² = 729 patches.

Why this combination for UAV navigation:
  - SigLIP   : semantic / language-aligned features (task goal understanding)
  - DINOv2   : geometric / spatial features (obstacle depth, scene layout)
  The two representations are complementary for embodied 3D navigation.

Architecture reference:
  UAV-OpenVLA  (UAV-Openvla/prismatic/models/backbones/vision/dinosiglip_vit.py)
  OpenVLA-OFT  (Kim et al. 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize


# ─────────────────────────────────────────────────────────────────────────────
# Encoder registries
# ─────────────────────────────────────────────────────────────────────────────

# Single-encoder presets
SINGLE_ENCODERS: dict[str, Union[str, tuple[str, ...]]] = {
    "siglip2_so_384": (
        "vit_so400m_patch16_siglip2_384",
        "vit_so400m_patch16_siglip2_384.webli",
        "vit_so400m_patch16_siglip2_384.google",
        "vit_so400m_patch14_siglip2_384",
    ),
    "siglip_l_384":   "vit_large_patch16_siglip_384",
    "siglip_so_384":  "vit_so400m_patch14_siglip_384",
    "siglip_b_224":   "vit_base_patch16_siglip_224",
    "clip_l_336":     "vit_large_patch14_clip_336.openai",
    "dinov2_l":       "vit_large_patch14_dinov2.lvd142m",
    "dinov2_l_reg":   "vit_large_patch14_reg4_dinov2.lvd142m",
}

SINGLE_IMG_SIZES: dict[str, int] = {
    "siglip2_so_384": 384,
    "siglip_l_384":   384,
    "siglip_so_384":  384,
    "siglip_b_224":   224,
    "clip_l_336":     336,
    "dinov2_l":       518,
    "dinov2_l_reg":   518,
}

# DinoSigLIP dual-encoder presets  — (dino_timm, siglip_timm, shared_img_size)
DINOSIGLIP_ENCODERS: dict[str, tuple] = {
    # Matches UAV-OpenVLA "dinosiglip-vit-so-384px" exactly
    "dinosiglip_so_384": (
        "vit_large_patch14_reg4_dinov2.lvd142m",  # DINOv2-L w/ registers
        "vit_so400m_patch14_siglip_384",           # SigLIP-SO400M/14
        384,
    ),
    # Lighter 224px variant for experiments / debug
    "dinosiglip_so_224": (
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_so400m_patch14_siglip_224",
        224,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_tuple(fn):
    """Unwrap single-element tuples/lists returned by get_intermediate_layers.

    Newer versions of timm (>= 0.9.x) return a list instead of a tuple,
    so we guard against both sequence types here.
    """
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        # Unpack single-element list or tuple → bare tensor
        if isinstance(result, (tuple, list)):
            return result[0]
        return result
    return wrapper


def _resolve_timm_model_name(encoder_type: str) -> str:
    names = SINGLE_ENCODERS[encoder_type]
    if isinstance(names, str):
        return names
    available = set(timm.list_models())
    for name in names:
        if name in available:
            return name
    return names[0]


def _build_transform(backbone, img_size: int) -> Compose:
    """Build a torchvision Compose transform for a timm backbone, fixing the
    over-large default resize that SigLIP and IN1K models apply."""
    data_cfg = timm.data.resolve_model_data_config(backbone)
    data_cfg["input_size"] = (3, img_size, img_size)
    tfm = timm.data.create_transform(**data_cfg, is_training=False)
    assert isinstance(tfm, Compose)
    # Fix: SigLIP default resize → larger than img_size (crops), override.
    if isinstance(tfm.transforms[0], Resize):
        tfm = Compose(
            [Resize(img_size, interpolation=tfm.transforms[0].interpolation),
             *tfm.transforms[1:]]
        )
    return tfm


# ─────────────────────────────────────────────────────────────────────────────
# Single VisionEncoder
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Single vision encoder wrapping a timm ViT.

    Returns penultimate-block patch features [B, N_patches, hidden_size].

    Args:
        encoder_type : Key from SINGLE_ENCODERS (default 'siglip_l_384').
        freeze       : Freeze backbone weights on construction.
    """

    def __init__(self, encoder_type: str = "siglip_l_384", freeze: bool = True):
        super().__init__()
        assert encoder_type in SINGLE_ENCODERS, (
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: {list(SINGLE_ENCODERS)} or use DinoSigLIPEncoder."
        )
        self.encoder_type = encoder_type
        self.img_size     = SINGLE_IMG_SIZES[encoder_type]
        self.timm_model_name = _resolve_timm_model_name(encoder_type)

        extra = {}
        if encoder_type.startswith("clip"):
            extra["act_layer"] = "quick_gelu"

        import os
        offline = os.environ.get("AEROMAMBA_OFFLINE", "0") == "1"
        pretrained = not offline
        
        if offline:
            print("\n[VisionEncoder] Running in OFFLINE mode. Initializing vision backbone with random weights.")
            self.backbone = timm.create_model(
                self.timm_model_name,
                pretrained=False,
                num_classes=0,
                img_size=self.img_size,
                **extra,
            )
        else:
            try:
                self.backbone = timm.create_model(
                    self.timm_model_name,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.img_size,
                    **extra,
                )
            except Exception as e:
                print(f"\n[VisionEncoder] Warning: Failed to load pretrained vision weights ({e}). Initializing with random weights.")
                self.backbone = timm.create_model(
                    self.timm_model_name,
                    pretrained=False,
                    num_classes=0,
                    img_size=self.img_size,
                    **extra,
                )
        self.backbone.eval()

        # Monkey-patch forward → penultimate-layer intermediate features
        # (UAV-OpenVLA / RoboMamba convention)
        self.backbone.forward = _unpack_tuple(
            partial(
                self.backbone.get_intermediate_layers,
                n={len(self.backbone.blocks) - 2},
            )
        )

        self.transform  = _build_transform(self.backbone, self.img_size)
        self.hidden_size: int = self.backbone.embed_dim
        self.num_patches: int = self.backbone.patch_embed.num_patches

        if freeze:
            self.freeze()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values : [B, 3, H, W]
        Returns:
            patch_features : [B, num_patches, hidden_size]
        """
        out = self.backbone(pixel_values)
        # Defensive unpack: guard against any list/tuple leakage from timm
        # (newer timm versions change the return type of get_intermediate_layers)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# DinoSigLIP dual-encoder  (UAV-OpenVLA style)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DinoSigLIPTransform:
    """
    Dual transform: accepts a PIL Image and returns a dict of two tensors.

    Usage in DataLoader:
        item = transform(pil_img)    →  {"dino": Tensor, "siglip": Tensor}
    """
    dino_transform:   Compose
    siglip_transform: Compose
    is_dual: bool = True

    def __call__(self, img) -> Dict[str, torch.Tensor]:
        return {
            "dino":   self.dino_transform(img),
            "siglip": self.siglip_transform(img),
        }


class DinoSigLIPEncoder(nn.Module):
    """
    Dual vision encoder: DINOv2 + SigLIP channel-wise patch concatenation.

    Strictly mirrors UAV-OpenVLA `DinoSigLIPViTBackbone`:
      - Both featurizers use penultimate-block intermediate features
      - Must share the same patch grid (same image size + same patch stride)
      - Channel-wise concat: [B, N, D_dino + D_siglip]

    Args:
        encoder_id : Key from DINOSIGLIP_ENCODERS (default 'dinosiglip_so_384').
        freeze     : Freeze both backbones on construction.
    """

    def __init__(
        self,
        encoder_id: str  = "dinosiglip_so_384",
        freeze:     bool = True,
    ):
        super().__init__()
        assert encoder_id in DINOSIGLIP_ENCODERS, (
            f"Unknown encoder_id '{encoder_id}'. "
            f"Choose from: {list(DINOSIGLIP_ENCODERS)}"
        )
        dino_id, siglip_id, img_size = DINOSIGLIP_ENCODERS[encoder_id]
        self.encoder_id = encoder_id
        self.img_size   = img_size

        # ── DINOv2 featurizer ────────────────────────────────────────────────
        self.dino_featurizer = timm.create_model(
            dino_id, pretrained=True, num_classes=0, img_size=img_size
        )
        self.dino_featurizer.eval()
        self.dino_featurizer.forward = _unpack_tuple(
            partial(
                self.dino_featurizer.get_intermediate_layers,
                n={len(self.dino_featurizer.blocks) - 2},
            )
        )

        # ── SigLIP featurizer ────────────────────────────────────────────────
        self.siglip_featurizer = timm.create_model(
            siglip_id, pretrained=True, num_classes=0, img_size=img_size
        )
        self.siglip_featurizer.eval()
        self.siglip_featurizer.forward = _unpack_tuple(
            partial(
                self.siglip_featurizer.get_intermediate_layers,
                n={len(self.siglip_featurizer.blocks) - 2},
            )
        )

        # Validate same patch grid  (UAV-OpenVLA assertion)
        assert (
            self.dino_featurizer.patch_embed.num_patches
            == self.siglip_featurizer.patch_embed.num_patches
        ), (
            f"DINOv2 ({self.dino_featurizer.patch_embed.num_patches}) and "
            f"SigLIP ({self.siglip_featurizer.patch_embed.num_patches}) must "
            f"have the same patch count."
        )

        # ── Transforms ──────────────────────────────────────────────────────
        dino_tfm   = _build_transform(self.dino_featurizer,   img_size)
        siglip_tfm = _build_transform(self.siglip_featurizer, img_size)
        self.transform = DinoSigLIPTransform(dino_tfm, siglip_tfm)

        # ── Expose dimensions ────────────────────────────────────────────────
        self.hidden_size: int = (
            self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim
        )
        self.num_patches: int = self.dino_featurizer.patch_embed.num_patches

        if freeze:
            self.freeze()

    def forward(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Accept either:
          - dict {"dino": [B,3,H,W], "siglip": [B,3,H,W]}   (standard)
          - plain tensor [B,3,H,W]  (applied to both, for quick testing)

        Returns:
            fused_patches : [B, N_patches, D_dino + D_siglip]
        """
        if isinstance(pixel_values, dict):
            dino_px   = pixel_values["dino"]
            siglip_px = pixel_values["siglip"]
        else:
            # Convenience: same tensor to both (pre-condition: img size matches both)
            dino_px = siglip_px = pixel_values

        dino_patches   = self.dino_featurizer(dino_px)     # [B, N, D_dino]
        siglip_patches = self.siglip_featurizer(siglip_px) # [B, N, D_siglip]

        return torch.cat([dino_patches, siglip_patches], dim=2)  # [B, N, D_dino+D_siglip]

    def freeze(self):
        for p in self.dino_featurizer.parameters():
            p.requires_grad = False
        for p in self.siglip_featurizer.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.dino_featurizer.parameters():
            p.requires_grad = True
        for p in self.siglip_featurizer.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Factory: unified encoder builder
# ─────────────────────────────────────────────────────────────────────────────

def build_vision_encoder(
    encoder_type: str = "siglip_l_384",
    freeze:       bool = True,
) -> Union[VisionEncoder, DinoSigLIPEncoder]:
    """
    Build the appropriate vision encoder from a string key.

    Single encoders  : 'siglip_l_384', 'siglip_so_384', 'dinov2_l', etc.
    Dual encoders    : 'dinosiglip_so_384', 'dinosiglip_so_224'

    Returns a module with attributes:
        .hidden_size  (int)  — output feature dim per patch
        .num_patches  (int)  — number of patch tokens
        .transform         — torchvision Compose (single) or DinoSigLIPTransform (dual)
        .forward(...)      — [B, N, D]
    """
    if encoder_type in DINOSIGLIP_ENCODERS:
        return DinoSigLIPEncoder(encoder_id=encoder_type, freeze=freeze)
    elif encoder_type in SINGLE_ENCODERS:
        return VisionEncoder(encoder_type=encoder_type, freeze=freeze)
    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Single: {list(SINGLE_ENCODERS)}. "
            f"Dual: {list(DINOSIGLIP_ENCODERS)}."
        )
