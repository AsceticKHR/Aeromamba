"""
AeroMamba-VLA: Main model class.

A streamlined Mamba-based Vision-Language-Action model tailored for UAV
embodied navigation, referencing RoboMamba and UAV-OpenVLA design patterns.

Full pipeline:

  [FPV image]
      │ SigLIP-SO400M/14 @ 384px  →  [B, 729, 1152]  ─┐
      │ DINOv2-L/14     @ 384px   →  [B, 729, 1024]  ─┴─ cat → [B, 729, 2176]
      │                                                         │
      │                                               MLPProjector [B, 729, D_m]
      │                                                         │
  [Language]  →  Mamba embed  →  [B, L, D_m]  ──────┐         │
  [UAV state] →  ProprioEncoder → [B, 1, D_m]  ──────┤         │
                                                      └──cat────┘
                                                 [text | prop | vis]   (causal order)
                                                         │
                                                Mamba-2 backbone
                                                         │
                                                   h[:, -1, :]   (last token)
                                                         │
                                               UAVActionHead MLP
                                                         │
                                                  [B, K, 4]  (Δx, Δy, Δz, Δyaw_rad)

Token sequence order:  [text (L) | proprio (1) | vision (N_vis)]
  Rationale: Mamba causal model — language context first, then state,
  then vision.  Last token's hidden state aggregates all context.

Vision choices:
  Single  : 'siglip_l_384', 'siglip_so_384', 'dinov2_l', etc.
  Dual ★  : 'dinosiglip_so_384'  ← recommended for UAV navigation
             SigLIP semantic + DINOv2 geometric features, channel-wise fused.

Design references:
  UAV-OpenVLA (prismatic/models/backbones/vision/dinosiglip_vit.py)
  RoboMamba   (lmzpai/roboMamba)    — Mamba + MLP head + last-token pooling
  LLaVA-1.5   (Liu et al. 2023)    — MLP projector, no cross-attention
  OpenVLA-OFT (Kim et al. 2025)    — action chunking + direct L1 regression
  UAV-Flow                          — 4-DOF action space + temporal ensemble
"""

from __future__ import annotations

import io
import sys
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, MambaForCausalLM

# Reconfigure stdout/stderr to UTF-8 on Windows (default is CP1252 in PowerShell)
# This prevents UnicodeEncodeError when printing box-drawing / emoji characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .vision          import build_vision_encoder, DinoSigLIPEncoder
from .projector       import MLPProjector
from .proprio_encoder import ProprioEncoder
from .action_head     import UAVActionHead, UAVDynamicsActionHead, aero_action_loss
from .resampler       import PerceiverResampler


# ─────────────────────────────────────────────────────────────────────────────
# Mamba backbone registry
# ─────────────────────────────────────────────────────────────────────────────

MAMBA_PRESETS: dict[str, str] = {
    "mamba-130m":   "state-spaces/mamba-130m-hf",
    "mamba-370m":   "state-spaces/mamba-370m-hf",
    "mamba-790m":   "state-spaces/mamba-790m-hf",
    "mamba-1.4b":   "state-spaces/mamba-1.4b-hf",
    "mamba-2.8b":   "state-spaces/mamba-2.8b-hf",
    "mamba2-370m":  "state-spaces/mamba2-370m",
    "mamba-2-370m": "state-spaces/mamba2-370m",
    "mamba-zephyr": "xiuyul/mamba-2.8b-zephyr",
}

# Default LoRA target modules for Mamba's SSM projections
MAMBA_LORA_TARGETS = ["in_proj", "out_proj", "x_proj", "dt_proj"]


def _mamba_model_cls(config):
    if getattr(config, "model_type", "") == "mamba2":
        from transformers import Mamba2ForCausalLM
        return Mamba2ForCausalLM
    return MambaForCausalLM


def _load_pretrained_mamba(hub_name: str):
    if hub_name == "state-spaces/mamba2-370m":
        from transformers import Mamba2Config, Mamba2ForCausalLM
        config = Mamba2Config(
            hidden_size=1024,
            num_hidden_layers=48,
            vocab_size=50288,
            state_size=128,
            expand=2,
            num_heads=32,
            n_groups=1,
            head_dim=64,
            tie_word_embeddings=True,
        )
        return Mamba2ForCausalLM.from_pretrained(
            hub_name,
            config=config,
            trust_remote_code=True,
        )
    config = AutoConfig.from_pretrained(hub_name, trust_remote_code=True)
    model_cls = _mamba_model_cls(config)
    return model_cls.from_pretrained(hub_name, config=config, trust_remote_code=True)


def _init_random_mamba(mamba_type: str):
    if mamba_type in {"mamba2-370m", "mamba-2-370m"}:
        raise RuntimeError(
            "Mamba-2-370M requires pretrained config/weights from "
            "'state-spaces/mamba2-370m'. Disable AEROMAMBA_OFFLINE or cache the "
            "model before training."
        )
    from transformers import MambaConfig
    presets = {
        "mamba-130m": {"num_hidden_layers": 24, "hidden_size": 768},
        "mamba-370m": {"num_hidden_layers": 48, "hidden_size": 1024},
        "mamba-790m": {"num_hidden_layers": 48, "hidden_size": 1536},
    }
    cfg_args = presets.get(mamba_type, presets["mamba-370m"])
    return MambaForCausalLM(MambaConfig(**cfg_args))


# ─────────────────────────────────────────────────────────────────────────────
# Mock Tokenizer for offline mode
# ─────────────────────────────────────────────────────────────────────────────

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = 50277
    def __call__(self, text, **kwargs):
        words = text.split()
        input_ids = []
        for w in words:
            h = sum(ord(c) * (i + 1) for i, c in enumerate(w))
            token_id = (h % (self.vocab_size - 10)) + 5
            input_ids.append(token_id)
        return {"input_ids": input_ids}
    def decode(self, token_ids, **kwargs):
        return " ".join(f"token_{tid}" for tid in token_ids)

# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class AeroMambaVLA(nn.Module):
    """
    AeroMamba-VLA — UAV-tailored Mamba VLA model with DinoSigLIP vision.

    Args:
        mamba_type    : Mamba variant key (see MAMBA_PRESETS).
        vision_type   : Vision encoder key.
                        Single:  'siglip_l_384', 'siglip_so_384', 'dinov2_l', ...
                        Dual ★:  'dinosiglip_so_384'  (recommended)
        chunk_size    : Action chunk length K (default 5 waypoints).
        proprio_dim   : Proprioceptive state dimensionality (default 4).
        freeze_vision : Freeze vision backbone(s) on construction (default True).
    """

    def __init__(
        self,
        mamba_type:    str  = "mamba-370m",
        vision_type:   str  = "dinosiglip_so_384",
        chunk_size:    int  = 5,
        proprio_dim:   int  = 4,
        freeze_vision: bool = True,
        use_token_pooling: bool = False,
        pool_size:     int  = 8,
        token_resampler: str = "none",
        num_visual_queries: int = 32,
        resampler_layers: int = 2,
        resampler_heads: int = 8,
        action_head_type: str = "mlp",
        action_bound: float = 1.0,
    ):
        super().__init__()
        self.chunk_size   = chunk_size
        self.vision_type  = vision_type
        self.use_token_pooling = use_token_pooling
        self.pool_size = pool_size
        self.token_resampler_type = token_resampler
        self.num_visual_queries = num_visual_queries
        self.action_head_type = action_head_type
        self.is_dual_vision = isinstance(
            build_vision_encoder.__wrapped__ if hasattr(build_vision_encoder, "__wrapped__")
            else None, type(None)
        )  # resolved below after building encoder

        # ── 1. Vision Encoder ────────────────────────────────────────────────
        self.vision_encoder = build_vision_encoder(vision_type, freeze=freeze_vision)
        self.is_dual_vision = isinstance(self.vision_encoder, DinoSigLIPEncoder)
        D_v = self.vision_encoder.hidden_size  # 2176 for dinosiglip_so_384

        # ── 2. Mamba Backbone ────────────────────────────────────────────────
        if mamba_type not in MAMBA_PRESETS:
            raise ValueError(
                f"Unknown mamba_type '{mamba_type}'. "
                f"Available: {list(MAMBA_PRESETS)}"
            )
        import os
        offline = os.environ.get("AEROMAMBA_OFFLINE", "0") == "1"
        hub_name = MAMBA_PRESETS[mamba_type]
        
        if offline:
            print("\n[AeroMambaVLA] Running in OFFLINE mode. Initializing Mamba with random weights.")
            self.mamba = _init_random_mamba(mamba_type)
            self.tokenizer = MockTokenizer()
        else:
            try:
                self.mamba = _load_pretrained_mamba(hub_name)
                self.tokenizer = AutoTokenizer.from_pretrained(hub_name)
            except Exception as e:
                if mamba_type in {"mamba2-370m", "mamba-2-370m"}:
                    raise RuntimeError(
                        f"Failed to load pretrained Mamba-2 weights from {hub_name}: {e}"
                    ) from e
                print(f"\n[AeroMambaVLA] Warning: Failed to load pretrained Mamba weights ({e}). Initializing with random weights.")
                self.mamba = _init_random_mamba(mamba_type)
                
                # Use gpt2 tokenizer as a fallback, and if that fails, use MockTokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                except Exception as e2:
                    print(f"[AeroMambaVLA] Warning: Failed to load gpt2 tokenizer ({e2}). Using MockTokenizer.")
                    self.tokenizer = MockTokenizer()
        D_m            = self.mamba.config.hidden_size

        # NOTE: We do NOT replace lm_head with nn.Identity here.
        # Newer transformers versions access `self.lm_head.weight` for dtype casting
        # inside MambaForCausalLM.forward before calling lm_head, which would
        # raise AttributeError on nn.Identity.
        # Instead, _run_mamba calls self.mamba.backbone directly to get hidden
        # states without going through lm_head at all.

        # ── 3. MLP Projector  (LLaVA-1.5 / OpenVLA style) ───────────────────
        # Input dim automatically adapts: D_v=1024 (single) or 2176 (dual)
        self.projector = MLPProjector(
            vision_hidden_size=D_v,
            mamba_hidden_size=D_m,
        )

        if token_resampler == "none":
            self.token_resampler = nn.Identity()
        elif token_resampler == "perceiver":
            self.token_resampler = PerceiverResampler(
                hidden_size=D_m,
                num_queries=num_visual_queries,
                num_layers=resampler_layers,
                num_heads=resampler_heads,
            )
        else:
            raise ValueError(
                f"Unknown token_resampler '{token_resampler}'. "
                "Choose from: 'none', 'perceiver'."
            )

        # ── 4. Proprioception Encoder ────────────────────────────────────────
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=proprio_dim,
            mamba_hidden_size=D_m,
        )

        # ── 5. Action Head ───────────────────────────────────────────────────
        if action_head_type == "mlp":
            self.action_head = UAVActionHead(
                mamba_hidden_size=D_m,
                chunk_size=chunk_size,
            )
        elif action_head_type == "dynamics":
            self.action_head = UAVDynamicsActionHead(
                mamba_hidden_size=D_m,
                chunk_size=chunk_size,
                proprio_dim=proprio_dim,
                action_bound=action_bound,
            )
        else:
            raise ValueError(
                f"Unknown action_head_type '{action_head_type}'. "
                "Choose from: 'mlp', 'dynamics'."
            )

        # Store key dimensions
        self.D_m = D_m
        self.D_v = D_v

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs [B, L]  →  embeddings [B, L, D_m]."""
        return self.mamba.backbone.embeddings(input_ids)

    def _run_mamba(
        self,
        inputs_embeds: torch.Tensor,
        cache_params=None,
    ) -> torch.Tensor:
        """
        Forward through Mamba backbone to obtain hidden states [B, T, D_m].

        We call self.mamba.backbone directly (the MambaModel layer) rather than
        self.mamba (MambaForCausalLM) to avoid the lm_head entirely.  This is
        the correct approach for feature extraction:
          - MambaForCausalLM.forward internally accesses lm_head.weight for
            dtype casting; nn.Identity has no .weight, causing AttributeError
            in newer transformers versions.
          - MambaModel.forward returns a BaseModelOutputWithNoAttention whose
            .last_hidden_state is the raw hidden state [B, T, D_m] we need.
        """
        out = self.mamba.backbone(
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=(cache_params is not None),
        )
        # last_hidden_state: [B, T, D_m]
        return out.last_hidden_state

    def _encode_vision(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Run vision encoder (single or dual) under no_grad for efficiency.

        Both single and dual encoders are frozen during training by default.
        Using no_grad avoids storing the computation graph through the large
        ViT(s), saving significant GPU memory.

        Returns:
            vis_patches : [B, N_vis, D_v]
        """
        with torch.no_grad():
            vis_patches = self.vision_encoder(pixel_values)

        if self.use_token_pooling:
            import math
            import torch.nn.functional as F
            B, N_vis, D_v = vis_patches.shape
            grid_size = int(math.sqrt(N_vis))
            assert grid_size * grid_size == N_vis, f"Patch count {N_vis} is not a perfect square."
            # [B, N_vis, D_v] -> [B, grid_size, grid_size, D_v]
            x = vis_patches.view(B, grid_size, grid_size, D_v)
            # [B, grid_size, grid_size, D_v] -> [B, D_v, grid_size, grid_size]
            x = x.permute(0, 3, 1, 2)
            # [B, D_v, grid_size, grid_size] -> [B, D_v, pool_size, pool_size]
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
            # [B, D_v, pool_size, pool_size] -> [B, pool_size, pool_size, D_v]
            x = x.permute(0, 2, 3, 1)
            # [B, pool_size, pool_size, D_v] -> [B, pool_size * pool_size, D_v]
            vis_patches = x.flatten(1, 2)

        return vis_patches  # [B, N_vis, D_v]

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values:  Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_ids:     torch.Tensor,                    # [B, L]
        proprio:       Optional[torch.Tensor] = None,   # [B, 4]
        gt_action:     Optional[torch.Tensor] = None,   # [B, K, 4]  train only
        cache_params=None,
        return_loss:   bool  = False,
        lambda_smooth: float = 0.1,
    ) -> dict:
        """
        Multimodal forward pass.

        pixel_values : For single encoders — [B, 3, H, W].
                       For DinoSigLIP     — dict {"dino": [B,3,H,W],
                                                  "siglip": [B,3,H,W]}
                       (produced automatically by DinoSigLIPTransform)

        Token sequence: [text (L) | proprio (1) | vision (N_vis)]
        Global token  : hidden[:, -1, :]  — last vision token accumulates all.

        Returns dict with:
            'action'       : [B, K, 4]  predicted action chunk
            'loss'         : scalar     (only if return_loss=True & gt_action given)
            'loss_detail'  : dict       (only if return_loss=True & gt_action given)
        """
        B = (
            pixel_values["dino"].size(0)
            if isinstance(pixel_values, dict)
            else pixel_values.size(0)
        )

        # ── 1. Text embeddings ───────────────────────────────────────────────
        text_embs = self._embed_text(input_ids)           # [B, L, D_m]

        # ── 2. Proprioception token ──────────────────────────────────────────
        if proprio is None:
            proprio = torch.zeros(
                B, self.proprio_encoder.proprio_dim,
                device=input_ids.device,
                dtype=text_embs.dtype,
            )
        prop_token = self.proprio_encoder(proprio)         # [B, 1, D_m]

        # ── 3. Vision tokens ─────────────────────────────────────────────────
        vis_patches = self._encode_vision(pixel_values)   # [B, N_vis, D_v]
        vis_tokens  = self.projector(vis_patches)          # [B, N_vis, D_m]
        vis_tokens  = self.token_resampler(vis_tokens)     # [B, N_resampled, D_m]

        # ── 4. Concatenate: [text | proprio | vision] ────────────────────────
        # Causal order: language context → flight state → visual observation.
        # Mamba retains earlier tokens more strongly; last visual token
        # accumulates the full multimodal context.
        inputs_embeds = torch.cat(
            [text_embs, prop_token, vis_tokens], dim=1
        )  # [B, L + 1 + N_vis, D_m]

        # ── 5. Mamba backbone ─────────────────────────────────────────────────
        hidden = self._run_mamba(inputs_embeds, cache_params)
        # [B, L + 1 + N_vis, D_m]

        # ── 6. Global token: last hidden state (RoboMamba convention) ─────────
        global_token = hidden[:, -1, :]    # [B, D_m]

        # ── 7. Action chunking head ───────────────────────────────────────────
        pred = self.action_head(global_token, proprio=proprio)   # {"action": [B, K, 4]}

        # ── 8. Loss (training only) ───────────────────────────────────────────
        if return_loss and gt_action is not None:
            loss, detail = aero_action_loss(
                pred, gt_action, lambda_smooth=lambda_smooth
            )
            pred["loss"]        = loss
            pred["loss_detail"] = detail

        return pred

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helper
    # ─────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_step(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_ids:    torch.Tensor,
        proprio:      Optional[torch.Tensor] = None,
        cache_params=None,
    ) -> dict:
        """Single-step inference. Returns {"action": [B, K, 4]}."""
        self.eval()
        return self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            proprio=proprio,
            cache_params=cache_params,
            return_loss=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LoRA
    # ─────────────────────────────────────────────────────────────────────────

    def apply_lora(
        self,
        r:              int       = 16,
        lora_alpha:     int       = 32,
        lora_dropout:   float     = 0.05,
        target_modules: list[str] = None,
    ) -> "AeroMambaVLA":
        """Wrap Mamba backbone with LoRA adapters (requires `peft`)."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or MAMBA_LORA_TARGETS,
            bias="none",
            inference_mode=False,
        )
        self.mamba = get_peft_model(self.mamba, cfg)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Stage configuration
    # ─────────────────────────────────────────────────────────────────────────

    def configure_stage1(self) -> None:
        """
        Stage 1 — Projector alignment (InfoNCE).
        Trainable : MLPProjector only.
        Frozen    : VisionEncoder(s), Mamba, ProprioEncoder, ActionHead.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True

    def configure_stage2(self, lora_r: int = 16, lora_alpha: int = 32) -> None:
        """
        Stage 2 — VLM instruction fine-tuning.
        Trainable : MLPProjector + Mamba-LoRA adapters.
        Frozen    : VisionEncoder(s), ProprioEncoder, ActionHead.
        """
        self.configure_stage1()                         # freeze everything first
        self.apply_lora(r=lora_r, lora_alpha=lora_alpha)  # LoRA params auto-trainable

    def configure_stage3(self, train_lora: bool = False) -> None:
        """
        Stage 3 — Action head training.
        Trainable : UAVActionHead + ProprioEncoder + optional token resampler
                    + optional Mamba-LoRA adapters.
        Frozen    : VisionEncoder(s), base Mamba backbone, MLPProjector.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.action_head.parameters():
            p.requires_grad = True
        for p in self.proprio_encoder.parameters():
            p.requires_grad = True
        if not isinstance(self.token_resampler, nn.Identity):
            for p in self.token_resampler.parameters():
                p.requires_grad = True
        if train_lora:
            for name, p in self.mamba.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def print_trainable_params(self) -> None:
        """Pretty-print trainable vs total parameters per sub-module."""
        def _count(mod):
            total     = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return trainable, total

        modules: dict[str, nn.Module] = {
            "vision_encoder":  self.vision_encoder,
            "projector":       self.projector,
            "token_resampler": self.token_resampler,
            "proprio_encoder": self.proprio_encoder,
            "mamba":           self.mamba,
            "action_head":     self.action_head,
        }
        # Break down dual encoder sub-components
        if self.is_dual_vision:
            modules["  └─ dino_featurizer"]   = self.vision_encoder.dino_featurizer
            modules["  └─ siglip_featurizer"] = self.vision_encoder.siglip_featurizer

        SEP = "-" * 66
        print(f"\n{'Module':<26} {'Trainable':>14} {'Total':>14} {'%':>8}")
        print(SEP)
        total_t, total_n = 0, 0
        for name, mod in modules.items():
            t, n = _count(mod)
            if not name.startswith("  L-"):  # only count top-level in totals
                total_t += t; total_n += n
            pct = 100.0 * t / max(n, 1)
            print(f"{name:<26} {t:>14,} {n:>14,} {pct:>7.2f}%")
        print(SEP)
        pct = 100.0 * total_t / max(total_n, 1)
        print(f"{'TOTAL':<26} {total_t:>14,} {total_n:>14,} {pct:>7.2f}%\n")

    def vision_info(self) -> str:
        """Return a human-readable summary of the active vision encoder."""
        enc = self.vision_encoder
        if self.is_dual_vision:
            d_dino   = enc.dino_featurizer.embed_dim
            d_siglip = enc.siglip_featurizer.embed_dim
            return (
                f"DinoSigLIP ({enc.encoder_id})  "
                f"patches={enc.num_patches}  "
                f"D_dino={d_dino}  D_siglip={d_siglip}  "
                f"D_fused={enc.hidden_size}"
            )
        return (
            f"Single ({enc.encoder_type})  "
            f"patches={enc.num_patches}  D_v={enc.hidden_size}"
        )
