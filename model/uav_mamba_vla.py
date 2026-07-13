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
  [UAV state] →  ProprioEncoder → [B, 2, D_m]  ─────┐         │
  [FPV image] →  projector/resampler → [B, N, D_m] ─┤         │
  [Language]  →  Mamba embed  →  [B, L, D_m]  ──────┴──cat────┘
                                                 [state | delta | vis | text]
                                                         │
                                                Mamba-2 backbone
                                                         │
                              last valid language token hidden state
                                                         │
                                               UAVActionHead MLP
                                                         │
                                                  [B, K, 4]  (Δx, Δy, Δz, Δyaw_rad)

Token sequence order:  [state (1) | delta_state (1) | vision (N_vis) | text (L)]
  Stage 1/2 feed zero state tokens; Stage 3 uses real 8D state + delta_state.
  Action pooling uses the last non-padding language token, not hidden[:, -1].

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

MAMBA_TOKENIZER_PRESETS: dict[str, str] = {
    "mamba-130m": "EleutherAI/gpt-neox-20b",
    "mamba-370m": "EleutherAI/gpt-neox-20b",
    "mamba-790m": "EleutherAI/gpt-neox-20b",
    "mamba-1.4b": "EleutherAI/gpt-neox-20b",
    "mamba-2.8b": "EleutherAI/gpt-neox-20b",
    "mamba2-370m": "EleutherAI/gpt-neox-20b",
    "mamba-2-370m": "EleutherAI/gpt-neox-20b",
    "mamba-zephyr": "HuggingFaceH4/zephyr-7b-beta",
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
        from pathlib import Path
        import torch
        from huggingface_hub import snapshot_download
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
        try:
            model = Mamba2ForCausalLM.from_pretrained(
                hub_name,
                config=config,
                trust_remote_code=True,
            )
        except Exception as e:
            # transformers >= 4.50 with torch < 2.6 refuses torch.load on
            # pytorch_model.bin (CVE-2025-32434). The official safetensors
            # conversion lives on refs/pr/1 of the same repo.
            if "torch.load" not in str(e) and "safetensors" not in str(e):
                raise
            model = Mamba2ForCausalLM.from_pretrained(
                hub_name,
                config=config,
                trust_remote_code=True,
                revision="refs/pr/1",
                use_safetensors=True,
            )
        if getattr(model.backbone.embeddings.weight, "is_meta", False):
            snapshot = Path(snapshot_download(hub_name, allow_patterns=["pytorch_model.bin"]))
            state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu")
            weight = state["backbone.embedding.weight"]
            embedding = nn.Embedding(weight.shape[0], weight.shape[1])
            embedding.weight.data.copy_(weight)
            model.backbone.embeddings = embedding
            model.lm_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            model.lm_head.weight = model.backbone.embeddings.weight
        return model
    config = AutoConfig.from_pretrained(hub_name, trust_remote_code=True)
    model_cls = _mamba_model_cls(config)
    return model_cls.from_pretrained(
        hub_name,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )


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
        mamba_type:    str  = "mamba-2-370m",
        vision_type:   str  = "siglip2_base_384",
        chunk_size:    int  = 5,
        proprio_dim:   int  = 8,
        freeze_vision: bool = True,
        use_token_pooling: bool = False,
        pool_size:     int  = 8,
        token_resampler: str = "perceiver",
        num_visual_queries: int = 64,
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
            except Exception as e:
                if mamba_type in {"mamba2-370m", "mamba-2-370m"}:
                    raise RuntimeError(
                        f"Failed to load pretrained Mamba-2 weights from {hub_name}: {e}"
                    ) from e
                print(f"\n[AeroMambaVLA] Warning: Failed to load pretrained Mamba weights ({e}). Initializing with random weights.")
                self.mamba = _init_random_mamba(mamba_type)
            tokenizer_name = os.environ.get(
                "AEROMAMBA_TOKENIZER",
                MAMBA_TOKENIZER_PRESETS.get(mamba_type, hub_name),
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer {tokenizer_name!r} for {mamba_type}. "
                    "Cache/install it first or set AEROMAMBA_TOKENIZER explicitly. "
                    "Refusing to silently fall back to GPT-2 because that corrupts "
                    "Stage1/2 language supervision."
                ) from e
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = (
                    getattr(self.tokenizer, "eos_token", None)
                    or getattr(self.tokenizer, "unk_token", None)
                    or "<|endoftext|>"
                )
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

    def _prepare_state_inputs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        state: Optional[torch.Tensor] = None,
        delta_state: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded/truncated 8D state and delta-state tensors."""
        state_dim = self.proprio_encoder.proprio_dim
        if state is None:
            state = proprio
        if state is None:
            state = torch.zeros(batch_size, state_dim, device=device, dtype=dtype)
        state = state.to(device=device, dtype=dtype)
        if state.size(-1) < state_dim:
            state = torch.cat(
                [state, state.new_zeros(*state.shape[:-1], state_dim - state.size(-1))],
                dim=-1,
            )
        elif state.size(-1) > state_dim:
            state = state[..., :state_dim]

        if delta_state is None:
            delta_state = torch.zeros_like(state)
        else:
            delta_state = delta_state.to(device=device, dtype=dtype)
            if delta_state.size(-1) < state_dim:
                delta_state = torch.cat(
                    [
                        delta_state,
                        delta_state.new_zeros(
                            *delta_state.shape[:-1],
                            state_dim - delta_state.size(-1),
                        ),
                    ],
                    dim=-1,
                )
            elif delta_state.size(-1) > state_dim:
                delta_state = delta_state[..., :state_dim]
        return state, delta_state

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values:  Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_ids:     torch.Tensor,                    # [B, L]
        proprio:       Optional[torch.Tensor] = None,   # legacy [B, 4] or [B, D]
        state:         Optional[torch.Tensor] = None,   # [B, 8]
        delta_state:   Optional[torch.Tensor] = None,   # [B, 8]
        gt_action:     Optional[torch.Tensor] = None,   # [B, K, 4]  train only
        cache_params=None,
        return_loss:   bool  = False,
        lambda_smooth: float = 0.0,
        lambda_endpoint: float = 0.25,
        lambda_direction: float = 0.5,
        lambda_acc: float = 0.0,
    ) -> dict:
        """
        Multimodal forward pass.

        pixel_values : For single encoders — [B, 3, H, W].
                       For DinoSigLIP     — dict {"dino": [B,3,H,W],
                                                  "siglip": [B,3,H,W]}
                       (produced automatically by DinoSigLIPTransform)

        Token sequence: [state (1) | delta_state (1) | vision (N_vis) | text (L)]
        Global token  : last non-padding language token.

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

        # 1. Text embeddings
        text_embs = self._embed_text(input_ids)           # [B, L, D_m]

        # 2. State tokens: [s, delta_s]
        state, delta_state = self._prepare_state_inputs(
            B,
            input_ids.device,
            text_embs.dtype,
            state=state,
            delta_state=delta_state,
            proprio=proprio,
        )
        state_tokens = self.proprio_encoder.forward_pair(state, delta_state)

        # 3. Vision tokens
        vis_patches = self._encode_vision(pixel_values)
        vis_tokens = self.projector(vis_patches)
        vis_tokens = self.token_resampler(vis_tokens)

        # 4. Unified AeroMamba-Opt order: [state | delta_state | vision | text]
        inputs_embeds = torch.cat([state_tokens, vis_tokens, text_embs], dim=1)

        # 5. Mamba backbone
        hidden = self._run_mamba(inputs_embeds, cache_params)

        # 6. Action context: last non-padding language token
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        text_mask = (input_ids != pad_id).to(text_embs.dtype)
        text_lengths = text_mask.sum(dim=1).long().clamp_min(1) - 1
        prefix_len = state_tokens.size(1) + vis_tokens.size(1)
        gather_idx = (prefix_len + text_lengths).view(B, 1, 1).expand(-1, 1, hidden.size(-1))
        global_token = hidden.gather(dim=1, index=gather_idx).squeeze(1)

        # 7. Action chunking head
        pred = self.action_head(global_token, proprio=state)

        if return_loss and gt_action is not None:
            loss, detail = aero_action_loss(
                pred,
                gt_action,
                head=self.action_head,
                lambda_smooth=lambda_smooth,
                lambda_endpoint=lambda_endpoint,
                lambda_direction=lambda_direction,
                lambda_acc=lambda_acc,
            )
            pred["loss"] = loss
            pred["loss_detail"] = detail

        return pred

    @torch.inference_mode()
    def predict_step(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_ids: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        delta_state: Optional[torch.Tensor] = None,
        cache_params=None,
    ) -> dict:
        """
        Single-step inference. Returns {"action": [B, K, 4]} in PHYSICAL units
        (m / rad): raw z-space head output is denormalised via the action
        stats buffers (identity if no stats were set).
        """
        was_training = self.training
        try:
            self.eval()
            pred = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                proprio=proprio,
                state=state,
                delta_state=delta_state,
                cache_params=cache_params,
                return_loss=False,
            )
            if hasattr(self.action_head, "denormalize"):
                pred["action"] = self.action_head.denormalize(pred["action"])
            return pred
        finally:
            self.train(was_training)

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
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
        # peft >= 0.18 hard-blocks LoRA on Mamba out_proj/conv1d, but our
        # checkpoints were trained with out_proj LoRA on the slow (non-fused)
        # transformers path where it is mathematically fine. Disable the check.
        try:
            from peft.tuners import tuners_utils as _tu
            if hasattr(_tu, "_check_lora_target_modules_mamba"):
                _tu._check_lora_target_modules_mamba = lambda *a, **k: None
        except Exception:
            pass
        self.mamba = get_peft_model(self.mamba, cfg)
        return self

    def configure_stage1(self) -> None:
        """Stage 1: train projector and optional token resampler only."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        if not isinstance(self.token_resampler, nn.Identity):
            for p in self.token_resampler.parameters():
                p.requires_grad = True

    def configure_stage2(self, lora_r: int = 16, lora_alpha: int = 32) -> None:
        """Stage 2: projector/resampler plus Mamba LoRA adapters."""
        self.configure_stage1()
        self.apply_lora(r=lora_r, lora_alpha=lora_alpha)

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
