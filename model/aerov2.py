"""AeroMamba v2 model skeleton (design doc: docs/AEROV2_S3_V5_REDESIGN_20260727.md).

S1-scope implementation:
    SigLIP2-base-384 (frozen)  ->  MLP projector  ->  Falcon-H1 embedding space
    CLM forward over [vision | text] with caption-only labels.

Key v2 decisions already baked in:
  - NO token resampler / compression: all vision patch tokens go to the LM
    (smoke campaign E0-E3 showed the 64-query resampler starved the policy
    of visual information).
  - Backbone loaded via AutoModelForCausalLM (Falcon-H1 hybrid attn+Mamba-2);
    any causal LM exposing `get_input_embeddings` + `inputs_embeds` works,
    which keeps an offline tiny-Llama structural-test path.

Action heads / meta-query readouts / streaming come in later stages and will
extend this class rather than reviving v1's AeroMambaVLA.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from model.vision import HFVisionEncoder

DEFAULT_BACKBONE = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"


class MLPProjector(nn.Module):
    """LLaVA-1.5 style 2-layer GELU projector."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_backbone(model_id: str, dtype: torch.dtype):
    """Load the causal-LM backbone; offline mode builds a tiny random LM so
    wiring can be tested without weights or Falcon-H1 support."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.environ.get("AEROMAMBA_OFFLINE", "0") == "1":
        from transformers import LlamaConfig, LlamaForCausalLM

        print("[AeroV2] OFFLINE mode: tiny random Llama stands in for Falcon-H1.")
        cfg = LlamaConfig(hidden_size=256, intermediate_size=512,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4, vocab_size=32000)
        lm = LlamaForCausalLM(cfg).to(dtype)
        from model.uav_mamba_vla import MockTokenizer

        tok = MockTokenizer()
        tok.vocab_size = cfg.vocab_size
        return lm, tok

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return lm, tok


class AeroV2(nn.Module):
    def __init__(
        self,
        backbone_id: str = DEFAULT_BACKBONE,
        vision_type: str = "siglip2_base_384",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if os.environ.get("AEROMAMBA_OFFLINE", "0") == "1":
            # timm VisionEncoder supports random-init offline; HF path needs
            # cached weights. Structure tests only.
            from model.vision import VisionEncoder

            self.vision_encoder = VisionEncoder("siglip_l_384", freeze=True)
        elif vision_type.startswith("cradio"):
            from model.vision import RADIOVisionEncoder
            self.vision_encoder = RADIOVisionEncoder(vision_type, freeze=True)
        else:
            self.vision_encoder = HFVisionEncoder(vision_type, freeze=True)
        self.lm, self.tokenizer = _load_backbone(backbone_id, dtype)
        self.d_lm = self.lm.get_input_embeddings().embedding_dim
        self.projector = MLPProjector(self.vision_encoder.hidden_size, self.d_lm)
        # projector always runs/keeps grads in fp32-master style via autocast;
        # cast its params to the backbone dtype for memory symmetry
        self.projector.to(dtype)
        self.dtype = dtype
        # S2 grounding head: text-conditioned cross-attention over the vision
        # hidden states (DETR-style decoupled readout). The previous design
        # appended generic query tokens at the *end* of a causal Mamba sequence
        # and read their terminal hidden state — spatial detail cannot survive
        # the compressed SSM state, so grd_h was near-constant across samples
        # and the loss froze at its init value. Here the query attends directly
        # to per-patch vision features, giving a real spatial gradient path.
        self.n_grd_queries = 4
        self.grd_queries = nn.Parameter(
            torch.randn(self.n_grd_queries, self.d_lm, dtype=dtype) * 0.02)
        # LayerNorms are essential: deep Mamba hidden states have large
        # magnitude, so an un-normalised readout saturates the box sigmoid at
        # init and the gradient dies on step 1. Normalise every input to the
        # grounding path (vision keys/values, query, head input).
        self.grd_ln_kv = nn.LayerNorm(self.d_lm).to(dtype)
        self.grd_ln_q = nn.LayerNorm(self.d_lm).to(dtype)
        self.grd_ln_out = nn.LayerNorm(self.d_lm).to(dtype)
        self.grd_txt_proj = nn.Linear(self.d_lm, self.d_lm).to(dtype)
        self.grd_attn = nn.MultiheadAttention(
            self.d_lm, num_heads=8, batch_first=True, dtype=dtype)
        grd_out = nn.Linear(self.d_lm // 2, 4)
        # Head output = [centre_dx, centre_dy, size_w, size_h] modulating a
        # soft-argmax box (see predict_boxes). Small weight init; centre bias 0 but
        # SIZE bias negative so the box starts SMALL (sigmoid(-1.4)≈0.20 → half-
        # extent ≈0.10 → 0.2-wide box). Starting small + an over-coverage penalty
        # forces the head to *grow into* each GT rather than hedge with a giant box
        # (the size-collapse degenerate seen after the KV fix).
        nn.init.normal_(grd_out.weight, std=0.02)
        with torch.no_grad():
            grd_out.bias.zero_()
            grd_out.bias[2:4].fill_(-1.4)
        self.grd_logit_scale = 0.5
        self.grd_head = nn.Sequential(
            nn.Linear(self.d_lm, self.d_lm // 2),
            nn.GELU(),
            grd_out,
        ).to(dtype)

    # ── stage configuration ──────────────────────────────────────────────
    def configure_stage1(self, grad_ckpt: bool = True) -> None:
        """S1: train projector only."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        if grad_ckpt:
            self._enable_grad_ckpt()

    def configure_stage0(
        self,
        lora_r: int = 64,
        lora_alpha: int = 128,
        vision_unfreeze_last_n: int = 4,
        grad_ckpt: bool = True,
    ) -> None:
        """S0 aerial CPT: projector + LoRA-LM + top vision blocks.

        Full-parameter Falcon-H1 + vision exceeds a comfortable 24GB recipe;
        high-rank LoRA (r=64) is the CosFly-style CPT compromise for 4090.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True

        # unfreeze last N transformer blocks of the HF vision tower
        self._unfreeze_vision_tail(vision_unfreeze_last_n)

        from peft import LoraConfig, get_peft_model, TaskType

        # Falcon-H1 hybrid linear names (probed on 1.5B-Deep-Instruct).
        # peft 0.19 bans out_proj/conv1d on Mamba-family model_types.
        targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj",
        ]
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
        )
        self.lm = get_peft_model(self.lm, cfg)
        print(f"[AeroV2] S0 LoRA r={lora_r} alpha={lora_alpha} "
              f"vision_tail={vision_unfreeze_last_n}")
        if grad_ckpt:
            self._enable_grad_ckpt()

    def configure_stage2(
        self,
        vision_unfreeze_last_n: int = 4,
        grad_ckpt: bool = True,
    ) -> None:
        """S2: keep S0 LoRA, train projector + vision tail + grounding head.

        Assumes `self.lm` is already a PeftModel loaded from S0.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        self.grd_queries.requires_grad = True
        for mod in (self.grd_ln_kv, self.grd_ln_q, self.grd_ln_out,
                    self.grd_txt_proj, self.grd_attn, self.grd_head):
            for p in mod.parameters():
                p.requires_grad = True
        for n, p in self.lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        self._unfreeze_vision_tail(vision_unfreeze_last_n)
        if grad_ckpt:
            self._enable_grad_ckpt()
        print("[AeroV2] S2: LoRA + projector + vision_tail + xattn grounding head")

    def _enable_grad_ckpt(self) -> None:
        # gradients flow THROUGH the frozen deep LM back to the projector;
        # without checkpointing the 66-layer Falcon-H1-Deep activations
        # blow past 24GB. use_reentrant=False is required because the
        # checkpointed params themselves have requires_grad=False.
        try:
            base = self.lm
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            elif hasattr(base, "base_model") and hasattr(
                    base.base_model, "gradient_checkpointing_enable"):
                base.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[AeroV2] LM gradient checkpointing enabled")
        except Exception as e:
            print(f"[AeroV2] gradient checkpointing unavailable: {e}")

    def _unfreeze_vision_tail(self, last_n: int) -> None:
        if last_n <= 0:
            return
        bb = getattr(self.vision_encoder, "backbone", None)
        if bb is None:
            print("[AeroV2] no HF vision backbone; vision stays frozen")
            return
        # SigLIP2: vision_model.encoder.layers
        layers = None
        for path in (
            ("vision_model", "encoder", "layers"),
            ("encoder", "layers"),
            ("blocks",),
        ):
            cur = bb
            ok = True
            for attr in path:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok:
                layers = cur
                break
        if layers is None:
            print("[AeroV2] could not locate vision layers; vision stays frozen")
            return
        n = len(layers)
        for layer in layers[max(0, n - last_n):]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"[AeroV2] unfroze vision layers [{max(0, n - last_n)}:{n}] / {n}")

    def _lm_config(self):
        cfg = getattr(self.lm, "config", None)
        if cfg is not None and hasattr(cfg, "embedding_multiplier"):
            return cfg
        base = getattr(self.lm, "base_model", None)
        if base is not None:
            # peft: base_model.model.config or base_model.config
            for obj in (base, getattr(base, "model", None)):
                c = getattr(obj, "config", None)
                if c is not None:
                    return c
        return getattr(self.lm, "config", None)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def load_projector(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.projector.load_state_dict(ckpt["projector"], strict=True)
        print(f"[AeroV2] loaded projector from {ckpt_path} "
              f"(step={ckpt.get('step')} val={ckpt.get('val')})")

    def print_param_census(self) -> None:
        mods = (("vision", self.vision_encoder),
                ("projector", self.projector),
                ("grd_head", self.grd_head),
                ("lm", self.lm))
        for name, mod in mods:
            tot = sum(p.numel() for p in mod.parameters())
            tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"[AeroV2] {name:<10} total={tot/1e6:9.1f}M trainable={tr/1e6:9.1f}M")
        gq = self.grd_queries.numel()
        print(f"[AeroV2] {'grd_query':<10} total={gq/1e6:9.4f}M "
              f"trainable={(gq if self.grd_queries.requires_grad else 0)/1e6:9.4f}M")

    def _get_trunk(self):
        base_lm = self.lm
        if hasattr(base_lm, "get_base_model"):
            try:
                base_lm = base_lm.get_base_model()
            except Exception:
                bm = getattr(base_lm, "base_model", None)
                base_lm = getattr(bm, "model", None) or bm or base_lm
        # Prefer *.model (FalconH1Model) over *ForCausalLM
        trunk = getattr(base_lm, "model", None)
        if trunk is not None and hasattr(trunk, "embed_tokens"):
            return trunk
        if trunk is not None:
            return trunk
        return base_lm

    def _trunk_hidden(self, inputs_embeds, attn):
        trunk = self._get_trunk()
        out = trunk(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        # Mistakenly hit CausalLM wrapper — dig one level deeper.
        if hasattr(trunk, "model"):
            out = trunk.model(inputs_embeds=inputs_embeds, attention_mask=attn,
                              use_cache=False)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
        raise RuntimeError(
            f"trunk {type(trunk)} returned {type(out)} without last_hidden_state")

    def _encode_multimodal(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        append_grd: bool = False,
    ):
        vision_trainable = any(p.requires_grad for p in self.vision_encoder.parameters())
        if vision_trainable:
            vis = self.vision_encoder(pixel_values)
        else:
            with torch.no_grad():
                vis = self.vision_encoder(pixel_values)
        vis_tok = self.projector(vis.to(self.dtype))

        emb = self.lm.get_input_embeddings()
        cfg = self._lm_config()
        emb_mult = float(getattr(cfg, "embedding_multiplier", 1.0) or 1.0)
        head_mult = float(getattr(cfg, "lm_head_multiplier", 1.0) or 1.0)
        with torch.no_grad():
            txt_tok = emb(input_ids) * emb_mult

        parts = [vis_tok, txt_tok.to(vis_tok.dtype)]
        n_vis = vis_tok.size(1)
        if append_grd:
            q = self.grd_queries.to(vis_tok.dtype).unsqueeze(0).expand(
                vis_tok.size(0), -1, -1)
            parts.append(q)
        inputs_embeds = torch.cat(parts, dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                          device=inputs_embeds.device)
        return {
            "inputs_embeds": inputs_embeds,
            "attn": attn,
            "n_vis": n_vis,
            "head_mult": head_mult,
            "L_txt": input_ids.size(1),
        }

    # ── S1 forward: CLM over [vision | text] ─────────────────────────────
    def forward_clm(
        self,
        pixel_values: torch.Tensor,   # [B, 3, H, W]
        input_ids: torch.Tensor,      # [B, L]
        labels: Optional[torch.Tensor] = None,  # [B, L], -100 masked
    ) -> dict:
        enc = self._encode_multimodal(pixel_values, input_ids, append_grd=False)
        n_vis = enc["n_vis"]
        if labels is None:
            out = self.lm(inputs_embeds=enc["inputs_embeds"],
                          attention_mask=enc["attn"], use_cache=False)
            return {"loss": None, "logits": out.logits, "n_vis_tokens": n_vis}

        trunk = self._get_trunk()
        hidden = self._trunk_hidden(enc["inputs_embeds"], enc["attn"])
        L_txt = enc["L_txt"]
        pred_h = hidden[:, n_vis - 1: n_vis - 1 + L_txt, :]
        logits = self.lm.get_output_embeddings()(pred_h) * enc["head_mult"]
        loss = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits, "n_vis_tokens": n_vis}

    def _grid_pos_embed(self, n_vis, device, dtype):
        """Fixed 2D sinusoidal positional embedding for the vision patch grid.

        Without an explicit spatial basis on the cross-attention keys/values the
        pooled attention output cannot encode *where* the object is, so the box
        head collapses to a constant prior box (its init bias). Adding a 2D PE to
        the vision K/V gives the head a coordinate basis to decode locations.
        Parameter-free (not saved) so it does not change checkpoint state_dict.
        """
        cache = getattr(self, "_pe_cache", None)
        if cache is None:
            cache = {}
            self._pe_cache = cache
        key = (int(n_vis), str(device), str(dtype))
        if key in cache:
            return cache[key]
        D = self.d_lm
        g = int(round(float(n_vis) ** 0.5))
        yy, xx = torch.meshgrid(torch.arange(g, dtype=torch.float32),
                                torch.arange(g, dtype=torch.float32),
                                indexing="ij")
        xs = xx.reshape(-1)
        ys = yy.reshape(-1)
        d4 = D // 4
        div = torch.exp(torch.arange(d4, dtype=torch.float32)
                        * -(torch.log(torch.tensor(10000.0)) / max(d4, 1)))
        pe = torch.zeros(g * g, D, dtype=torch.float32)
        pe[:, 0:d4] = torch.sin(xs[:, None] * div)
        pe[:, d4:2 * d4] = torch.cos(xs[:, None] * div)
        pe[:, 2 * d4:3 * d4] = torch.sin(ys[:, None] * div)
        pe[:, 3 * d4:4 * d4] = torch.cos(ys[:, None] * div)
        if pe.size(0) < n_vis:
            pe = torch.cat([pe, pe.new_zeros(n_vis - pe.size(0), D)], dim=0)
        else:
            pe = pe[:n_vis]
        pe = pe.to(device=device, dtype=dtype).unsqueeze(0)  # [1, n_vis, D]
        cache[key] = pe
        return pe

    def _grid_coords(self, n_vis, device, dtype):
        """Normalised (x, y) patch-centre coordinates in [0,1] for the grid."""
        cache = getattr(self, "_gc_cache", None)
        if cache is None:
            cache = {}
            self._gc_cache = cache
        key = (int(n_vis), str(device), str(dtype))
        if key in cache:
            return cache[key]
        g = int(round(float(n_vis) ** 0.5))
        yy, xx = torch.meshgrid(torch.arange(g, dtype=torch.float32),
                                torch.arange(g, dtype=torch.float32),
                                indexing="ij")
        xs = ((xx.reshape(-1) + 0.5) / g)
        ys = ((yy.reshape(-1) + 0.5) / g)
        coords = torch.stack([xs, ys], dim=-1)  # [g*g, 2]
        if coords.size(0) < n_vis:
            coords = torch.cat(
                [coords, coords.new_full((n_vis - coords.size(0), 2), 0.5)], dim=0)
        else:
            coords = coords[:n_vis]
        coords = coords.to(device=device, dtype=dtype)
        cache[key] = coords
        return coords

    def predict_boxes(self, hidden, n_vis, L_txt, input_ids=None, labels=None,
                      vis_kv=None):
        """Decode a box by SOFT-ARGMAX over a text-conditioned attention heatmap.

        The earlier design regressed 4 coordinates from a pooled attention feature.
        That path repeatedly collapsed to a single constant box (the dataset mean),
        because coordinate regression from a global feature has no structural link
        to *where* the model attends, so the mean box is a strong local optimum.

        Here the box CENTRE is the attention-weighted mean of the patch-grid
        coordinates (soft-argmax), so it is input-dependent by construction and the
        gradient directly pushes attention toward the ground-truth centre. The box
        SIZE derives from the attention spread (2nd moment) modulated by a learned
        head, so tighter attention → tighter box.

        Localisation KV (``vis_kv``): the CLEAN per-patch projector tokens (pre-LM),
        NOT the LM's vision-position hidden states. In a causal SSM/attention trunk
        with vision-first ordering, the vision hidden state at patch i is a 1D
        prefix-scan over patches 0..i (raster order) — the 2D patch identity is
        compressed away, so attention over it cannot localise and the soft-argmax
        centre collapses to the image centroid (observed center_std≈1e-3). The
        projector tokens keep full 2D patch structure (+ our grid PE), and this also
        gives the projector a direct spatial gradient instead of one washed through
        66 LM layers. The text QUERY still comes from the LM hidden (last prompt
        token) so we keep the LM's strength: understanding the referring expression
        after it has attended to the image. Falls back to LM hidden if vis_kv=None.

        Conditioning uses the LAST PROMPT token (causal summary of image + referring
        expression, matching where a box is read at inference).
        """
        B = hidden.size(0)
        kv_dtype = self.grd_ln_kv.weight.dtype
        vis_src = hidden[:, :n_vis, :] if vis_kv is None else vis_kv
        vis_h = self.grd_ln_kv(vis_src.to(kv_dtype))              # [B, N, D]
        # Inject spatial coordinate basis so attention can localise.
        vis_h = vis_h + self._grid_pos_embed(n_vis, vis_h.device, vis_h.dtype)
        txt_h = hidden[:, n_vis:n_vis + L_txt, :]                 # [B, L, D]
        ar = torch.arange(B, device=txt_h.device)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if labels is not None:
            # labels: prompt & pad = -100, answer tokens = real ids. Last prompt
            # token sits just before the first answer token.
            real = (labels != -100)
            has_resp = real.any(dim=1)
            resp_start = torch.argmax(real.int(), dim=1)          # first answer idx
            last_prompt = torch.where(has_resp, resp_start - 1,
                                      torch.full_like(resp_start, L_txt - 1))
            last_prompt = last_prompt.clamp(min=0, max=L_txt - 1)
            txt = txt_h[ar, last_prompt]
        elif input_ids is not None and pad_id is not None:
            lengths = (input_ids != pad_id).sum(dim=1).clamp(min=1)
            txt = txt_h[ar, (lengths - 1).clamp(max=L_txt - 1)]
        else:
            txt = txt_h[:, -1]
        q = self.grd_queries.unsqueeze(0).expand(B, -1, -1).to(kv_dtype)
        q = q + self.grd_txt_proj(txt.to(kv_dtype)).unsqueeze(1)
        q = self.grd_ln_q(q)
        # attention weights over patches drive the soft-argmax; attn_out feeds size.
        attn_out, attn_w = self.grd_attn(q, vis_h, vis_h, need_weights=True,
                                         average_attn_weights=True)  # [B,n_q,N]
        a = attn_w.float().mean(dim=1)                           # [B, N]
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        grid = self._grid_coords(n_vis, a.device, a.dtype)      # [N, 2]
        gx, gy = grid[:, 0], grid[:, 1]
        cx = (a * gx).sum(-1)                                   # [B] soft-argmax
        cy = (a * gy).sum(-1)
        sx = (a * (gx - cx.unsqueeze(-1)) ** 2).sum(-1).clamp(min=1e-8).sqrt()
        sy = (a * (gy - cy.unsqueeze(-1)) ** 2).sum(-1).clamp(min=1e-8).sqrt()
        grd_h = self.grd_ln_out(attn_out.mean(dim=1))
        head_dtype = next(self.grd_head.parameters()).dtype
        sz = self.grd_head(grd_h.to(head_dtype)).float()        # [B,4]
        # Centre = soft-argmax (+ tiny learned refine). Size is regressed DIRECTLY
        # by the head, bounded to [0, 0.5] half-extent, decoupled from the
        # attention spread: coupling size to spread let the box grow to the whole
        # image (giant-box degenerate) as attention diffused.
        _ = (sx, sy)  # spread kept for potential diagnostics
        cx = (cx + 0.1 * torch.tanh(sz[:, 0])).clamp(0.0, 1.0)
        cy = (cy + 0.1 * torch.tanh(sz[:, 1])).clamp(0.0, 1.0)
        hw = (torch.sigmoid(sz[:, 2]) * 0.5).clamp(1e-3, 0.5)
        hh = (torch.sigmoid(sz[:, 3]) * 0.5).clamp(1e-3, 0.5)
        x1 = (cx - hw).clamp(0.0, 1.0)
        x2 = (cx + hw).clamp(0.0, 1.0)
        y1 = (cy - hh).clamp(0.0, 1.0)
        y2 = (cy + hh).clamp(0.0, 1.0)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward_s2(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,       # [B,4] xyxy in [0,1]
        has_bbox: Optional[torch.Tensor] = None,   # [B] 0/1
        lambda_grd: float = 1.0,
        lambda_blank: float = 0.5,
        lambda_shuffle: float = 0.75,
        blank_margin: float = 0.15,
        shuffle_margin: float = 0.10,
        blank_bs: int = 2,
    ) -> dict:
        """CLM + grounding (L1+GIoU) + blank & shuffle contrastive hinges."""
        enc = self._encode_multimodal(pixel_values, input_ids, append_grd=False)
        hidden = self._trunk_hidden(enc["inputs_embeds"], enc["attn"])
        n_vis, L_txt = enc["n_vis"], enc["L_txt"]
        pred_h = hidden[:, n_vis - 1: n_vis - 1 + L_txt, :]
        logits = self.lm.get_output_embeddings()(pred_h) * enc["head_mult"]
        loss_clm = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1), ignore_index=-100)

        loss_grd = pixel_values.new_zeros(())
        iou = pixel_values.new_zeros(())
        if bbox is not None and has_bbox is not None and has_bbox.sum() > 0:
            # KV = clean projector patch tokens (enc inputs_embeds[:, :n_vis]),
            # NOT the causally-scanned LM hidden. See predict_boxes docstring.
            vis_kv = enc["inputs_embeds"][:, :n_vis, :]
            pred = self.predict_boxes(hidden, n_vis, L_txt, input_ids, labels,
                                      vis_kv=vis_kv)  # [B,4] xyxy
            mask = has_bbox > 0.5
            p, g = pred[mask], bbox[mask].float()
            l1 = nn.functional.l1_loss(p, g, reduction="mean")
            iou_vals = _batch_iou(p, g)
            # L1 + (1 - CIoU) + over-coverage penalty. CIoU adds an aspect-ratio /
            # shape term on top of DIoU's centre distance, and the over-coverage
            # term relu(area_pred - area_gt) explicitly punishes predicting a box
            # LARGER than the GT — together they break the size-collapse degenerate
            # (a giant box that hedges IoU) that survived plain DIoU.
            area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
            area_g = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
            over = nn.functional.relu(area_p - area_g).mean()
            loss_grd = l1 + _ciou_loss(p, g).mean() + 0.5 * over
            with torch.no_grad():
                iou = iou_vals.mean()

        loss_blank = pixel_values.new_zeros(())
        if lambda_blank > 0 and blank_bs > 0:
            b = min(blank_bs, pixel_values.size(0))
            pv0 = torch.zeros_like(pixel_values[:b])
            out_b = self.forward_clm(pv0, input_ids[:b], labels[:b])
            gap = out_b["loss"] - loss_clm.detach()
            loss_blank = nn.functional.relu(
                torch.as_tensor(blank_margin, device=gap.device, dtype=gap.dtype) - gap)

        # Shuffle contrastive hinge: real *wrong* image must be worse than the
        # matched image by a margin. Directly targets the shuffle metric that
        # blank-only supervision failed to move.
        loss_shuffle = pixel_values.new_zeros(())
        if lambda_shuffle > 0 and pixel_values.size(0) > 1:
            b = min(max(blank_bs, 2), pixel_values.size(0))
            pv_shuf = pixel_values[:b].roll(1, dims=0)
            out_s = self.forward_clm(pv_shuf, input_ids[:b], labels[:b])
            gap_s = out_s["loss"] - loss_clm.detach()
            loss_shuffle = nn.functional.relu(
                torch.as_tensor(shuffle_margin, device=gap_s.device,
                                dtype=gap_s.dtype) - gap_s)

        loss = (loss_clm + lambda_grd * loss_grd
                + lambda_blank * loss_blank + lambda_shuffle * loss_shuffle)
        return {
            "loss": loss,
            "loss_clm": loss_clm.detach(),
            "loss_grd": loss_grd.detach() if torch.is_tensor(loss_grd) else loss_grd,
            "loss_blank": loss_blank.detach() if torch.is_tensor(loss_blank) else loss_blank,
            "loss_shuffle": loss_shuffle.detach() if torch.is_tensor(loss_shuffle) else loss_shuffle,
            "iou": float(iou) if torch.is_tensor(iou) else 0.0,
            "logits": logits,
            "n_vis_tokens": n_vis,
        }

    # ── S3 action head (waypoint regression) ─────────────────────────────
    def enable_action_head(
        self,
        chunk_size: int = 8,
        proprio_dim: int = 4,
        action_dim: int = 4,
        num_freqs: int = 8,
        proprio_dropout: float = 0.0,
        proprio_residual: bool = False,
        use_proprio: bool = True,
        use_grounding_target: bool = False,
        readout: str = "last",
        n_bins: int = 0,
        bin_range: float = 1.5,
        readout_layers: int = 2,
    ) -> None:
        """Attach the ProprioEncoder + UAVActionHead sized to the LM width.

        Kept in fp32 (master precision) rather than the bf16 backbone dtype:
        the action regression works in per-(k,dim) z-space where dz/dyaw std is
        ~1/8 of dx, so bf16 rounding on the head would swamp exactly the small
        channels we need. Autocast still runs the matmuls in bf16 for speed.

        ``proprio_dim=4`` (episode-local pose [x, y, z, yaw]) DROPS the velocity
        channels on purpose: v1 fed 8-DOF [pose|velocity] and the head learnt to
        just EXTRAPOLATE the velocity — a shortcut that scored well offline (GT
        velocity leaks the next step) but COLLAPSED in closed loop (cold-start
        velocity=0 -> forward creep, ignoring instruction+vision). Pose-only +
        ``proprio_dropout`` forces the policy to ground language + image instead.
        """
        from model.proprio_encoder import ProprioEncoder
        from model.action_head import UAVActionHead, UAVActionHeadV5

        self.action_readout = readout
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.proprio_dropout = float(proprio_dropout)
        # Residual DISABLED by default: the direct hidden+prop_tok path let the
        # absolute pose dominate the readout (G4: prop_sens 0.48 > instr_sens
        # 0.23) even though gt_action is anchor-relative body-frame and must not
        # depend on absolute pose. Proprio now influences ONLY as an in-context
        # token, so vision+instruction grounding drives the terminal hidden.
        self.proprio_residual = bool(proprio_residual)
        # use_proprio=False -> PROPRIO-FREE policy: the proprio token is zeroed in
        # BOTH train and eval so absolute pose can never influence the action.
        # gt_action is anchor-relative body-frame, so pose is a pure shortcut
        # (redundant with the FPV image); at sustained full-run LR the LoRA
        # re-learned to route pose and suppress the instruction (val@1000:
        # instr 0.004 << prop 0.315) even with the residual removed + dropout.
        # Dropping pose entirely removes that degenerate basin.
        self.use_proprio = bool(use_proprio)
        if not self.use_proprio and readout == "xattn":
            # v5: the cross-attention readout carries its OWN K learned queries,
            # so the single learned action-query token that stood in for the
            # dropped pose token is redundant — the readout subsumes it.
            pass
        elif not self.use_proprio:
            # Learned action-query readout token REPLACING the pose token. A zeros
            # token is out-of-distribution for the LM (it killed grounding:
            # instr/vis sens -> 0); a learned in-distribution query attends over
            # [vision|text] and its terminal hidden is the action readout. With NO
            # pose input there is no pose-shortcut attractor, so grounding cannot
            # be eroded by longer training (unlike pose-only, where prop_sens
            # crept 0.008 -> 0.76 from step 750 -> 1500).
            self.action_query = nn.Parameter(torch.randn(1, 1, self.d_lm) * 0.02)
        # GROUNDING-TARGET TOKEN (v5): closed-loop failed because the action head
        # was vision-blind (vis_sens 0.07): it executed an instruction->motion
        # template and marched straight, never homing on the target (no curving /
        # orbiting / stopping). The frozen S2 grounding head ALREADY localises the
        # referred object ("the car") as a box in the FPV image. We feed that box
        # (centre offset = steering direction, size = proximity) through a small
        # learned encoder into the action readout so the policy can turn toward /
        # approach / stop at the visual target. This is the GuidedVLA "object head"
        # / object-centric-token idea, reusing our own grounding capability; it
        # forces vision reliance because a changed image -> changed box -> changed
        # action (raises I(action; vision)).
        self.use_grounding_target = bool(use_grounding_target)
        if self.use_grounding_target:
            self.target_encoder = nn.Sequential(
                nn.Linear(6, self.d_lm), nn.GELU(),
                nn.Linear(self.d_lm, self.d_lm),
            ).to(torch.float32)
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=proprio_dim, mamba_hidden_size=self.d_lm,
            num_freqs=num_freqs)
        if readout == "xattn":
            self.action_head = UAVActionHeadV5(
                mamba_hidden_size=self.d_lm, chunk_size=chunk_size,
                action_dim=action_dim, n_bins=n_bins, bin_range=bin_range,
                n_layers=readout_layers)
        elif readout == "last":
            self.action_head = UAVActionHead(
                mamba_hidden_size=self.d_lm, chunk_size=chunk_size,
                action_dim=action_dim)
        else:
            raise ValueError(f"unknown readout '{readout}' (last|xattn)")
        print(f"[AeroV2] action head enabled: K={chunk_size} d_lm={self.d_lm} "
              f"readout={readout} n_bins={n_bins} "
              f"proprio_dim={proprio_dim} action_dim={action_dim} "
              f"use_proprio={self.use_proprio} "
              f"grounding_target={self.use_grounding_target}")

    def configure_stage3(self, train_lora: bool = False,
                         grad_ckpt: bool = True) -> None:
        """S3: freeze vision+projector+grounding+LM; train proprio + action head.

        The action head reads only the trunk's terminal hidden state (modules
        AFTER the frozen LM), but the proprio encoder sits BEFORE the trunk, so
        gradients still traverse the (frozen) 66-layer LM — hence we checkpoint
        it to bound activation memory. Set ``train_lora`` to also adapt the S0/S2
        LoRA to UAV-Flow dynamics.
        """
        assert hasattr(self, "action_head"), "call enable_action_head() first"
        for p in self.parameters():
            p.requires_grad = False
        if self.use_proprio:
            for p in self.proprio_encoder.parameters():
                p.requires_grad = True
        for p in self.action_head.parameters():
            p.requires_grad = True
        if hasattr(self, "action_query"):
            self.action_query.requires_grad = True
        if getattr(self, "use_grounding_target", False):
            for p in self.target_encoder.parameters():
                p.requires_grad = True
        if train_lora:
            for n, p in self.lm.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
        if grad_ckpt:
            self._enable_grad_ckpt()
        print("[AeroV2] S3: proprio_encoder + action_head"
              + (" + LoRA" if train_lora else "") + " trainable")

    def forward_action(
        self,
        pixel_values: torch.Tensor,     # [B, 3, H, W]
        input_ids: torch.Tensor,        # [B, L]
        proprio: torch.Tensor,          # [B, proprio_dim]
        gt_action: Optional[torch.Tensor] = None,   # [B, K, 4] physical units
        lambda_smooth: float = 0.0,
        lambda_endpoint: float = 0.25,
        lambda_direction: float = 0.5,
        lambda_acc: float = 0.0,
        lambda_var: float = 0.0,
        var_floor: float = 0.3,
        channel_weights: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        drop_proprio: bool = False,
    ) -> dict:
        """[vision | text | proprio] -> trunk -> last token -> action chunk.

        The proprio token is appended LAST so the causal trunk's terminal hidden
        state (RoboMamba global-token convention) has integrated the full image +
        instruction + current UAV state before the head decodes K waypoints.

        proprio dropout: during training the proprio token is zeroed per-sample
        with prob ``self.proprio_dropout`` (classifier-free style) so the policy
        cannot lean on proprio and MUST decode the action from the grounded
        image+instruction hidden state. ``drop_proprio=True`` forces this at eval
        for the grounding probe (output must still respond to instruction/vision).
        """
        enc = self._encode_multimodal(pixel_values, input_ids, append_grd=False)
        if getattr(self, "action_readout", "last") == "xattn":
            return self._forward_action_v5(
                enc, input_ids, proprio, gt_action=gt_action,
                lambda_endpoint=lambda_endpoint,
                lambda_direction=lambda_direction,
                lambda_var=lambda_var,
                var_floor=var_floor,
                channel_weights=channel_weights)
        prop_tok = self.proprio_encoder(proprio.float()).to(
            enc["inputs_embeds"].dtype)                     # [B, 1, D]
        p_drop = getattr(self, "proprio_dropout", 0.0)
        if not getattr(self, "use_proprio", True):
            # proprio-free: use the learned action-query token (pose ignored)
            prop_tok = self.action_query.expand(prop_tok.size(0), -1, -1).to(
                prop_tok.dtype)
        elif drop_proprio:
            prop_tok = torch.zeros_like(prop_tok)     # eval grounding probe
        elif self.training and p_drop > 0.0:
            keep = (torch.rand(prop_tok.size(0), 1, 1, device=prop_tok.device)
                    >= p_drop).to(prop_tok.dtype)
            prop_tok = prop_tok * keep
        inputs_embeds = torch.cat([enc["inputs_embeds"], prop_tok], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                          device=inputs_embeds.device)
        hidden = self._trunk_hidden(inputs_embeds, attn)
        # Terminal hidden = the grounded [vision|text|proprio] readout. The
        # optional proprio residual (below) gives ProprioEncoder a direct grad
        # path but ALSO lets absolute pose dominate the readout (see
        # proprio_residual note); with LoRA trainable the proprio token's grad
        # now survives the trunk, so we default the residual OFF and let
        # vision+instruction grounding own the terminal hidden.
        global_token = hidden[:, -1, :]
        if getattr(self, "proprio_residual", False):
            global_token = global_token + prop_tok.squeeze(1).to(hidden.dtype)
        # GROUNDING-TARGET fusion: localise the referred object with the frozen S2
        # grounding head and inject its box (where + how large = direction +
        # proximity) into the action readout so the policy homes on the visual
        # target instead of marching a fixed instruction template (vis_sens 0.07).
        if getattr(self, "use_grounding_target", False):
            n_vis = enc["n_vis"]
            vis_kv = enc["inputs_embeds"][:, :n_vis, :]
            # box is a FROZEN-head function of vision+text; detach so it acts as a
            # stable target descriptor (the learned target_encoder consumes it).
            box = self.predict_boxes(hidden, n_vis, enc["L_txt"],
                                     input_ids=input_ids, vis_kv=vis_kv).detach()
            x1, y1, x2, y2 = box.unbind(-1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            w, h = (x2 - x1), (y2 - y1)
            desc = torch.stack([cx, cy, w, h, cx - 0.5, cy - 0.5], dim=-1)  # [B,6]
            tgt_tok = self.target_encoder(desc.float())            # [B, D]
            global_token = global_token + tgt_tok.to(global_token.dtype)
        head_dtype = next(self.action_head.parameters()).dtype
        pred = self.action_head(global_token.to(head_dtype), proprio)
        out = {"action": pred["action"]}
        if gt_action is not None:
            from model.action_head import aero_action_loss
            loss, detail = aero_action_loss(
                pred, gt_action, head=self.action_head,
                lambda_smooth=lambda_smooth, lambda_endpoint=lambda_endpoint,
                lambda_direction=lambda_direction, lambda_acc=lambda_acc,
                lambda_var=lambda_var,
                channel_weights=channel_weights, sample_weights=sample_weights)
            out["loss"] = loss
            out["loss_detail"] = detail
        return out

    def _forward_action_v5(
        self,
        enc: dict,
        input_ids: torch.Tensor,
        proprio: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None,
        lambda_endpoint: float = 0.5,
        lambda_direction: float = 0.5,
        lambda_var: float = 0.5,
        var_floor: float = 0.3,
        channel_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """v5 policy forward: K action queries cross-attend [patches | text].

        Keys/values are the projector's 2D patches — taken BEFORE the trunk, so
        the causal scan has not yet compressed them into a global summary —
        concatenated with the trunk's contextualised text states. Both
        modalities therefore reach the action only through the attention, with
        no additive path that could carry the instruction while skipping the
        image (the failure the first v5 smoke measured: |ctx| 2955 vs |q| 0.82).
        """
        inputs_embeds = enc["inputs_embeds"]
        n_vis = enc["n_vis"]
        if self.use_proprio:
            prop_tok = self.proprio_encoder(proprio.float()).to(
                inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, prop_tok], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                          device=inputs_embeds.device)
        hidden = self._trunk_hidden(inputs_embeds, attn)

        kv = torch.cat([enc["inputs_embeds"][:, :n_vis], hidden[:, n_vis:]], dim=1)
        # Instructions are padded to max_text_len; without a mask the readout
        # would spend most of its text budget on pad positions.
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            key_padding_mask = None
        else:
            text_pad = input_ids.eq(pad_id)
            key_padding_mask = torch.cat([
                torch.zeros(kv.size(0), n_vis, dtype=torch.bool, device=kv.device),
                text_pad,
                torch.zeros(kv.size(0), kv.size(1) - n_vis - text_pad.size(1),
                            dtype=torch.bool, device=kv.device),
            ], dim=1)

        head_dtype = next(self.action_head.parameters()).dtype
        pred = self.action_head(kv.to(head_dtype), key_padding_mask)
        out = {"action": pred["action"]}
        if gt_action is not None:
            from model.action_head import aero_action_loss_v5
            loss, detail = aero_action_loss_v5(
                pred, gt_action, self.action_head,
                lambda_endpoint=lambda_endpoint,
                lambda_direction=lambda_direction,
                lambda_var=lambda_var,
                var_floor=var_floor,
                channel_weights=channel_weights)
            out["loss"] = loss
            out["loss_detail"] = detail
        return out


def _batch_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """IoU for boxes in [0,1] xyxy. pred/gt: [N,4]."""
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + 1e-6
    return inter / union


def _diou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - DIoU for boxes in [0,1] xyxy. pred/gt: [N,4].

    DIoU = IoU - d²(centres) / c²(enclosing diagonal). The centre-distance term
    gives a per-sample gradient toward the ground-truth centre even when boxes
    do not overlap, so it cannot be satisfied by one shared/giant box.
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + 1e-6
    iou = inter / union
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    gcx, gcy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
    d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7
    return 1.0 - (iou - d2 / c2)


def _ciou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - CIoU for boxes in [0,1] xyxy. pred/gt: [N,4].

    CIoU = IoU - d²/c² - α·v. On top of DIoU's centre-distance term it adds an
    aspect-ratio consistency term v = (4/π²)(atan(wg/hg) - atan(wp/hp))², so a
    box that overlaps the GT centre but has the WRONG shape/size (the giant-box
    degenerate that survives DIoU) is still penalised.
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    gw = (gx2 - gx1).clamp(min=1e-6)
    gh = (gy2 - gy1).clamp(min=1e-6)
    union = pw * ph + gw * gh - inter + 1e-6
    iou = inter / union
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    gcx, gcy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
    d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7
    import math
    v = (4.0 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + 1e-7)
    return 1.0 - (iou - d2 / c2 - alpha * v)


def _giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - GIoU for boxes in [0,1] xyxy. pred/gt: [N,4]."""
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + 1e-6
    iou = inter / union
    # smallest enclosing box
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    area_c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6
    giou = iou - (area_c - union) / area_c
    return 1.0 - giou
