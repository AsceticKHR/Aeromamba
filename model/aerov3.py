"""AeroV3 — a stateful sub-1B policy for HUGE-Bench.

The policy factorises as ``pi(a | z_ep, pose, phase)``:

    z_ep    episode-constant task frame, 128-d, from (first_image, instruction,
            initial pose). Computed **once per episode**.
    phase   64-d recurrent filter, supervised by the official ``subtask_id``.
    pose    given exactly by the benchmark, no estimation needed.

That split is the whole design. It comes from a training-free audit of the
benchmark: stage information cuts action error 22.8-55.5% on the Orbit family
while being unreadable from pose (4 of 6 groups score *below* chance), and
contributes nothing on Inspect, where a pose-only predictor already lands
within 0.24-0.78 m of an 11.33 m signal. So the missing quantity is about one
bit, in one task family -- a filter, not a memory system.

Two consequences show up directly in this file:

- The language model runs in ``encode_task`` only. Per control step the cost is
  a frozen vision encoder, a GRU cell and a 20-query cross-attention read-out.
- Vision only has to estimate ``z_ep`` and ``phase``, so it can refresh slower
  than control. ``vision_update`` carries the mask; the rate is trained as an
  input, which makes the frequency ablation inference-only.

The recurrence is detached between steps and supervised at every step. This is
deliberate. BPTT exists to solve long-range credit assignment, and per-step
stage labels remove that problem. It also explains why the 14 recurrent
variants in RoboMME failed while RB-VLA's recurrent belief worked: the failed
ones were driven by action loss alone.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BACKBONE = "Qwen/Qwen3-0.6B"
MAX_STAGES = 16
POSE_FEAT_DIM = 8


# ── configuration ────────────────────────────────────────────────────────────

@dataclass
class AeroV3Config:
    backbone_id: str = DEFAULT_BACKBONE
    vision_type: str = "cradio_v3_b"
    img_size: int = 256
    d_task: int = 128           # z_ep
    d_phase: int = 64           # phi
    d_dec: int = 256            # read-out width
    n_heads: int = 8
    horizon: int = 20
    action_dim: int = 4
    num_stages: int = MAX_STAGES
    use_phase: bool = True      # ablation A sets this False
    train_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    loss: dict = field(default_factory=lambda: {
        "action": 1.0, "stage": 0.5, "progress": 0.1})


# ── pose features ────────────────────────────────────────────────────────────

def pose_features(pose: torch.Tensor, pose0: torch.Tensor) -> torch.Tensor:
    """World pose -> a scale-sane 8-d feature.

    Displacement is divided by 50 m, roughly the working scale of these
    trajectories, and yaw goes in as sin/cos so the wrap at +-pi is not a cliff.
    Absolute altitude is kept because instructions name it ("fly to 60 meters
    above"), which relative displacement alone would throw away.
    """
    d = (pose[..., :3] - pose0[..., :3]) / 50.0
    a, a0 = pose[..., 3], pose0[..., 3]
    return torch.cat([d, (pose[..., 2:3]) / 50.0,
                      torch.sin(a).unsqueeze(-1), torch.cos(a).unsqueeze(-1),
                      torch.sin(a0).unsqueeze(-1), torch.cos(a0).unsqueeze(-1)],
                     dim=-1)


# ── building blocks ──────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), act(),
                                 nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


class ActionReadout(nn.Module):
    """``horizon`` learned queries cross-attend over ``[z_ep | patch tokens]``.

    No additive context path. The v2 line had one (``ctx_proj``) and it was
    measured at 3600x the query norm, which reduced the read-out to a bypass.
    Conditioning enters through the queries only.
    """

    def __init__(self, cfg: AeroV3Config, d_cond: int):
        super().__init__()
        d = cfg.d_dec
        self.query = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.cond = nn.Linear(d_cond, d)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.ff = MLP(d, 4 * d, d)
        self.norm_o = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.action_dim)
        # Small non-zero init. A zero-init final layer collapsed the action
        # spread on the v2 line and no amount of variance regularisation
        # recovered it.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.query.unsqueeze(0) + self.cond(cond).unsqueeze(1))
        o, _ = self.attn(q, self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        o = self.norm_o(q + o)
        return self.head(o + self.ff(o))


# ── backbone loading ─────────────────────────────────────────────────────────

def _offline() -> bool:
    return os.environ.get("AEROMAMBA_OFFLINE", "0") == "1"


def load_backbone(model_id: str, dtype=torch.float32):
    """Offline mode builds a tiny random Llama so the wiring contract can be
    tested on CPU without any weights."""
    from transformers import AutoTokenizer

    if _offline():
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=1024, hidden_size=128,
                          intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4)
        lm = LlamaForCausalLM(cfg)
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", local_files_only=True) \
            if os.environ.get("AEROV3_OFFLINE_TOK") else None
        return lm, tok

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                              trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return lm, tok


def build_vision(cfg: AeroV3Config):
    if _offline():
        from model.vision import VisionEncoder
        return VisionEncoder("siglip_l_384", freeze=True)
    if cfg.vision_type.startswith("cradio"):
        from model.vision import RADIOVisionEncoder
        return RADIOVisionEncoder(cfg.vision_type, img_size=cfg.img_size,
                                  freeze=True)
    from model.vision import HFVisionEncoder
    return HFVisionEncoder(cfg.vision_type, freeze=True)


# ── the model ────────────────────────────────────────────────────────────────

class AeroV3(nn.Module):
    def __init__(self, cfg: AeroV3Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or AeroV3Config()

        self.vision = build_vision(cfg)
        for p in self.vision.parameters():
            p.requires_grad = False
        d_vis = self.vision.hidden_size

        self.lm, self.tokenizer = load_backbone(cfg.backbone_id)
        d_lm = self.lm.config.hidden_size
        for p in self.lm.parameters():
            p.requires_grad = False
        if cfg.train_lora and not _offline():
            from peft import LoraConfig, TaskType, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha, lora_dropout=0.05, bias="none",
                target_modules=list(cfg.lora_targets)))

        # Two vision projections: one into LM space, used once per episode for
        # the task frame; one into read-out space, used every step.
        self.vis_to_lm = MLP(d_vis, d_lm, d_lm)
        self.vis_to_dec = nn.Linear(d_vis, cfg.d_dec)
        self.pose_to_lm = MLP(POSE_FEAT_DIM, d_lm, d_lm)

        self.task_head = MLP(d_lm, d_lm, cfg.d_task)
        self.task_to_dec = nn.Linear(cfg.d_task, cfg.d_dec)

        d_p = cfg.d_task + POSE_FEAT_DIM + cfg.d_dec
        self.phase_in = MLP(d_p, 2 * cfg.d_phase, cfg.d_phase)
        self.phase_cell = nn.GRUCell(cfg.d_phase, cfg.d_phase)
        self.phase_init = nn.Parameter(torch.zeros(cfg.d_phase))

        d_cond = cfg.d_task + POSE_FEAT_DIM + (cfg.d_phase if cfg.use_phase else 0)
        self.readout = ActionReadout(cfg, d_cond)

        # Stage and progress read from phi alone. Routing z_ep in here would let
        # the heads score well without the recurrence carrying anything, and G4
        # would stop measuring what it claims to.
        self.stage_head = nn.Linear(cfg.d_phase, cfg.num_stages)
        self.progress_head = nn.Linear(cfg.d_phase, 1)

    # ── frozen vision ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,N,d_vis). Always no-grad: the encoder is frozen and
        keeping the graph would cost memory for gradients that are zero."""
        return self.vision(pixel_values).float()

    # ── timescale 1: once per episode ────────────────────────────────────────

    def encode_task(self, first_pixel_values: torch.Tensor,
                    input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    pose0: torch.Tensor) -> torch.Tensor:
        """(first_image, instruction, initial pose) -> z_ep (B, d_task).

        Cacheable by construction: every input is episode-constant. The wiring
        test asserts the cached and recomputed values match bitwise, because
        that is an identity and not an approximation.
        """
        vis = self.encode_vision(first_pixel_values)
        v = self.vis_to_lm(vis)                                    # (B,N,d_lm)

        emb = self.lm.get_input_embeddings()(input_ids)            # (B,L,d_lm)
        p = self.pose_to_lm(pose_features(pose0, pose0)).unsqueeze(1)

        seq = torch.cat([v, emb, p], dim=1)
        mask = torch.cat([
            torch.ones(v.shape[:2], dtype=attention_mask.dtype, device=v.device),
            attention_mask,
            torch.ones(p.shape[:2], dtype=attention_mask.dtype, device=v.device),
        ], dim=1)

        out = self.lm(inputs_embeds=seq, attention_mask=mask,
                      output_hidden_states=True)
        h = out.hidden_states[-1]
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.task_head(pooled.float())

    # ── timescale 2: once per inference step ─────────────────────────────────

    def step(self, z_ep: torch.Tensor, pose: torch.Tensor, pose0: torch.Tensor,
             vis_tokens: torch.Tensor, phi_prev: torch.Tensor | None):
        """One inference step. ``vis_tokens`` may be a cached encoding from an
        earlier step -- that is the whole point of the vision-rate knob."""
        cfg = self.cfg
        B = z_ep.shape[0]
        pf = pose_features(pose, pose0)
        kv_vis = self.vis_to_dec(vis_tokens)                       # (B,N,d_dec)
        z_tok = self.task_to_dec(z_ep).unsqueeze(1)                # (B,1,d_dec)
        kv = torch.cat([z_tok, kv_vis], dim=1)                     # order fixed

        if phi_prev is None:
            # The learned initial state must stay attached, or it never trains.
            phi_prev = self.phase_init.expand(B, cfg.d_phase)
        else:
            # The filter is trained by its own per-step labels, so no gradient
            # should cross a step boundary.
            phi_prev = phi_prev.detach()
        x = self.phase_in(torch.cat([z_ep, pf, kv_vis.mean(1)], dim=-1))
        phi = self.phase_cell(x, phi_prev)

        cond = torch.cat([z_ep, pf, phi], dim=-1) if cfg.use_phase \
            else torch.cat([z_ep, pf], dim=-1)
        action = self.readout(cond, kv)
        return {
            "action": action,
            "phase": phi,
            "stage_logits": self.stage_head(phi),
            "progress": self.progress_head(phi).squeeze(-1),
        }

    # ── training forward over a window ───────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        px = batch["pixel_values"]                      # (B,W,3,H,W)
        B, W = px.shape[:2]

        z_ep = self.encode_task(batch["first_pixel_values"], batch["input_ids"],
                                batch["attention_mask"], batch["pose0"])

        upd = batch["vision_update"]                    # (B,W)
        flat = self.encode_vision(px.flatten(0, 1))
        vis = flat.view(B, W, *flat.shape[1:])

        phi, out = None, []
        held = None
        for k in range(W):
            fresh = upd[:, k].view(B, *([1] * (vis.dim() - 2)))
            held = vis[:, k] if held is None else \
                fresh * vis[:, k] + (1.0 - fresh) * held
            r = self.step(z_ep, batch["pose"][:, k], batch["pose0"], held, phi)
            phi = r["phase"]
            out.append(r)

        stack = lambda k: torch.stack([o[k] for o in out], dim=1)
        pred = {"action": stack("action"), "stage_logits": stack("stage_logits"),
                "progress": stack("progress"), "phase": stack("phase"),
                "z_ep": z_ep}
        pred.update(self.losses(pred, batch))
        return pred

    def losses(self, pred: dict, batch: dict) -> dict:
        w = self.cfg.loss
        m = batch["action_mask"].unsqueeze(-1)
        l_act = (F.l1_loss(pred["action"], batch["action"], reduction="none")
                 * m).sum() / m.sum().clamp(min=1.0) / self.cfg.action_dim

        st = batch["stage"]
        valid = st >= 0
        if valid.any():
            tgt = st[valid].clamp(max=self.cfg.num_stages - 1)
            l_stage = F.cross_entropy(pred["stage_logits"][valid], tgt)
            acc = (pred["stage_logits"][valid].argmax(-1) == tgt).float().mean()
        else:
            l_stage = pred["action"].sum() * 0.0
            acc = torch.zeros((), device=st.device)

        l_prog = F.l1_loss(pred["progress"], batch["progress"])
        total = w["action"] * l_act + w["stage"] * l_stage + w["progress"] * l_prog
        return {"loss": total, "loss_action": l_act.detach(),
                "loss_stage": l_stage.detach(), "loss_progress": l_prog.detach(),
                "stage_acc": acc.detach()}

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_report(self) -> str:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        per_step = sum(p.numel() for m in (self.vis_to_dec, self.task_to_dec,
                                           self.phase_in, self.phase_cell,
                                           self.readout, self.stage_head,
                                           self.progress_head)
                       for p in m.parameters())
        return (f"total {tot/1e6:.1f}M  trainable {tr/1e6:.1f}M  "
                f"per-step (excl. vision) {per_step/1e6:.2f}M")
