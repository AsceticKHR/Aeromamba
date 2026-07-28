"""
UAV Action Head for AeroMamba-VLA.

Direct chunk regression — no intermediate sequence decoder.

Architecture (RoboMamba two_mlp dimension-cascade style):
    global_token [B, D_m]
        ↓  3-layer MLP  (D_m → D_m/2 → D_m/4 → K*4)
    [B, K*4]
        ↓  reshape
    [B, K, 4]   action chunk: (Δx, Δy, Δz, Δyaw_rad)

Why simpler than the original SSMDecoder approach:
  - K=5 is tiny; Cross-Attention decoder is overkill for such a short sequence.
  - Direct L1 regression over K*4 outputs is more stable than sin/cos split
    (no normalisation constraints, no atan2 at inference time).
  - Matches OpenVLA-OFT "parallel decode + L1 regression" that achieves 26×
    faster inference than token-by-token autoregressive decoding.
  - Zero-init on the final layer keeps initial actions near zero,
    which is safe for physical deployment.

References:
  RoboMamba (lmzpai/roboMamba)  — two_mlp policy head
  OpenVLA-OFT (Kim et al. 2025) — action chunking + L1 regression
  ACT (Zhao et al. 2023)        — temporal ensemble for smooth execution
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ---------------------------------------------------------------------------
# Action normalisation mixin
# ---------------------------------------------------------------------------

class ActionNormalizerMixin:
    """
    Per-(k, dim) z-score normalisation for action targets.

    Buffers `action_mean` / `action_std` [K, action_dim] are registered on the
    head so they persist in checkpoints. Defaults (mean=0, std=1) make the
    normalisation an identity, keeping old checkpoints fully compatible.

    Training : loss is computed in normalised (z) space
               → every DOF / horizon position contributes equally.
    Inference: `denormalize()` maps raw head output back to physical units.

    Bonus: with a zero-initialised output layer, the initial physical
    prediction equals the dataset mean action (gentle forward motion) instead
    of a frozen zero — avoiding the "stationary at start" failure mode.
    """

    def _init_action_stats(self, chunk_size: int, action_dim: int):
        self.register_buffer("action_mean", torch.zeros(chunk_size, action_dim))
        self.register_buffer("action_std", torch.ones(chunk_size, action_dim))

    @torch.no_grad()
    def set_normalization(self, mean, std, std_floor: float = 1e-3):
        mean = torch.as_tensor(mean, dtype=torch.float32).view_as(self.action_mean)
        std = torch.as_tensor(std, dtype=torch.float32).view_as(self.action_std)
        std = std.clamp_min(std_floor)
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    @torch.no_grad()
    def set_quantile_normalization(self, q01, q99, span_floor: float = 1e-3):
        """Map the 1st/99th percentile of each (k, dim) to [-1, +1].

        Reuses the mean/std buffers as (centre, half-span), so checkpoints and
        `denormalize` stay identical — only the fitted constants differ. This is
        the pi0 / FAST / LeRobot default: robust to the heavy tails that make a
        z-score of a forward-dominated distribution allocate most of its range
        to a handful of long-displacement chunks.
        """
        q01 = torch.as_tensor(q01, dtype=torch.float32).view_as(self.action_mean)
        q99 = torch.as_tensor(q99, dtype=torch.float32).view_as(self.action_std)
        centre = (q99 + q01) * 0.5
        half = ((q99 - q01) * 0.5).clamp_min(span_floor)
        self.action_mean.copy_(centre)
        self.action_std.copy_(half)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Physical [.., K, 4] → z-space."""
        return (action - self.action_mean.to(action.dtype)) / self.action_std.to(action.dtype)

    def denormalize(self, action_norm: torch.Tensor) -> torch.Tensor:
        """z-space [.., K, 4] → physical units (m / rad)."""
        return action_norm * self.action_std.to(action_norm.dtype) + self.action_mean.to(action_norm.dtype)

    def has_normalization(self) -> bool:
        return bool(
            (self.action_std != 1.0).any().item() or (self.action_mean != 0.0).any().item()
        )


# ---------------------------------------------------------------------------
# Action Head
# ---------------------------------------------------------------------------

class UAVActionHead(ActionNormalizerMixin, nn.Module):
    """
    Lightweight direct-regression action head for UAV navigation.

    Predicts K future 4-DOF waypoints from Mamba's last hidden state in a
    single forward pass (no autoregressive decoding).

    Args:
        mamba_hidden_size : D_m from the Mamba backbone (default 1024).
        chunk_size        : K future waypoints per prediction (default 5).
        action_dim        : Per-step action dimension (default 4:
                            Δx, Δy, Δz, Δyaw_rad).
    """

    def __init__(
        self,
        mamba_hidden_size: int = 1024,
        chunk_size:        int = 5,
        action_dim:        int = 4,
        hidden_layers:     int = 2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        out_dim = chunk_size * action_dim
        d = mamba_hidden_size

        if hidden_layers == 0:
            # AnoleVLA uses a single linear projection, on the grounds that the
            # recurrent backbone already aggregates the whole sequence into the
            # final token. The cascade below also narrows to d//4, which at
            # large K (K=50 -> out_dim 200) becomes a real bottleneck.
            self.mlp = nn.Sequential(nn.Linear(d, out_dim))
        else:
            # RoboMamba-style dimension cascade: D → D/2 → D/4 → K*4
            self.mlp = nn.Sequential(
                nn.Linear(d,        d // 2),
                nn.SiLU(),
                nn.Linear(d // 2,  d // 4),
                nn.SiLU(),
                nn.Linear(d // 4,  out_dim),
            )
        self._init_action_stats(chunk_size, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the output layer: initial actions near zero for safe deployment
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        global_token: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            global_token : [B, D_m]  — last hidden state from Mamba backbone

        Returns:
            dict with:
                'action' : [B, K, 4]  predicted chunk (Δx, Δy, Δz, Δyaw_rad)
        """
        raw    = self.mlp(global_token)                          # [B, K*4]
        action = raw.view(-1, self.chunk_size, self.action_dim)  # [B, K, 4]
        return {"action": action}


class UAVDynamicsActionHead(ActionNormalizerMixin, nn.Module):
    """
    Dynamics-aware lightweight action chunk head for UAV-Flow.

    Instead of predicting K waypoints independently, this head rolls a compact
    recurrent dynamics state forward for K steps. The recurrent state is
    conditioned on the Mamba global token and the current UAV proprioception,
    which makes smooth velocity/yaw-rate evolution an architectural prior
    rather than relying only on loss regularisation.
    """

    def __init__(
        self,
        mamba_hidden_size: int = 1024,
        chunk_size: int = 5,
        action_dim: int = 4,
        proprio_dim: int = 4,
        hidden_ratio: float = 0.5,
        action_bound: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.action_bound = action_bound
        d = mamba_hidden_size
        h = max(128, int(d * hidden_ratio))

        self.context = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.SiLU(),
        )
        self.proprio_proj = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, h),
            nn.SiLU(),
        )
        self.step_embed = nn.Parameter(torch.randn(chunk_size, h) * 0.02)
        self.cell = nn.GRUCell(input_size=action_dim + h, hidden_size=h)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, action_dim),
        )
        self._init_action_stats(chunk_size, action_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.delta_head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        global_token: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> dict:
        batch = global_token.size(0)
        state = self.context(global_token)
        if proprio is None:
            proprio = global_token.new_zeros(batch, self.proprio_dim)
        prop = self.proprio_proj(proprio.to(dtype=global_token.dtype))
        hidden = state + prop

        prev_action = global_token.new_zeros(batch, self.action_dim)
        actions = []
        for step in range(self.chunk_size):
            step_context = prop + self.step_embed[step].unsqueeze(0)
            cell_input = torch.cat([prev_action, step_context], dim=-1)
            hidden = self.cell(cell_input, hidden)
            delta = self.delta_head(hidden)
            if self.action_bound > 0:
                delta = self.action_bound * torch.tanh(delta / self.action_bound)
            actions.append(delta)
            prev_action = delta

        return {"action": torch.stack(actions, dim=1)}


# Backward compatibility alias (for any code importing the old name)
UAVActionChunkHead = UAVActionHead


# ---------------------------------------------------------------------------
# v5: spatial readout + distributional output
# ---------------------------------------------------------------------------

class ActionReadout(nn.Module):
    """K learned action queries cross-attending the projector's 2D patch tokens.

    Replaces ``hidden[:, -1, :]``. The terminal hidden state is a global summary
    produced by a causal scan; the action needs to know *which patch* holds the
    referred target, and that sparse signal is exactly what a global summary
    averages away (measured: vis_sens 0.07 while the same trunk scores
    Δshuffle +0.918 on Stage-2 language tasks).

    Keys/values are the projector's 2D patches concatenated with the trunk's
    text hidden states, and the queries are purely learned — there is no
    additive context term. That is deliberate, and it is the second thing this
    module got wrong before: injecting the trunk's terminal hidden into the
    query residual reproduced the very bottleneck it was meant to remove.
    Measured on the first v5 smoke, |ctx_term| was 2955 against |queries| 0.82,
    so the attention branch carried 10% of the stream and vision was drowned
    even though the attention itself had become selective (entropy 43% of
    uniform, peak weight 150x uniform).

    With language in the KV instead, both modalities reach the output only
    through the attention, so the softmax — not an unnormalised residual —
    arbitrates between them. This is the arrangement pi0 and GR00T N1 use for
    their action experts, and it is strictly less machinery than a separate
    context path.

    Cost is negligible: queries are K=8 long, so the attention matrix is
    [B, heads, 8, N_kv] rather than [B, heads, N_kv, N_kv].
    """

    def __init__(
        self,
        d_lm: int,
        chunk_size: int,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.queries = nn.Parameter(torch.randn(chunk_size, d_lm) * 0.02)
        self.ln_kv = nn.LayerNorm(d_lm)
        self.ln_q = nn.ModuleList([nn.LayerNorm(d_lm) for _ in range(n_layers)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_lm, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d_lm) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_lm, d_lm * ffn_ratio),
                nn.GELU(),
                nn.Linear(d_lm * ffn_ratio, d_lm),
            )
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_lm)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        kv_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            kv_tokens        : [B, N_kv, D]  [vision patches | text hidden]
            key_padding_mask : [B, N_kv] bool, True at positions to ignore
        Returns:
            [B, K, D] — one readout token per waypoint
        """
        kv = self.ln_kv(kv_tokens)
        q = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        for i in range(self.n_layers):
            q = q + self.attn[i](self.ln_q[i](q), kv, kv, need_weights=False,
                                 key_padding_mask=key_padding_mask)[0]
            q = q + self.ffn[i](self.ln_f[i](q))
        return self.ln_out(q)

    @torch.no_grad()
    def diagnose(
        self,
        kv_tokens: torch.Tensor,
        n_vis: int,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Where the readout actually looks, when the vision gate fails.

        With language and vision competing inside one softmax, a low vis_sens
        has two distinguishable causes that call for different fixes:

          TEXT-DOMINATED — attention mass sits on the text keys, so the policy
                           is an instruction template that ignores the scene.
          UNIFORM        — attention is flat, so the readout is a mean-pool,
                           i.e. the same global summary it was meant to replace.
        """
        kv = self.ln_kv(kv_tokens)
        q = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        rec = {"q_norm": float(q.norm(dim=-1).mean()),
               "uniform_entropy": math.log(kv.size(1)),
               "n_vis": n_vis, "n_kv": kv.size(1), "layers": []}
        for i in range(self.n_layers):
            a_out, a_w = self.attn[i](self.ln_q[i](q), kv, kv, need_weights=True,
                                      average_attn_weights=True,
                                      key_padding_mask=key_padding_mask)
            p = a_w.clamp_min(1e-9)
            rec["layers"].append({
                "res_norm": float(q.norm(dim=-1).mean()),
                "attn_norm": float(a_out.norm(dim=-1).mean()),
                "entropy": float(-(p * p.log()).sum(-1).mean()),
                "max_w": float(a_w.max(dim=-1).values.mean()),
                "vis_mass": float(a_w[..., :n_vis].sum(-1).mean()),
            })
            q = q + a_out
            q = q + self.ffn[i](self.ln_f[i](q))
        return rec

    @staticmethod
    def format_diagnosis(rec: dict) -> str:
        uni = rec["uniform_entropy"]
        vis_prior = rec["n_vis"] / rec["n_kv"]
        lines = [f"    n_vis={rec['n_vis']} n_kv={rec['n_kv']} "
                 f"(uniform would put {vis_prior:.2%} of the mass on vision)"]
        for i, L in enumerate(rec["layers"]):
            share = L["attn_norm"] / max(L["res_norm"], 1e-9)
            lines.append(
                f"    L{i}: vis_mass={L['vis_mass']:6.2%}  "
                f"entropy={L['entropy']:.3f}/{uni:.3f} "
                f"({L['entropy'] / uni:6.2%} of uniform)  "
                f"max_w={L['max_w']:.4f}  attn/res={share:6.2%}")
        L0 = rec["layers"][0]
        if L0["entropy"] / uni > 0.98:
            verdict = ("UNIFORM: attention never sharpened, the readout is a "
                       "mean-pool")
        elif L0["vis_mass"] < 0.5 * vis_prior:
            verdict = ("TEXT-DOMINATED: attention avoids the patches; the "
                       "policy is an instruction template")
        else:
            verdict = ("attention is selective and does reach the patches — "
                       "a low vis_sens is then about the probe or the data, "
                       "not this module")
        lines.append(f"    => {verdict}")
        return "\n".join(lines)


def hl_gauss_targets(
    y: torch.Tensor,          # [...]  target in normalised action space
    edges: torch.Tensor,      # [n_bins + 1]
    sigma: float,
) -> torch.Tensor:
    """Histogram-of-Gaussian soft labels (Farebrother et al., ICML 2024).

    Projects a scalar target onto a categorical distribution over bins by
    integrating N(y, sigma) over each bin. Cross-entropy against these labels
    keeps the ordinal structure of regression while giving a *distributional*
    output, which is what stops the head collapsing to the conditional median.

    Targets are clamped into the representable span first. Without the clamp a
    far-out-of-range target saturates every CDF term, the bin differences are
    all zero, and the renormalised label becomes the zero vector — whose
    cross-entropy gradient is also zero. That would silently drop exactly the
    largest-displacement chunks, i.e. the samples that carry the overshoot
    signal. Clamping instead places their mass in the edge bin, which is the
    same quantile-clipping OpenVLA applies to its bin edges.
    """
    lo, hi = edges[0], edges[-1]
    pad = (hi - lo) / (edges.numel() - 1) * 0.5
    y = y.clamp(lo + pad, hi - pad)
    y = y.unsqueeze(-1).float()
    cdf = 0.5 * (1.0 + torch.erf((edges - y) / (sigma * math.sqrt(2.0))))
    p = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class UAVActionHeadV5(ActionNormalizerMixin, nn.Module):
    """Spatial-readout + optionally distributional action head.

    Two independent switches, so the smoke can A/B each in isolation:

      readout    : K cross-attention queries over vision patches (always on
                   here — this class exists for that) instead of a single
                   terminal hidden state.
      n_bins > 0 : HL-Gauss distributional output. Per (k, dim) logits over
                   `n_bins` bins spanning [-bin_range, +bin_range] in
                   normalised space; the action is decoded as the softmax
                   expectation, so it is bounded by construction and cannot
                   degenerate to a constant the way an L1 point estimate does.
      n_bins = 0 : plain L1 regression from the same readout (ablation arm
                   that isolates the readout change).
    """

    def __init__(
        self,
        mamba_hidden_size: int = 1280,
        chunk_size: int = 8,
        action_dim: int = 4,
        n_bins: int = 0,
        bin_range: float = 1.5,
        hl_sigma_ratio: float = 0.75,
        n_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = int(n_bins)
        self.bin_range = float(bin_range)
        d = mamba_hidden_size

        self.readout = ActionReadout(d, chunk_size, n_heads=n_heads,
                                     n_layers=n_layers)
        out_per_step = action_dim * self.n_bins if self.n_bins > 0 else action_dim
        self.step_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, out_per_step),
        )

        if self.n_bins > 0:
            edges = torch.linspace(-bin_range, bin_range, self.n_bins + 1)
            self.register_buffer("bin_edges", edges)
            self.register_buffer("bin_centers", (edges[:-1] + edges[1:]) * 0.5)
            self.hl_sigma = hl_sigma_ratio * (2.0 * bin_range / self.n_bins)

        self._init_action_stats(chunk_size, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.step_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.step_mlp[-1].bias)
        # Small-scale init for BOTH arms. Pure-zero L1 weights make every sample
        # share the same bias action at step 0 (act_std=0), so the variance-floor
        # gradient w.r.t. bias is identically zero and the head can stay stuck in
        # the constant basin (sim smoke: act_std stayed <1e-2 for 500 steps then
        # aborted). std=1e-3 keeps physical actions near the quantile centre
        # (still deploy-safe) while giving batch-dependent h a non-zero path.
        nn.init.normal_(self.step_mlp[-1].weight, std=1e-3)

    def forward(
        self,
        kv_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        tokens = self.readout(kv_tokens, key_padding_mask)   # [B, K, D]
        raw = self.step_mlp(tokens)                          # [B, K, out_per_step]
        if self.n_bins > 0:
            logits = raw.view(-1, self.chunk_size, self.action_dim, self.n_bins)
            probs = logits.float().softmax(dim=-1)
            action = probs @ self.bin_centers.float()  # [B, K, 4] expectation
            return {"action": action, "logits": logits}
        return {"action": raw.view(-1, self.chunk_size, self.action_dim),
                "logits": None}


def aero_action_loss_v5(
    pred: dict,
    gt_action: torch.Tensor,          # [B, K, 4] physical units
    head,                             # UAVActionHeadV5 (carries norm stats + bins)
    lambda_endpoint: float = 0.5,
    lambda_direction: float = 0.5,
    lambda_var: float = 0.5,
    var_floor: float = 0.3,
    channel_weights: torch.Tensor | None = None,  # [4] for L1 main/endpoint
) -> tuple:
    """Action-chunk loss for the v5 head.

    Fit (HL-Gauss CE or L1) + endpoint + heading, plus the same anti-collapse
    variance floor used by the L1 head. HL-Gauss bounds each sample's action
    but the *expectation* can still converge to an input-independent constant
    across the batch — without the floor, G3/G4/G5 all fail. Per-channel
    weights (z/yaw) apply to the L1 arm so small DOFs are not ignored.
    """
    action = pred["action"]                            # [B, K, 4] normalised
    logits = pred.get("logits")
    gt_norm = head.normalize(gt_action)
    pred_phys = head.denormalize(action)

    if logits is not None:
        tgt = hl_gauss_targets(gt_norm.detach(), head.bin_edges.float(),
                               head.hl_sigma)          # [B, K, 4, n_bins]
        logp = logits.float().log_softmax(dim=-1)
        ce = -(tgt * logp).sum(dim=-1)                 # [B, K, 4]
        if channel_weights is not None:
            w = channel_weights.view(1, 1, -1).to(ce.dtype)
            ce = ce * w
        loss_main = ce.mean()
        loss_endpoint = ce[:, -1].mean()
    else:
        err = (action - gt_norm).abs()
        if channel_weights is not None:
            err = err * channel_weights.view(1, 1, -1).to(err.dtype)
        loss_main = err.mean()
        loss_endpoint = err[:, -1].mean()

    pred_xy = pred_phys[:, -1, :2].float()
    gt_xy = gt_action[:, -1, :2].float()
    valid = (pred_xy.norm(dim=-1) > 1e-4) & (gt_xy.norm(dim=-1) > 1e-4)
    if valid.any():
        cos = F.cosine_similarity(pred_xy[valid], gt_xy[valid], dim=-1)
        loss_direction = (1.0 - cos).mean()
    else:
        loss_direction = action.new_zeros(())

    # Variance floor over the batch in normalised action space (same units as
    # the L1 head's floor). Requires B>=2. Weight z/yaw floors harder when
    # channel_weights are set so G5 small-DOF collapse is directly penalised.
    if lambda_var > 0 and action.size(0) >= 2:
        pred_std = action.float().std(dim=0)           # [K, 4]
        gap = F.relu(var_floor - pred_std)
        if channel_weights is not None:
            gap = gap * channel_weights.view(1, -1).to(gap.dtype)
        loss_var = gap.mean()
    else:
        loss_var = action.new_zeros(())

    total = (loss_main
             + lambda_endpoint * loss_endpoint
             + lambda_direction * loss_direction
             + lambda_var * loss_var)

    with torch.no_grad():
        err_phys = (pred_phys.float() - gt_action.float()).abs()
        detail = {
            "main": loss_main.item(),
            "endpoint": loss_endpoint.item(),
            "direction": loss_direction.item(),
            "pos_err_m": err_phys[..., :3].mean().item(),
            "yaw_err_deg": (err_phys[..., 3].mean() * (180.0 / math.pi)).item(),
            "end_pos_err_m": err_phys[:, -1, :3].mean().item(),
        }
        if lambda_var > 0:
            detail["var"] = float(loss_var.item())
    return total, detail


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def aero_action_loss(
    pred:          dict,
    gt_action:     torch.Tensor,    # [B, K, 4]  physical units (Δm, Δm, Δm, Δyaw_rad)
    head=None,                       # ActionNormalizerMixin head (for z-score stats)
    lambda_smooth: float = 0.0,
    lambda_endpoint: float = 0.25,
    lambda_direction: float = 0.5,
    lambda_acc: float = 0.0,
    lambda_var: float = 0.0,
    var_floor: float = 0.3,
    channel_weights: torch.Tensor | None = None,  # [4] per-DOF weights (z-space main+endpoint)
    sample_weights: torch.Tensor | None = None,   # [B] per-sample weights (main term)
) -> tuple:
    """
    Action-chunk loss in per-(k, dim) z-score space.

    Rationale (UAV-Flow specifics):
      - Raw targets are dominated by forward motion (x ≈ 0.13 m/step) while
        lateral / vertical / yaw components are 10-100× smaller. A physical-
        space loss lets the model win by predicting the forward mean and
        ignoring turns. Normalising each (horizon-position, DOF) to unit
        variance makes every degree of freedom contribute equally.
      - Pure L1 (not Smooth-L1): Smooth-L1's quadratic zone (|e|<beta)
        underweights exactly the small-magnitude errors that matter after
        normalisation; L1 keeps constant gradient toward the target.

    Components:
      main      : L1 in z-space over all K×4 outputs.
      endpoint  : extra L1 on the final waypoint (goal accuracy).
      direction : 1−cos between predicted and GT endpoint xy displacement in
                  *physical* space (heading correctness for turn commands).
      smooth/acc: optional temporal regularisers (z-space).

    `pred["action"]` is interpreted as z-space output when `head` carries
    normalisation stats; otherwise the loss operates in physical space
    (backward compatible with old checkpoints / no-stats runs).

    AeroStream Round-A weighting (change plan §A3):
      channel_weights : [4] per-DOF weights applied to the z-space main and
                        endpoint terms (e.g. [1,1,2.5,2.5] boosts dz/dyaw,
                        whose std is ~1/8 of dx — the collapsed channels).
      sample_weights  : [B] per-sample weights for the main term, from
                        1 + log1p(endpoint displacement in metres): large-
                        displacement samples get more gradient, countering
                        the regression-to-small-mean collapse.

    Returns:
        (total_loss, detail_dict)
    """
    pred_action = pred["action"]    # [B, K, 4]  (z-space if head has stats)

    if head is not None and hasattr(head, "normalize"):
        gt_norm = head.normalize(gt_action)
        pred_phys = head.denormalize(pred_action)
    else:
        gt_norm = gt_action
        pred_phys = pred_action

    if channel_weights is not None:
        cw = channel_weights.to(device=pred_action.device, dtype=pred_action.dtype)
        cw = cw / cw.mean().clamp_min(1e-6)   # keep the loss scale comparable
    else:
        cw = None

    # Main: pure L1 in normalised space (optionally channel/sample weighted)
    err_main = (pred_action - gt_norm).abs()          # [B, K, 4]
    if cw is not None:
        err_main = err_main * cw
    if sample_weights is not None:
        sw = sample_weights.to(device=err_main.device, dtype=err_main.dtype)
        sw = sw / sw.mean().clamp_min(1e-6)           # normalise batch mass
        err_main = err_main * sw.view(-1, 1, 1)
    loss_main = err_main.mean()

    # Endpoint: extra weight on the final waypoint
    err_end = (pred_action[:, -1] - gt_norm[:, -1]).abs()
    if cw is not None:
        err_end = err_end * cw
    loss_endpoint = err_end.mean()

    # Direction: endpoint xy heading in physical space
    pred_xy = pred_phys[:, -1, :2].float()
    gt_xy = gt_action[:, -1, :2].float()
    valid_direction = (pred_xy.norm(dim=-1) > 1e-4) & (gt_xy.norm(dim=-1) > 1e-4)
    if valid_direction.any():
        cosine = F.cosine_similarity(pred_xy[valid_direction], gt_xy[valid_direction], dim=-1)
        loss_direction = (1.0 - cosine).mean()
    else:
        loss_direction = pred_action.new_zeros(())

    # Smoothness: penalise jerk (2nd-order finite difference along K dimension)
    if lambda_smooth > 0 and pred_action.size(1) >= 3:
        d1 = pred_action[:, 1:] - pred_action[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_smooth = d2.pow(2).mean()
    else:
        loss_smooth = pred_action.new_zeros(())

    if lambda_acc > 0 and pred_action.size(1) >= 2:
        pred_delta = pred_action[:, 1:] - pred_action[:, :-1]
        gt_delta = gt_norm[:, 1:] - gt_norm[:, :-1]
        loss_acc = F.l1_loss(pred_delta, gt_delta)
    else:
        loss_acc = pred_action.new_zeros(())

    # Anti-collapse variance floor: a constant/mean-only head has zero batch
    # variance per (k, dim) yet minimises the L1 main loss on a forward-biased
    # dataset — the degenerate the pose/velocity-free policy fell into (act_std
    # → 0, instr/vis sensitivity → 0). Penalise per-(k,dim) batch std below a
    # floor so the head MUST keep input-dependent variance; main+direction then
    # force that variance to be GT-aligned (noise would raise those terms).
    if lambda_var > 0 and pred_action.size(0) >= 2:
        pred_std = pred_action.float().std(dim=0)            # [K, 4] over batch
        loss_var = F.relu(var_floor - pred_std).mean()
    else:
        loss_var = pred_action.new_zeros(())

    total = (
        loss_main
        + lambda_smooth * loss_smooth
        + lambda_endpoint * loss_endpoint
        + lambda_direction * loss_direction
        + lambda_acc * loss_acc
        + lambda_var * loss_var
    )

    # Physical-unit diagnostics (no grad)
    with torch.no_grad():
        err_phys = (pred_phys.float() - gt_action.float()).abs()
        pos_err_m = err_phys[..., :3].mean()
        yaw_err_deg = err_phys[..., 3].mean() * (180.0 / math.pi)
        end_pos_err_m = err_phys[:, -1, :3].mean()

    detail = {
        "main": loss_main.item(),
        "endpoint": loss_endpoint.item(),
        "direction": loss_direction.item(),
        "pos_err_m": pos_err_m.item(),
        "yaw_err_deg": yaw_err_deg.item(),
        "end_pos_err_m": end_pos_err_m.item(),
    }
    if lambda_smooth > 0:
        detail["smooth"] = loss_smooth.item()
    if lambda_acc > 0:
        detail["acc"] = loss_acc.item()
    if lambda_var > 0:
        detail["var"] = loss_var.item()
    return total, detail


# ---------------------------------------------------------------------------
# Temporal Ensemble (inference utility)
# ---------------------------------------------------------------------------

class TemporalEnsemble:
    """
    Aggregates action-chunk predictions over a sliding window using
    exponentially decaying weights to produce smooth executed actions.

    Reference: ACT (Zhao et al. 2023) — temporal ensemble strategy.

    Usage:
        ens = TemporalEnsemble(window=5, decay=0.7)
        for each control step:
            chunk = model(...)["action"][0]  # [K, 4]
            action = ens.update(chunk)       # [4]  ← execute this
    """

    def __init__(self, window: int = 5, decay: float = 0.7):
        self.window = window
        self.decay  = decay
        self.buffer: deque = deque(maxlen=window)

    def update(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            new_chunk : [K, 4]  model prediction for current step
        Returns:
            action    : [4]     exponentially weighted average action
        """
        self.buffer.append(new_chunk)
        weights = [self.decay ** (len(self.buffer) - 1 - i)
                   for i in range(len(self.buffer))]
        w_sum   = sum(weights)
        act_dim = new_chunk.shape[-1]
        action_sum = torch.zeros(act_dim, device=new_chunk.device,
                                  dtype=new_chunk.dtype)
        for j, (chunk, w) in enumerate(zip(self.buffer, weights)):
            # Each buffered chunk[i] predicted steps from time i forwards;
            # to get the estimate for *current* step, we index into the chunk.
            idx = len(self.buffer) - 1 - j
            if idx < chunk.shape[0]:
                action_sum += w * chunk[idx]
        return action_sum / w_sum

    def reset(self):
        self.buffer.clear()
