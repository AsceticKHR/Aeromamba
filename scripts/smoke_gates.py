"""AeroMamba gated smoke-test pipeline (pre-training validation).

Runs an ordered series of cheap, high-signal gates BEFORE any full training
run. Modeled on standard deep-learning engineering practice (init-loss
sanity, single-batch overfit, input ablation, baseline comparison,
checkpoint round-trip, throughput budgeting), with AeroMamba-specific
audits derived from v1 post-mortems (flip/text consistency, channel-wise
action errors, instruction sensitivity).

Gates (in order; later gates reuse earlier artifacts):

  G0 data-audit        real-data integrity + flip/direction-word consistency
  G1 init-sanity       finite loss at init; zero-init action head outputs ~0
  G2 grad-flow         every trainable param gets a finite grad; frozen get none
  G3 overfit-batch     memorize ONE batch to ~zero loss (wiring test)
  G4 overfit-subset    learn 256 real samples; beat predict-zero / predict-mean
  G5 input-ablation    shuffle instruction / blank image / zero proprio must hurt
  G6 ckpt-roundtrip    save -> load -> bitwise-identical forward
  G7 throughput        samples/s + peak VRAM + dataloader-only rate at target bs

Exit code = number of failed HARD gates (WARN does not fail the run).
A JSON report is written to --report (default reports/smoke_report.json).

Usage:
  # offline structural check (no data / weights needed)
  AEROMAMBA_OFFLINE=1 python scripts/smoke_gates.py --dummy

  # full pre-flight on remote before a real run
  python scripts/smoke_gates.py --data_root /root/autodl-tmp/datasets/uav-flow \
      --mamba_type mamba-2-370m --vision_type siglip2_base_384 \
      --chunk_size 8 --target_batch 48

  # single gate
  python scripts/smoke_gates.py --dummy --gates G3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import DummyUAVDataset, UAVFlowDataset, aero_collate_fn  # noqa: E402
from model.uav_mamba_vla import AeroMambaVLA  # noqa: E402


# ─────────────────────────────────────────────────────────────── helpers
def bfloat_safe_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def forward_loss(model, batch, device, proprio_mode="full", binding_lambda=0.0, **kw):
    b = batch_to_device(batch, device)
    state = b.get("state8")
    delta = b.get("delta_state8")
    if proprio_mode == "pose_only":
        # official OpenVLA-UAV protocol: pose relative to first frame only,
        # no velocity feedback (v1 OOD positive-feedback loop)
        if state is not None:
            state = state.clone()
            state[..., 4:] = 0.0
        if delta is not None:
            delta = torch.zeros_like(delta)
    extra = {}
    if binding_lambda > 0.0 and "binding_labels" in b:
        extra = {"binding_labels": b["binding_labels"], "lambda_binding": binding_lambda}
    return model(
        b["pixel_values"], b["input_ids"],
        proprio=b.get("proprio"),
        state=state,
        delta_state=delta,
        gt_action=b.get("gt_action"),
        return_loss=True, **extra, **kw,
    )


def pred_action(out: dict) -> torch.Tensor | None:
    for key in ("action", "actions", "pred_action", "action_pred"):
        v = out.get(key)
        if isinstance(v, torch.Tensor) and v.dim() == 3:
            return v
    for v in out.values():
        if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[-1] == 4:
            return v
    return None


def fmt(x) -> str:
    return f"{x:.5f}" if isinstance(x, float) else str(x)


def unfreeze_vision_top(model: AeroMambaVLA, n_layers: int) -> int:
    """Unfreeze the top n transformer blocks of the vision tower.

    Note: VisionEncoder.forward taps hidden_states[-2], so n_layers must be
    >= 2 for the unfrozen blocks to influence the features actually used.
    Returns the number of parameters unfrozen.
    """
    bb = model.vision_encoder.backbone
    vm = getattr(bb, "vision_model", bb)
    enc = getattr(vm, "encoder", None)
    layers = None
    if enc is not None:
        layers = getattr(enc, "layers", None) or getattr(enc, "blocks", None)
    if layers is None:
        layers = getattr(vm, "blocks", None)  # timm ViT
    if layers is None:
        raise RuntimeError("cannot locate vision encoder transformer blocks")
    n_params = 0
    for blk in list(layers)[-n_layers:]:
        for p in blk.parameters():
            p.requires_grad = True
            n_params += p.numel()
    return n_params


class GateResult:
    def __init__(self, name, status, metrics=None, note=""):
        self.name, self.status, self.metrics, self.note = name, status, metrics or {}, note

    def row(self):
        icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}[self.status]
        ms = "  ".join(f"{k}={fmt(v)}" for k, v in self.metrics.items())
        return f"[{icon}] {self.name:<16} {ms}  {self.note}"


# ─────────────────────────────────────────────────────────────── context
class Ctx:
    """Lazy shared objects so single-gate runs stay cheap."""

    def __init__(self, args):
        self.args = args
        self.device = bfloat_safe_device()
        self._model = None
        self._ds = None
        self.trained_model = None  # produced by G4, consumed by G5

    def fl(self, model, batch, **kw):
        """forward_loss with the experiment's ablation settings applied."""
        return forward_loss(model, batch, self.device,
                            proprio_mode=self.args.proprio_mode,
                            binding_lambda=self.args.lambda_binding, **kw)

    @property
    def model(self) -> AeroMambaVLA:
        if self._model is None:
            self._model = self.fresh_model()
        return self._model

    def fresh_model(self) -> AeroMambaVLA:
        m = AeroMambaVLA(
            mamba_type=self.args.mamba_type,
            vision_type=self.args.vision_type,
            chunk_size=self.args.chunk_size,
        ).to(self.device)
        m.configure_stage3(train_lora=self.args.train_lora)
        if self.args.unfreeze_vision_layers > 0:
            n = unfreeze_vision_top(m, self.args.unfreeze_vision_layers)
            print(f"[ctx] unfroze top {self.args.unfreeze_vision_layers} vision blocks "
                  f"({n / 1e6:.1f}M params)")
        return m

    def dataset(self, aug_flip: bool = False):
        if self.args.dummy or not self.args.data_root:
            dual = self.args.vision_type.startswith("dino") and "siglip" in self.args.vision_type
            return DummyUAVDataset(size=1024, chunk_size=self.args.chunk_size,
                                   max_text_len=self.args.max_text_len, dual_vision=dual)
        return UAVFlowDataset(
            data_root=self.args.data_root,
            tokenizer=self.model.tokenizer,
            transform=self.model.vision_encoder.transform,
            chunk_size=self.args.chunk_size,
            max_text_len=self.args.max_text_len,
            pos_scale=self.args.pos_scale,
            aug_flip=aug_flip,
            emit_binding_labels=self.args.lambda_binding > 0.0,
        )

    @property
    def ds(self):
        if self._ds is None:
            self._ds = self.dataset(aug_flip=False)
        return self._ds

    def loader(self, ds, bs, shuffle=False, workers=0):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          collate_fn=aero_collate_fn, num_workers=workers)

    def one_batch(self, bs=8, seed=0):
        g = random.Random(seed)
        idx = [g.randrange(len(self.ds)) for _ in range(bs)]
        return aero_collate_fn([self.ds[i] for i in idx])


# ─────────────────────────────────────────────────────────────── gates
DIR_RE = re.compile(r"\b(left|right)\b", re.I)


def g0_data_audit(ctx: Ctx) -> GateResult:
    if ctx.args.dummy or not ctx.args.data_root:
        return GateResult("G0 data-audit", "SKIP", note="dummy mode (no real data)")
    n_probe = min(256, len(ctx.ds))
    g = random.Random(0)
    idxs = [g.randrange(len(ctx.ds)) for _ in range(n_probe)]
    bad_finite = 0
    act_absmax = 0.0
    ids_sigs = set()
    for i in idxs:
        s = ctx.ds[i]
        for k in ("pixel_values", "proprio", "state8", "delta_state8", "gt_action"):
            v = s.get(k)
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                bad_finite += 1
        act_absmax = max(act_absmax, float(s["gt_action"].abs().max()))
        ids_sigs.add(tuple(s["input_ids"][:8].tolist()))

    # flip/direction-word consistency: correlation between "left/right" in the
    # (post-augmentation) instruction and the sign of lateral displacement must
    # NOT be destroyed by augmentation (v1 incident: aug_flip zeroed it).
    # We decode input_ids so the check sees exactly what the model sees.
    tok = ctx.model.tokenizer

    def dir_corr(ds) -> tuple[float, int]:
        if not hasattr(tok, "decode"):
            return float("nan"), 0
        agree = tot = 0
        g2 = random.Random(1)
        for _ in range(2000):
            i = g2.randrange(len(ds))
            s = ds[i]
            try:
                instr = tok.decode([t for t in s["input_ids"].tolist() if t > 0])
            except Exception:
                return float("nan"), 0
            m = DIR_RE.search(instr)
            if not m:
                continue
            dy = float(s["gt_action"][:, 1].sum())
            if abs(dy) < 1e-3:
                continue
            tot += 1
            if (m.group(1).lower() == "right") == (dy > 0):
                agree += 1
        return (agree / tot if tot else float("nan")), tot

    corr_off, n_off = dir_corr(ctx.ds)
    ds_flip = ctx.dataset(aug_flip=True)
    corr_on, n_on = dir_corr(ds_flip)

    metrics = {"n_probe": n_probe, "non_finite": bad_finite,
               "act_absmax": act_absmax, "uniq_instr_sig": len(ids_sigs)}
    note = ""
    status = "PASS"
    if bad_finite > 0 or act_absmax > 50.0:
        status = "FAIL"
        note = "non-finite tensors or wildly out-of-range actions"
    if not math.isnan(corr_off) and n_off >= 30:
        metrics.update({"dir_agree_noflip": round(corr_off, 3),
                        "dir_agree_flip": round(corr_on, 3),
                        "dir_n_noflip": n_off, "dir_n_flip": n_on})
        # with flip ON, agreement collapsing to ~0.5 while OFF is far from 0.5
        # means augmentation destroys language direction semantics
        if abs(corr_off - 0.5) > 0.1 and abs(corr_on - 0.5) < 0.05:
            status = "FAIL"
            note = "aug_flip destroys direction-word/label correlation (v1 bug)"
    else:
        note += " (tokenizer not decodable or too few direction samples)"
    return GateResult("G0 data-audit", status, metrics, note)


def g1_init_sanity(ctx: Ctx) -> GateResult:
    model = ctx.model
    model.eval()
    batch = ctx.one_batch(bs=ctx.args.batch)
    with torch.no_grad():
        out = ctx.fl(model, batch)
    loss = float(out["loss"])
    pa = pred_action(out)
    pred_mag = float(pa.abs().mean()) if pa is not None else float("nan")
    gt_mag = float(batch["gt_action"].abs().mean())
    metrics = {"init_loss": loss, "pred_abs_mean": pred_mag, "gt_abs_mean": gt_mag}
    if not math.isfinite(loss):
        return GateResult("G1 init-sanity", "FAIL", metrics, "non-finite init loss")
    if not math.isnan(pred_mag) and pred_mag > 1e-2:
        return GateResult("G1 init-sanity", "WARN", metrics,
                          "action head not near-zero at init (zero-init expected)")
    return GateResult("G1 init-sanity", "PASS", metrics)


def g2_grad_flow(ctx: Ctx) -> GateResult:
    model = ctx.model
    model.train()
    model.zero_grad(set_to_none=True)
    out = ctx.fl(model, ctx.one_batch(bs=4))
    out["loss"].backward()
    no_grad, nan_grad, frozen_grad = [], [], []
    gnorm_sq = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad.append(n)
            elif not torch.isfinite(p.grad).all():
                nan_grad.append(n)
            else:
                gnorm_sq += float(p.grad.norm()) ** 2
        elif p.grad is not None and p.grad.abs().sum() > 0:
            frozen_grad.append(n)
    model.zero_grad(set_to_none=True)
    metrics = {"missing_grad": len(no_grad), "nan_grad": len(nan_grad),
               "frozen_with_grad": len(frozen_grad), "grad_norm": math.sqrt(gnorm_sq)}
    if no_grad or nan_grad or frozen_grad:
        detail = (no_grad + nan_grad + frozen_grad)[:5]
        return GateResult("G2 grad-flow", "FAIL", metrics, f"e.g. {detail}")
    return GateResult("G2 grad-flow", "PASS", metrics)


def g3_overfit_batch(ctx: Ctx) -> GateResult:
    model = ctx.fresh_model()
    model.train()
    batch = ctx.one_batch(bs=8, seed=1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=0.0)
    init = first = None
    steps = ctx.args.overfit_steps
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = ctx.fl(model, batch)
        loss = out["loss"]
        loss.backward()
        opt.step()
        if init is None:
            init = first = float(loss)
        last = float(loss)
        if last < max(0.02, 0.02 * init):
            break
    metrics = {"init_loss": init, "final_loss": last, "steps": step + 1}
    if last < max(0.02, 0.02 * init):
        return GateResult("G3 overfit-batch", "PASS", metrics)
    if last < 0.10 * init:
        return GateResult("G3 overfit-batch", "WARN", metrics, "slow but converging")
    return GateResult("G3 overfit-batch", "FAIL", metrics,
                      "cannot memorize one batch: wiring/loss/optimizer bug")


def g4_overfit_subset(ctx: Ctx) -> GateResult:
    """Fixed OPTIMIZER-STEP budget (fair across batch sizes / subset sizes)."""
    n = min(ctx.args.subset_size, len(ctx.ds))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(ctx.ds), generator=g)[:n].tolist()
    n_val = max(8, n // 10)
    train_ds, val_ds = Subset(ctx.ds, perm[n_val:]), Subset(ctx.ds, perm[:n_val])
    model = ctx.fresh_model()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=ctx.args.subset_lr, weight_decay=0.0)
    tl = ctx.loader(train_ds, ctx.args.batch, shuffle=True)
    vl = ctx.loader(val_ds, ctx.args.batch)

    def eval_loss(m):
        m.eval()
        tot = cnt = 0.0
        zero = mean = 0.0
        with torch.no_grad():
            for b in vl:
                out = ctx.fl(m, b)
                tot += float(out["loss"]) * len(b["gt_action"])
                cnt += len(b["gt_action"])
                gt = b["gt_action"]
                zero += float(gt.abs().mean()) * len(gt)          # predict-zero L1
                mean += float((gt - gt.mean(dim=(0, 1), keepdim=True)).abs().mean()) * len(gt)
        return tot / cnt, zero / cnt, mean / cnt

    steps_budget = ctx.args.subset_steps
    first_train = last_train = None
    window = []
    step = 0
    model.train()
    while step < steps_budget:
        for b in tl:
            opt.zero_grad(set_to_none=True)
            out = ctx.fl(model, b)
            out["loss"].backward()
            opt.step()
            step += 1
            window.append(float(out["loss"]))
            if len(window) > 20:
                window.pop(0)
            if first_train is None and len(window) == 20:
                first_train = sum(window) / len(window)
            if step >= steps_budget:
                break
    last_train = sum(window) / len(window)
    if first_train is None:
        first_train = last_train
    val, zero_base, mean_base = eval_loss(model)
    ctx.trained_model = model  # for G5

    red = (first_train - last_train) / max(first_train, 1e-9)
    metrics = {"n": n, "steps": step, "train_first20": first_train,
               "train_last20": last_train, "train_reduction": round(red, 3),
               "val": val, "baseline_zero": zero_base, "baseline_mean": mean_base}
    if red < 0.5:
        return GateResult("G4 overfit-subset", "FAIL", metrics,
                          "train loss barely drops on tiny subset")
    if val >= zero_base:
        return GateResult("G4 overfit-subset", "FAIL", metrics,
                          "val worse than predict-zero baseline: no real signal learned")
    if val >= mean_base:
        return GateResult("G4 overfit-subset", "WARN", metrics,
                          "beats zero but not mean baseline")
    return GateResult("G4 overfit-subset", "PASS", metrics)


def g5_input_ablation(ctx: Ctx) -> GateResult:
    model = ctx.trained_model
    if model is None:
        return GateResult("G5 input-ablation", "SKIP", note="run G4 first (same invocation)")
    model.eval()
    batch = ctx.one_batch(bs=min(32, ctx.args.batch * 2), seed=2)

    def loss_with(mutate):
        """Action-loss only ('main' if exposed) so aux binding loss cannot
        inflate the sensitivity numbers."""
        b = copy.deepcopy(batch)
        mutate(b)
        with torch.no_grad():
            out = ctx.fl(model, b)
        detail = out.get("loss_detail") or {}
        return float(detail.get("main", out["loss"]))

    base = loss_with(lambda b: None)
    l_instr = loss_with(lambda b: b.update(input_ids=torch.roll(b["input_ids"], 1, dims=0)))
    def blank_img(b):
        pv = b["pixel_values"]
        if isinstance(pv, dict):
            for k in pv:
                pv[k] = torch.zeros_like(pv[k])
        else:
            b["pixel_values"] = torch.zeros_like(pv)
    l_img = loss_with(blank_img)
    def zero_prop(b):
        for k in ("proprio", "state8", "delta_state8"):
            if k in b and isinstance(b[k], torch.Tensor):
                b[k] = torch.zeros_like(b[k])
    l_prop = loss_with(zero_prop)

    d_instr = (l_instr - base) / max(base, 1e-9)
    d_img = (l_img - base) / max(base, 1e-9)
    d_prop = (l_prop - base) / max(base, 1e-9)
    metrics = {"base": base, "instr_shuffle_pct": round(100 * d_instr, 1),
               "image_blank_pct": round(100 * d_img, 1),
               "proprio_zero_pct": round(100 * d_prop, 1)}
    note = ""
    status = "PASS"
    if d_instr < ctx.args.instr_ablation_min:
        status = "FAIL"
        note = "instruction not used (shortcut learning) — v1 failure mode"
    elif d_img < 0.05:
        status = "WARN"
        note = "vision barely used"
    return GateResult("G5 input-ablation", status, metrics, note)


def g6_ckpt_roundtrip(ctx: Ctx) -> GateResult:
    model = ctx.trained_model or ctx.model
    model.eval()
    batch = ctx.one_batch(bs=4, seed=3)
    with torch.no_grad():
        out1 = ctx.fl(model, batch)
    tmp = ROOT / "checkpoints" / "_smoke_roundtrip.pth"
    tmp.parent.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict()}, tmp)
    m2 = ctx.fresh_model()
    missing, unexpected = m2.load_state_dict(
        torch.load(tmp, map_location=ctx.device, weights_only=False)["model_state"], strict=False)
    m2.eval()
    with torch.no_grad():
        out2 = ctx.fl(m2, batch)
    tmp.unlink(missing_ok=True)
    diff = abs(float(out1["loss"]) - float(out2["loss"]))
    metrics = {"loss_diff": diff, "missing_keys": len(missing), "unexpected_keys": len(unexpected)}
    if diff > 1e-4 or unexpected:
        return GateResult("G6 ckpt-roundtrip", "FAIL", metrics, "reload changes outputs")
    return GateResult("G6 ckpt-roundtrip", "PASS", metrics)


def g7_throughput(ctx: Ctx) -> GateResult:
    model = ctx.model
    model.train()
    bs = ctx.args.target_batch
    # dataloader-only rate (I/O ceiling; v1 hit a 16-worker disk collapse)
    dl = ctx.loader(ctx.ds, bs, shuffle=True, workers=ctx.args.workers)
    it = iter(dl)
    n_io = min(20, max(2, len(ctx.ds) // bs - 1))
    t0 = time.perf_counter()
    for _ in range(n_io):
        next(it)
    io_rate = n_io * bs / (time.perf_counter() - t0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch = ctx.one_batch(bs=bs, seed=4)
    for _ in range(2):  # warmup
        opt.zero_grad(set_to_none=True)
        ctx.fl(model, batch)["loss"].backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 8
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        ctx.fl(model, batch)["loss"].backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_rate = n_steps * bs / (time.perf_counter() - t0)
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    epoch_h = len(ctx.ds) / max(step_rate, 1e-9) / 3600
    metrics = {"gpu_samples_s": round(step_rate, 1), "io_samples_s": round(io_rate, 1),
               "peak_vram_gb": round(vram, 2), "est_epoch_h": round(epoch_h, 2),
               "batch": bs, "workers": ctx.args.workers}
    if io_rate < step_rate:
        return GateResult("G7 throughput", "WARN", metrics,
                          "dataloader slower than GPU: I/O will starve training (v1 incident)")
    return GateResult("G7 throughput", "PASS", metrics)


GATES = {
    "G0": g0_data_audit,
    "G1": g1_init_sanity,
    "G2": g2_grad_flow,
    "G3": g3_overfit_batch,
    "G4": g4_overfit_subset,
    "G5": g5_input_ablation,
    "G6": g6_ckpt_roundtrip,
    "G7": g7_throughput,
}
HARD = {"G0", "G1", "G2", "G3", "G4", "G6"}  # G5/G7 default to advisory pre-S3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--mamba_type", default="mamba-130m")
    ap.add_argument("--vision_type", default="siglip_so_384")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--train_lora", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--target_batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--subset_size", type=int, default=256)
    ap.add_argument("--subset_steps", type=int, default=1500,
                    help="fixed optimizer-step budget for G4 (fair metric)")
    ap.add_argument("--subset_lr", type=float, default=1e-3)
    ap.add_argument("--proprio_mode", choices=["full", "pose_only"], default="full",
                    help="pose_only zeroes velocity channels (official-protocol test)")
    ap.add_argument("--lambda_binding", type=float, default=0.0,
                    help=">0 enables binding labels + auxiliary loss in G4")
    ap.add_argument("--unfreeze_vision_layers", type=int, default=0,
                    help="unfreeze top N vision blocks (>=2 to affect the tapped layer)")
    ap.add_argument("--instr_ablation_min", type=float, default=0.03,
                    help="min relative loss increase when instructions shuffled")
    ap.add_argument("--gates", default="all", help="comma list, e.g. G1,G3")
    ap.add_argument("--strict", action="store_true",
                    help="treat G5/G7 as hard gates too")
    ap.add_argument("--report", default=str(ROOT / "reports" / "smoke_report.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    wanted = list(GATES) if args.gates == "all" else [g.strip().upper() for g in args.gates.split(",")]
    hard = HARD | ({"G5", "G7"} if args.strict else set())

    ctx = Ctx(args)
    results: list[GateResult] = []
    print(f"\n{'=' * 72}\nAeroMamba smoke gates  |  device={ctx.device}  "
          f"model={args.mamba_type}+{args.vision_type}  data={'dummy' if args.dummy or not args.data_root else args.data_root}\n{'=' * 72}")
    for name in wanted:
        fn = GATES.get(name)
        if fn is None:
            print(f"[SKIP] unknown gate {name}")
            continue
        t0 = time.perf_counter()
        try:
            r = fn(ctx)
        except Exception as e:  # a crashing gate is a failing gate
            r = GateResult(f"{name} {fn.__name__}", "FAIL", note=f"exception: {e}")
        r.metrics["sec"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(r.row(), flush=True)
        if r.status == "FAIL" and name in hard and name in ("G1", "G2", "G3"):
            print(">> hard wiring gate failed — aborting remaining gates (fix first).")
            break

    n_fail = sum(1 for r in results if r.status == "FAIL" and r.name.split()[0] in hard)
    n_warn = sum(1 for r in results if r.status == "WARN")
    print(f"\n{'=' * 72}\nRESULT: {len(results)} gates  |  hard failures={n_fail}  warnings={n_warn}")
    verdict = "GO — safe to launch pilot run" if n_fail == 0 else "NO-GO — fix hard failures before training"
    print(f"VERDICT: {verdict}\n{'=' * 72}")

    report = {
        "verdict": verdict, "hard_failures": n_fail, "warnings": n_warn,
        "config": vars(args),
        "gates": [{"name": r.name, "status": r.status, "metrics": r.metrics, "note": r.note}
                  for r in results],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
