"""Systematic rigorous QC for a trained Stage-3 (AeroV2) action head.

Evaluates the frozen-backbone + proprio + action-head policy on REAL UAV-Flow
trajectories, broken down by motion class, so we can see WHERE it is good/weak
rather than trusting a single averaged waypoint error.

Metrics (physical units; z-space denormalised via the head's action stats):
  - pos_err_m           : mean |Δ| over all K waypoints, xyz (metres)
  - end_pos_err_m       : endpoint (goal) xyz error (metres)
  - yaw_err_deg         : mean |Δyaw| (degrees)
  - direction_cos       : cos between predicted & GT endpoint xy displacement
  - yaw_sign_acc        : does the model turn the correct way (left/right/straight)
  - dz_sign_acc         : does it climb/descend/hold correctly
  - channel MAE         : per-DOF dx/dy/dz/dyaw error (spot small-channel collapse)
  - pred vs GT std      : per-channel endpoint std ratio (collapse to mean ⇒ ~0)

Broken down by motion class computed from the GT endpoint:
  turn (|Δyaw|>10°) / vertical (|Δz|>0.3m, non-turn) / straight (else).

Rigorous QC gates (hard):
  Q1 overall pos_err_m   < 0.12
  Q2 turn direction      : turn-class yaw_sign_acc > 0.70 AND direction_cos > 0.50
  Q3 no channel collapse : endpoint dz & dyaw pred-std > 0.25 x GT-std
  Q4 global not collapsed: overall action std (z-space) > 1e-2

NOTE on holdout: Stage-3 training used a per-sample random split, so ~97% of
every trajectory's frames were seen. These numbers are therefore an IN-SAMPLE
per-class DIAGNOSTIC (channel collapse / turn direction / sign accuracy), not a
generalisation claim — closed-loop simulator rollout is the true generalisation
test.

Pass --holdout_trajectories to score only the trajectories the checkpoint never
trained on. This requires the checkpoint to have been produced with
--split_by trajectory, which records the held-out set in 'val_traj_files'.

Usage:
  python scripts/eval_s3_systematic.py \
    --s2_ckpt_dir checkpoints/v2_stage2_full_cradio --s2_tag best \
    --s3_ckpt checkpoints/v2_stage3_cradio/best.pth \
    --vision_type cradio_v3_b \
    --data_root /root/autodl-tmp/datasets/stage3_uavflow \
    --action_stats /root/autodl-tmp/datasets/uav-flow/action_stats_k8.json \
    --n_eval 4000 --batch 16 --out reports/s3_systematic_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import UAVFlowDataset, aero_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402
from training.v2_stage3_action import load_s2, load_action_stats  # noqa: E402

TURN_DEG = 10.0        # |Δyaw endpoint| above this ⇒ "turn"
VERT_M = 0.3           # |Δz endpoint| above this (non-turn) ⇒ "vertical"
SIGN_YAW_DEG = 3.0     # deadband for left/right/straight sign
SIGN_DZ_M = 0.15       # deadband for up/down/hold sign


def load_s3(model: AeroV2, payload: dict) -> dict:
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.action_head.load_state_dict(payload["action_head"], strict=True)
    if "action_query" in payload and hasattr(model, "action_query"):
        with torch.no_grad():
            model.action_query.copy_(payload["action_query"].to(model.action_query))
    if "target_encoder" in payload and hasattr(model, "target_encoder"):
        model.target_encoder.load_state_dict(payload["target_encoder"], strict=True)
    # CRITICAL: restore the S3-trained LoRA. Stage-3 (`--train_lora`) updates the
    # LoRA trunk that the action head was fitted on top of; load_s2 only restores
    # the S2 LoRA, so without this the head is fed OUT-OF-DISTRIBUTION features
    # (produced a near-constant, reversed endpoint: dcos -0.53, pred dx<0 vs gt>0).
    if "lora" in payload:
        named = dict(model.lm.named_parameters())
        loaded, missing = 0, 0
        with torch.no_grad():
            for n, p in payload["lora"].items():
                if n in named:
                    named[n].copy_(p.to(named[n]))
                    loaded += 1
                else:
                    missing += 1
        print(f"[S3] restored S3 LoRA: {loaded} tensors ({missing} unmatched)",
              flush=True)
    return {"step": payload.get("step"), "val": payload.get("val"),
            "chunk_size": payload.get("chunk_size")}


def _sign(v, dead):
    if v > dead:
        return 1
    if v < -dead:
        return 2
    return 0


def _motion_class(dyaw_deg, dz):
    if abs(dyaw_deg) > TURN_DEG:
        return "turn"
    if abs(dz) > VERT_M:
        return "vertical"
    return "straight"


@torch.no_grad()
def evaluate(model, loader, device, proprio_dim=4):
    model.eval()
    # accumulators keyed by group ("all", "turn", "vertical", "straight")
    rows = defaultdict(list)   # group -> list of per-sample metric dicts
    ep_pred = defaultdict(list)  # group -> [K? no, endpoint] pred endpoint [4]
    ep_gt = defaultdict(list)
    act_z_all = []             # z-space predicted actions for global collapse
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        # slice state8 -> proprio_dim (ProprioEncoder is sized to proprio_dim even
        # in proprio-free mode, where forward_action then ignores the token)
        proprio = batch["state8"].to(device).float()[:, :proprio_dim]
        gt = batch["gt_action"].to(device).float()          # [B,K,4] physical
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_action(pv, ids, proprio)     # no loss
        act_z = out["action"].float()                        # [B,K,4] z-space
        pred = model.action_head.denormalize(act_z)          # [B,K,4] physical
        act_z_all.append(act_z.cpu())
        B, K, _ = pred.shape
        for i in range(B):
            p, g = pred[i], gt[i]                             # [K,4]
            pe, ge = p[-1], g[-1]                             # endpoints
            gdyaw_deg = float(ge[3]) * 180.0 / math.pi
            pdyaw_deg = float(pe[3]) * 180.0 / math.pi
            grp = _motion_class(gdyaw_deg, float(ge[2]))
            pos_err = float((p[:, :3] - g[:, :3]).abs().mean())
            end_err = float((pe[:3] - ge[:3]).abs().mean())
            yaw_err = float((p[:, 3] - g[:, 3]).abs().mean()) * 180.0 / math.pi
            # direction cosine on endpoint xy
            pxy, gxy = pe[:2], ge[:2]
            if pxy.norm() > 1e-4 and gxy.norm() > 1e-4:
                dcos = float(torch.dot(pxy, gxy) / (pxy.norm() * gxy.norm()))
            else:
                dcos = float("nan")
            yaw_ok = _sign(pdyaw_deg, SIGN_YAW_DEG) == _sign(gdyaw_deg, SIGN_YAW_DEG)
            dz_ok = _sign(float(pe[2]), SIGN_DZ_M) == _sign(float(ge[2]), SIGN_DZ_M)
            m = {"pos_err_m": pos_err, "end_pos_err_m": end_err,
                 "yaw_err_deg": yaw_err, "dcos": dcos,
                 "yaw_sign_ok": float(yaw_ok), "dz_sign_ok": float(dz_ok),
                 "mae_dx": float(abs(pe[0] - ge[0])), "mae_dy": float(abs(pe[1] - ge[1])),
                 "mae_dz": float(abs(pe[2] - ge[2])),
                 "mae_dyaw_deg": abs(pdyaw_deg - gdyaw_deg)}
            for grp_key in ("all", grp):
                rows[grp_key].append(m)
                ep_pred[grp_key].append(pe.cpu())
                ep_gt[grp_key].append(ge.cpu())
    return rows, ep_pred, ep_gt, torch.cat(act_z_all, 0)


def _agg(ms):
    if not ms:
        return {}
    keys = [k for k in ms[0] if k != "dcos"]
    out = {"n": len(ms)}
    for k in keys:
        out[k] = round(sum(x[k] for x in ms) / len(ms), 4)
    dcos = [x["dcos"] for x in ms if not math.isnan(x["dcos"])]
    out["direction_cos"] = round(sum(dcos) / len(dcos), 4) if dcos else None
    return out


def _channel_std(ep_pred, ep_gt):
    if not ep_pred:
        return {}
    p = torch.stack(ep_pred)   # [N,4]
    g = torch.stack(ep_gt)
    names = ["dx", "dy", "dz", "dyaw"]
    out = {}
    for j, nm in enumerate(names):
        ps = float(p[:, j].std())
        gs = float(g[:, j].std())
        out[nm] = {"pred_std": round(ps, 4), "gt_std": round(gs, 4),
                   "ratio": round(ps / gs, 3) if gs > 1e-6 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ckpt_dir", default="checkpoints/v2_stage2_full_cradio")
    ap.add_argument("--s2_tag", default="best")
    ap.add_argument("--s3_ckpt", default="checkpoints/v2_stage3_cradio/best.pth")
    ap.add_argument("--vision_type", default="cradio_v3_b")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--data_root", default="/root/autodl-tmp/datasets/stage3_uavflow")
    ap.add_argument("--action_stats",
                    default="/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--proprio_dim", type=int, default=4)
    ap.add_argument("--no_proprio", action="store_true",
                    help="proprio-free-query policy (matches v4 training)")
    ap.add_argument("--use_grounding_target", action="store_true",
                    help="v5: grounding-target-token action head")
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--holdout_trajectories", action="store_true",
                    help="Score only the trajectories held out during training "
                         "(read from the checkpoint's 'val_traj_files'). "
                         "Required for any generalisation claim.")
    ap.add_argument("--out", default="reports/s3_systematic_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The checkpoint records the head geometry and target scaling it was trained
    # with. Rebuilding from CLI flags instead would let a mismatched readout or
    # norm_mode load "successfully" and report physically wrong metrics.
    payload = torch.load(args.s3_ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("policy_cfg", {})
    if cfg:
        print(f"[S3-EVAL] policy_cfg from checkpoint: {cfg}", flush=True)
    readout = cfg.get("readout", "last")
    no_proprio = cfg.get("no_proprio", args.no_proprio)
    norm_mode = cfg.get("norm_mode", "zscore")
    chunk_offset = cfg.get("chunk_offset", 0)

    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    load_s2(model, Path(args.s2_ckpt_dir), args.s2_tag)
    model.enable_action_head(chunk_size=args.chunk_size, proprio_dim=args.proprio_dim,
                             use_proprio=not no_proprio,
                             use_grounding_target=args.use_grounding_target,
                             readout=readout, n_bins=cfg.get("n_bins", 0),
                             bin_range=cfg.get("bin_range", 1.5),
                             readout_layers=cfg.get("readout_layers", 2))
    model.to(device)
    load_action_stats(model, args.action_stats, args.chunk_size, args.pos_scale,
                      norm_mode, chunk_offset)
    meta = load_s3(model, payload)
    model.eval()
    print(f"[S3-EVAL] S2={args.s2_ckpt_dir}/{args.s2_tag}  S3={args.s3_ckpt} "
          f"(step={meta['step']} val={meta['val']})", flush=True)

    ds = UAVFlowDataset(
        data_root=args.data_root, tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform, chunk_size=args.chunk_size,
        max_text_len=args.max_text_len, split="train", pos_scale=args.pos_scale,
        aug_flip=False, oversample_turn_factor=1, oversample_class_factor=1,
        chunk_offset=chunk_offset)

    held = payload.get("val_traj_files")
    if args.holdout_trajectories:
        if not held:
            raise SystemExit(
                "[S3-EVAL] --holdout_trajectories requested but the checkpoint "
                "carries no 'val_traj_files'. It was trained with the legacy "
                "chunk split, so no trajectory was ever held out and a "
                "generalisation number cannot be recovered from it.")
        held = set(held)
        keep = {t for t, p in enumerate(ds.kept_traj_files)
                if str(p.relative_to(ds.data_root)) in held}
        idx = [i for i, (t, _) in enumerate(ds.index) if t in keep]
        print(f"[S3-EVAL] holdout: {len(keep)}/{len(held)} val trajectories "
              f"matched -> {len(idx)} unseen chunks", flush=True)
        if len(keep) != len(held):
            raise SystemExit("[S3-EVAL] held-out trajectory list does not match "
                             "this data_root; refusing to report a number.")
    else:
        idx = list(range(len(ds)))
        print("[S3-EVAL] WARNING: scoring the full index, which includes the "
              "trajectories this checkpoint trained on. These are IN-SAMPLE "
              "diagnostics, not generalisation.", flush=True)
    random.Random(args.seed).shuffle(idx)
    idx = idx[:args.n_eval]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=False,
                        collate_fn=aero_collate_fn, num_workers=args.workers)
    print(f"[S3-EVAL] evaluating {len(idx)} samples", flush=True)

    rows, ep_pred, ep_gt, act_z = evaluate(model, loader, device, args.proprio_dim)

    per_group = {}
    for grp in ("all", "turn", "vertical", "straight"):
        a = _agg(rows.get(grp, []))
        if a:
            a["channel_std"] = _channel_std(ep_pred[grp], ep_gt[grp])
            per_group[grp] = a

    global_act_std = round(float(act_z.std()), 5)

    # ── QC gates ────────────────────────────────────────────────────────────
    allg = per_group.get("all", {})
    turng = per_group.get("turn", {})
    cs = allg.get("channel_std", {})
    q1 = allg.get("pos_err_m", 9) < 0.12
    q2 = (turng.get("yaw_sign_ok", 0) > 0.70 and
          (turng.get("direction_cos") or 0) > 0.50) if turng else False
    dz_ratio = (cs.get("dz", {}) or {}).get("ratio") or 0
    yaw_ratio = (cs.get("dyaw", {}) or {}).get("ratio") or 0
    q3 = dz_ratio > 0.25 and yaw_ratio > 0.25
    q4 = global_act_std > 1e-2
    gates = {"Q1_pos_err<0.12": q1, "Q2_turn_direction": q2,
             "Q3_channel_not_collapsed": q3, "Q4_global_not_collapsed": q4}
    verdict = "PASS" if all(gates.values()) else "REVIEW"

    for grp, a in per_group.items():
        cs = a.get("channel_std", {})
        print(f"[{grp:9s}] n={a['n']:4d} pos_err_m={a['pos_err_m']:.4f} "
              f"end={a['end_pos_err_m']:.4f} yaw={a['yaw_err_deg']:.2f} "
              f"dcos={a.get('direction_cos')} yaw_sign={a['yaw_sign_ok']:.3f} "
              f"dz_sign={a['dz_sign_ok']:.3f} | dz_ratio={cs.get('dz',{}).get('ratio')} "
              f"dyaw_ratio={cs.get('dyaw',{}).get('ratio')}", flush=True)
    print(f"[global act z-std] {global_act_std}")
    print("=== S3 QC GATES ===")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"VERDICT={verdict}")

    out = {"s2_ckpt": f"{args.s2_ckpt_dir}/{args.s2_tag}", "s3_ckpt": args.s3_ckpt,
           "step": meta["step"], "val": meta["val"], "n_eval": len(idx),
           "per_group": per_group, "global_act_z_std": global_act_std,
           "gates": gates, "verdict": verdict}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {op}")
    print(f"S3_EVAL_EXIT={0 if verdict == 'PASS' else 1}")


if __name__ == "__main__":
    main()
