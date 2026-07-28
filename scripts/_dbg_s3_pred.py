"""Dump pred-vs-gt endpoints for a few UAV-Flow samples to diagnose the
train-val (pos_err 0.14, dir 0.22) vs systematic-eval (pos_err 0.45,
direction_cos -0.53) discrepancy on best_grounded.pth (proprio-free-query)."""
import argparse, math, random, torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from model.aerov2 import AeroV2
from data.dataset import UAVFlowDataset, aero_collate_fn
from scripts.eval_s3_systematic import load_s2, load_s3, load_action_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ckpt_dir", default="checkpoints/v2_stage2_full_cradio")
    ap.add_argument("--s3_ckpt", default="checkpoints/v2_stage3_cradio_v4/best_grounded.pth")
    ap.add_argument("--vision_type", default="cradio_v3_b")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--data_root", default="/root/autodl-tmp/datasets/stage3_uavflow")
    ap.add_argument("--action_stats", default="/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--proprio_dim", type=int, default=4)
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--save_npz", default="")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(dev)
    load_s2(model, Path(args.s2_ckpt_dir), "best")
    model.enable_action_head(chunk_size=args.chunk_size, proprio_dim=args.proprio_dim,
                             use_proprio=False)
    model.to(dev)
    load_action_stats(model, args.action_stats, args.chunk_size, args.pos_scale)
    load_s3(model, Path(args.s3_ckpt))
    model.eval()

    ds = UAVFlowDataset(data_root=args.data_root, tokenizer=model.tokenizer,
                        transform=model.vision_encoder.transform,
                        chunk_size=args.chunk_size, max_text_len=64, split="train",
                        pos_scale=args.pos_scale, aug_flip=False,
                        oversample_turn_factor=1, oversample_class_factor=1)
    idx = list(range(len(ds))); random.Random(123).shuffle(idx); idx = idx[:args.n]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=False,
                        collate_fn=aero_collate_fn, num_workers=4)
    preds, gts = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(dev); ids = batch["input_ids"].to(dev)
        prop = batch["state8"].to(dev).float()[:, :args.proprio_dim]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_action(pv, ids, prop)
        preds.append(model.action_head.denormalize(out["action"].float()).cpu())
        gts.append(batch["gt_action"].float())
    pred = torch.cat(preds, 0); gt = torch.cat(gts, 0)
    if args.save_npz:
        import numpy as np
        # save endpoints [N,4] pred + gt (dx,dy,dz,dyaw_rad)
        np.savez(args.save_npz, pred_end=pred[:, -1].numpy(), gt_end=gt[:, -1].numpy())
        print(f"[SAVED] {args.save_npz}  N={pred.size(0)}")
    print("idx | pred_end(dx,dy,dz,dyawdeg) | gt_end(dx,dy,dz,dyawdeg) | dcos_xy")
    for i in range(min(args.n, 12)):
        pe, ge = pred[i, -1], gt[i, -1]
        pxy, gxy = pe[:2], ge[:2]
        dcos = float(torch.dot(pxy, gxy) / (pxy.norm() * gxy.norm() + 1e-9))
        print(f"{i:2d} | ({pe[0]:+.2f},{pe[1]:+.2f},{pe[2]:+.2f},{pe[3]*180/math.pi:+6.1f}) "
              f"| ({ge[0]:+.2f},{ge[1]:+.2f},{ge[2]:+.2f},{ge[3]*180/math.pi:+6.1f}) | {dcos:+.2f}")
    # global mean of pred (mean-collapse check)
    print("pred endpoint mean:", pred[:, -1].mean(0).tolist())
    print("pred endpoint std :", pred[:, -1].std(0).tolist())
    print("gt   endpoint mean:", gt[:, -1].mean(0).tolist())


if __name__ == "__main__":
    main()
