"""How much of the action is the instruction alone allowed to explain?

The vision gate assumes a policy *should* depend strongly on the image. That is
an assumption about the data, not about the model, and it is cheap to check
without training anything: group chunks by their (identical) instruction and
measure how much the ground-truth endpoint still varies inside a group.

  within-group spread / total spread  =  the share of endpoint variation that
  the instruction CANNOT explain, i.e. the ceiling for any visual (or
  historical) signal at this chunk horizon.

If that ceiling is small at K=8, no readout architecture can raise vis_sens and
the horizon — not the readout — is what needs to change. Reported across
several K so the trade-off is visible.

  python scripts/probe_action_information.py --data_root <stage3_uavflow>
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import UAVFlowDataset  # noqa: E402


def analyse(ds, k_label: int, min_group: int, max_samples: int):
    """Nested variance decomposition of the chunk endpoint.

        total = between-instruction        <- the instruction explains this
              + between-trajectory | instr <- same words, different scene:
                                              the CURRENT FRAME can explain it
              + within-trajectory          <- same scene, different phase:
                                              only HISTORY can explain it

    The last two are what separates "fix the readout" from "add temporal input".
    """
    by_instr = defaultdict(lambda: defaultdict(list))
    stride = max(1, len(ds.index) // max_samples)
    for i in range(0, len(ds.index), stride):
        traj_idx, step_idx = ds.index[i]
        traj = ds.trajectories[traj_idx]
        instr = traj[0].get("instruction", "")
        ep = ds._extract_chunk(traj, step_idx)[-1].numpy()
        by_instr[instr][traj_idx].append(ep)

    # Keep instructions spoken by several DIFFERENT trajectories, otherwise the
    # between-trajectory term is not estimable.
    groups = {k: {t: np.stack(v) for t, v in trajs.items() if len(v) >= 2}
              for k, trajs in by_instr.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 2
              and sum(g.shape[0] for g in v.values()) >= min_group}
    if not groups:
        return None

    allv = np.concatenate([g for trajs in groups.values() for g in trajs.values()])
    n_tot = allv.shape[0]
    total_var = allv.var(axis=0)

    within_traj = np.zeros(4)
    between_traj = np.zeros(4)
    for trajs in groups.values():
        cat = np.concatenate(list(trajs.values()))
        instr_mean = cat.mean(axis=0)
        for g in trajs.values():
            within_traj += g.var(axis=0) * g.shape[0]
            between_traj += ((g.mean(axis=0) - instr_mean) ** 2) * g.shape[0]
    within_traj /= n_tot
    between_traj /= n_tot
    between_instr = np.maximum(total_var - within_traj - between_traj, 0.0)

    return {
        "k": k_label,
        "n_instr": len(groups),
        "n_traj": sum(len(v) for v in groups.values()),
        "n_samples": int(n_tot),
        "mean_endpoint_m": float(np.abs(allv[:, :3]).sum(axis=1).mean()),
        "total_std": total_var ** 0.5,
        "f_instr": between_instr / np.maximum(total_var, 1e-12),
        "f_scene": between_traj / np.maximum(total_var, 1e-12),
        "f_phase": within_traj / np.maximum(total_var, 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/autodl-tmp/datasets/stage3_uavflow")
    ap.add_argument("--chunk_sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--chunk_offset", type=int, default=1)
    ap.add_argument("--min_group", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=120000)
    args = ap.parse_args()

    names = ("dx", "dy", "dz", "dyaw")
    print("variance share of the chunk endpoint:  instr | scene | phase")
    print("  instr = between instructions        -> the language explains it")
    print("  scene = same words, other trajectory-> the CURRENT FRAME can explain it")
    print("  phase = same trajectory, other step -> only HISTORY can explain it\n")
    print(f"{'K':>3} {'instr':>6} {'traj':>6} {'samples':>8} {'|end|m':>7}  "
          + "  ".join(f"{n:>18}" for n in names))
    for K in args.chunk_sizes:
        ds = UAVFlowDataset(data_root=args.data_root, tokenizer=None, transform=None,
                            chunk_size=K, split="train", aug_flip=False,
                            oversample_turn_factor=1, oversample_class_factor=1,
                            chunk_offset=args.chunk_offset)
        r = analyse(ds, K, args.min_group, args.max_samples)
        if r is None:
            print(f"{K:>3}  (too few groups)")
            continue
        cells = [f"{r['f_instr'][d]:.0%}|{r['f_scene'][d]:.0%}|{r['f_phase'][d]:.0%}"
                 .rjust(18) for d in range(4)]
        print(f"{r['k']:>3} {r['n_instr']:>6} {r['n_traj']:>6} {r['n_samples']:>8} "
              f"{r['mean_endpoint_m']:>7.2f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
