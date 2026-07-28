"""HUGE-Bench data self-check — the checklist in AEROV3_DATA_BUILD.md section 8.

    python scripts/hugebench_data_qc.py --data_root $HUGE --anno_root $ANNO

Run before training and again whenever the download changes. Each check exists
because getting it wrong is silent: a unit error, a stage misalignment or a
mis-split leaves training running and only shows up as a number that is hard to
argue with weeks later.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.hugebench_dataset import (STAGE_IGNORE, build_index, read_episode,
                                    task_family)

FAILED: list[str] = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--anno_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()

    eps = build_index(args.data_root, args.anno_root, args.split,
                      require_stages=False)
    print(f"\nindexed {len(eps)} episodes on disk for split '{args.split}'\n")
    if not eps:
        raise SystemExit("nothing indexed")

    rng = random.Random(0)
    pick = rng.sample(eps, min(args.sample, len(eps)))

    # ── 1. actions are exact world-frame pose deltas, in metres ──────────────
    worst = 0.0
    for e in pick:
        st, ac, _, _ = read_episode(e.path, with_images=False)
        n = min(len(st) - 1, len(ac))
        if n <= 0:
            continue
        recon = st[0, :3] + np.cumsum(ac[:n, :3], axis=0)
        worst = max(worst, float(np.abs(recon - st[1:n + 1, :3]).max()))
    check("state[0] + cumsum(actions) reproduces state[1:]", worst < 1e-3,
          f"max |err| = {worst:.6f} m")

    # ── 2. first_image really is constant within an episode ──────────────────
    import pyarrow.parquet as pq
    const = True
    for e in pick[:8]:
        t = pq.read_table(e.path, columns=["first_image"])
        b = t["first_image"].combine_chunks().field("bytes").to_pylist()
        if len({bytes(x) for x in b}) != 1:
            const = False
            break
    check("first_image is byte-identical across all rows", const)

    # ── 3. stage annotation coverage and label domain ────────────────────────
    cov = [float((e.stages >= 0).mean()) for e in eps]
    dom = set()
    for e in eps:
        dom.update(int(v) for v in np.unique(e.stages) if v >= 0)
    n_unann = sum(1 for c in cov if c == 0.0)
    check("stage coverage > 90% of frames", float(np.mean(cov)) > 0.90,
          f"mean {100 * np.mean(cov):.1f}%, {n_unann} episodes unannotated")
    check("subtask_id domain is contiguous from 0",
          dom == set(range(max(dom) + 1)) if dom else False,
          f"{sorted(dom)}")

    # ── 4. family mix; Orbit is where the partial observability lives ────────
    fam = collections.Counter(e.family for e in eps)
    frames = collections.Counter()
    for e in eps:
        frames[e.family] += e.length
    tot = sum(frames.values())
    share = 100 * frames["orbit"] / max(tot, 1)
    print("  families (episodes / frames):", {
        k: (fam[k], frames[k]) for k in sorted(fam)})
    check("orbit family share is 25-45% of frames", 25 <= share <= 45,
          f"{share:.1f}%")
    check("no episode falls outside the two named families",
          fam.get("other", 0) == 0, f"other={fam.get('other', 0)}")

    # ── 5. episode length distribution ───────────────────────────────────────
    L = np.array([e.length for e in eps])
    print(f"  length p50={int(np.median(L))} p90={int(np.percentile(L, 90))} "
          f"max={int(L.max())} min={int(L.min())}")
    check("median episode length near the documented 267",
          200 <= np.median(L) <= 340, f"p50={int(np.median(L))}")

    # ── 6. the two camera conventions differ on state[3] ─────────────────────
    # Non-obstacle tasks put yaw in state[3]; obstacle tasks put phi there. If
    # the ranges are indistinguishable, code that treats them uniformly will
    # fail silently, so this is worth knowing before writing the renderer path.
    rng3 = collections.defaultdict(list)
    for e in pick:
        st, _, _, _ = read_episode(e.path, with_images=False)
        rng3[e.task_id].extend(st[:, 3].tolist())
    for k in sorted(rng3):
        v = np.asarray(rng3[k])
        print(f"  task_id={k:12s} state[3] range [{v.min():+.3f}, {v.max():+.3f}] "
              f"n={len(v)}")

    # ── 7. instruction overlap between splits ────────────────────────────────
    try:
        tr = {e.instruction for e in build_index(args.data_root, args.anno_root,
                                                 "train", require_stages=False)}
        for s, want in (("test_seen", 0.957), ("test_unseen", 0.058)):
            te = build_index(args.data_root, args.anno_root, s,
                             require_stages=False)
            if not te:
                print(f"  {s}: not on disk, skipped")
                continue
            ov = np.mean([e.instruction in tr for e in te])
            check(f"{s} instruction overlap near {100 * want:.1f}%",
                  abs(ov - want) < 0.10, f"{100 * ov:.1f}%")
    except FileNotFoundError as e:
        print(f"  split comparison skipped: {e}")

    print()
    if FAILED:
        print(f"HUGEBENCH_QC: FAIL ({len(FAILED)}): {', '.join(FAILED)}")
        return 1
    print("HUGEBENCH_QC: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
