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

from data.hugebench_dataset import (PAPER_ORBIT_IDS, STAGE_IGNORE, build_index,
                                    read_episode, task_family)

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

    # Distribution checks run on the full annotated split; only the checks that
    # must open a parquet run on what is downloaded. The release is ordered so
    # that episode index tracks task_id, so a partial download is a skewed
    # sample and reading distributions off it produces confident nonsense.
    allep = build_index(args.data_root, args.anno_root, args.split,
                        require_stages=False, require_file=False)
    eps = build_index(args.data_root, args.anno_root, args.split,
                      require_stages=False)
    cover = len(eps) / max(len(allep), 1)
    print(f"\nsplit '{args.split}': {len(allep)} annotated, {len(eps)} on disk "
          f"({100 * cover:.1f}%)\n")
    if not eps:
        raise SystemExit("nothing on disk")

    rng = random.Random(0)
    pick = rng.sample(eps, min(args.sample, len(eps)))

    # ── 1. actions are exact world-frame pose deltas, in metres ──────────────
    # Tolerance is per-step, not absolute: both state and actions are stored as
    # float32, so reconstructing a 2,340-step episode accumulates rounding. A
    # millimetre after two thousand steps is float32; a unit error would be off
    # by 100x and show up immediately.
    worst_abs = worst_rate = 0.0
    for e in pick:
        st, ac, _, _ = read_episode(e.path, with_images=False)
        n = min(len(st) - 1, len(ac))
        if n <= 0:
            continue
        recon = st[0, :3].astype(np.float64) + np.cumsum(
            ac[:n, :3].astype(np.float64), axis=0)
        err = float(np.abs(recon - st[1:n + 1, :3]).max())
        worst_abs = max(worst_abs, err)
        worst_rate = max(worst_rate, err / n)
    check("state[0] + cumsum(actions) reproduces state[1:]", worst_rate < 1e-5,
          f"max |err| = {worst_abs * 1e3:.3f} mm, per step {worst_rate * 1e6:.2f} um")

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
    cov = [float((e.stages >= 0).mean()) for e in allep]
    dom = set()
    for e in allep:
        dom.update(int(v) for v in np.unique(e.stages) if v >= 0)
    n_unann = sum(1 for c in cov if c == 0.0)
    check("stage coverage > 90% of frames", float(np.mean(cov)) > 0.90,
          f"mean {100 * np.mean(cov):.1f}%, {n_unann} episodes unannotated")
    check("subtask_id domain is contiguous from 0",
          dom == set(range(max(dom) + 1)) if dom else False,
          f"{sorted(dom)}")

    # ── 4. family mix; Orbit is where the partial observability lives ────────
    fam, frames = collections.Counter(), collections.Counter()
    for e in allep:
        fam[e.family] += 1
        frames[e.family] += e.length
    tot_e, tot_f = sum(fam.values()), sum(frames.values())
    for k in sorted(fam):
        print(f"  {k:9s} eps {fam[k]:5d} ({100 * fam[k] / tot_e:4.1f}%)  "
              f"frames {frames[k]:7d} ({100 * frames[k] / tot_f:4.1f}%)")
    paper = sum(1 for e in allep if e.task_id in PAPER_ORBIT_IDS)
    check("published Orbit share (hl+orbit+orbit_multi) is ~34% of episodes",
          30 <= 100 * paper / tot_e <= 38, f"{100 * paper / tot_e:.1f}%")
    check("every episode maps to a known family", fam.get("other", 0) == 0,
          f"other={fam.get('other', 0)}")

    # ── 5. episode length distribution ───────────────────────────────────────
    L = np.array([e.length for e in allep])
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
    # Annotations again, not disk: the whole point of the number is that
    # test_seen reuses train instructions, and a partial train set understates
    # it. The two splits must always be reported separately.
    try:
        tr = {e.instruction for e in build_index(
            args.data_root, args.anno_root, "train", require_stages=False,
            require_file=False)}
        for s, want in (("test_seen", 0.957), ("test_unseen", 0.058)):
            te = build_index(args.data_root, args.anno_root, s,
                             require_stages=False, require_file=False)
            ov = float(np.mean([e.instruction in tr for e in te]))
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
