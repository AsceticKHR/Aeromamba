"""Is HUGE-Bench partially observable under its own observation protocol?

    python scripts/hugebench_partial_observability.py --data_root $D --anno_root $A

Two independent halves. Memory is required only if BOTH hold:

    H1  the stage label carries action information beyond the observation
    H2  the stage label cannot be inferred from the observation

Observation O = (env_id, instruction, initial pose, current pose). The scene is
static and the camera pose is a deterministic function of ``state``, so O is
exactly what the official "first frame + current frame + text" input encodes.
No images are read.

Results are reported per ``task_id``. An earlier run of this test labelled its
groups by instruction text and reported an "Inspect" family and an "Orbit"
family behaving in opposite directions; instruction text alone cannot tell
``task_id=0`` ("Fly to 60 meters above ...") from ``task_id=road`` ("Inspect
the road in the view, ..."), and the two are different tasks. Everything is
keyed on task_id here so the attribution is checkable.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.hugebench_dataset import build_index, read_episode

# Yaw in radians against position in metres: without a weight the neighbour
# search is decided entirely by position.
W = np.array([1.0, 1.0, 1.0, 10.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--anno_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--min_group", type=int, default=8)
    ap.add_argument("--groups_per_task", type=int, default=4)
    ap.add_argument("--queries", type=int, default=1200)
    args = ap.parse_args()
    K = args.horizon

    eps = build_index(args.data_root, args.anno_root, args.split)
    groups = collections.defaultdict(list)
    for e in eps:
        groups[(e.task_id, e.env_id, e.instruction)].append(e)

    by_task = collections.defaultdict(list)
    for k, v in groups.items():
        if len(v) >= args.min_group:
            by_task[k[0]].append((k, v))
    for t in by_task:
        by_task[t].sort(key=lambda kv: -len(kv[1]))
        by_task[t] = by_task[t][:args.groups_per_task]

    print(f"indexed {len(eps)} episodes, {len(groups)} groups; dense groups "
          f"per task: { {t: len(v) for t, v in sorted(by_task.items())} }\n",
          flush=True)

    summary = collections.defaultdict(list)
    for task in sorted(by_task):
        print(f"===== task_id = {task} =====", flush=True)
        for (_, env, ins), members in by_task[task]:
            F = []
            for e in members:
                try:
                    st, _, _, _ = read_episode(e.path, with_images=False)
                except Exception:
                    continue
                T = min(len(st), len(e.stages))
                if T <= K + 2:
                    continue
                fut = st[K:T, :3] - st[:T - K, :3]
                for t in range(T - K):
                    s = int(e.stages[t])
                    if s < 0:
                        continue
                    F.append((e.index, st[0], st[t], fut[t], s))
            if len(F) < 400:
                continue

            epi = np.array([f[0] for f in F])
            X = np.concatenate([np.array([f[1] for f in F]) * W,
                                np.array([f[2] for f in F]) * W], axis=1)
            Y = np.array([f[3] for f in F])
            P = np.array([f[4] for f in F])

            rng = np.random.default_rng(0)
            q = rng.choice(len(F), size=min(args.queries, len(F)), replace=False)
            eO, eOS, acc = [], [], []
            for j in q:
                # Exclude the query's own episode. Neighbouring frames of one
                # trajectory are near-duplicates and would score near zero.
                other = epi != epi[j]
                if other.sum() < args.knn * 3:
                    continue
                oi = np.where(other)[0]
                d = np.linalg.norm(X[oi] - X[j], axis=1)
                nn = oi[np.argsort(d)[:args.knn]]
                eO.append(np.linalg.norm(Y[nn].mean(0) - Y[j]))
                acc.append(collections.Counter(P[nn]).most_common(1)[0][0] == P[j])
                same = nn[P[nn] == P[j]]
                if len(same) == 0:
                    cand = oi[P[oi] == P[j]]
                    if len(cand) == 0:
                        continue
                    same = cand[np.argsort(np.linalg.norm(X[cand] - X[j], axis=1))
                                [:args.knn]]
                eOS.append(np.linalg.norm(Y[same].mean(0) - Y[j]))
            if not eO or not eOS:
                continue

            gt = float(np.linalg.norm(Y, axis=1).mean())
            eO_, eOS_ = float(np.mean(eO)), float(np.mean(eOS))
            a = float(np.mean(acc))
            maj = collections.Counter(P).most_common(1)[0][1] / len(P)
            drop = 100 * (1 - eOS_ / max(eO_, 1e-9))
            summary[task].append((len(F), gt, eO_, eOS_, a, maj, drop))
            print(f"  [{len(set(P.tolist()))}st n={len(F):6d}] "
                  f"|GT|={gt:6.2f}  E(O)={eO_:6.2f}  E(O,S)={eOS_:6.2f}  "
                  f"drop={drop:5.1f}%  acc(S|O)={a:.3f} (chance {maj:.3f})  "
                  f"{ins[:46]}", flush=True)

    print("\n================ PER-TASK SUMMARY ================")
    print(f"{'task_id':13s}{'|GT|':>8s}{'E(O)':>8s}{'E(O,S)':>9s}"
          f"{'drop%':>8s}{'acc':>7s}{'chance':>8s}{'verdict':>26s}")
    for task in sorted(summary):
        r = np.array([x[:7] for x in summary[task]], dtype=float)
        w = r[:, 0] / r[:, 0].sum()
        gt, eO, eOS, acc, maj, drop = (float(w @ r[:, i]) for i in range(1, 7))
        # Memory pays only where the stage both adds information and cannot be
        # read off the pose. Either half alone is not enough.
        if drop > 15 and acc < maj + 0.05:
            v = "PARTIALLY OBSERVABLE"
        elif drop > 15:
            v = "stage helps, but inferable"
        else:
            v = "fully observable"
        print(f"{task:13s}{gt:8.2f}{eO:8.2f}{eOS:9.2f}{drop:8.1f}"
              f"{acc:7.3f}{maj:8.3f}{v:>26s}")
    print("\nPARTIAL_OBS_DONE", flush=True)


if __name__ == "__main__":
    main()
