"""HUGE-Bench loader for AeroV3.

Sample unit is a *window* of W consecutive inference steps from one episode,
spaced by the official ``exec_steps`` stride, not a single frame. The phase
filter is recurrent, so it needs consecutive steps; the stride has to match
evaluation or the filter learns a different update rate than it runs at.

Three facts about the release drive the design here:

- Images are embedded in the parquet (LeRobot image mode), so training needs
  no renderer, but one episode file is ~28 MB and reading it per sample would
  dominate. Windows are therefore drawn several at a time per episode read.
- ``first_image`` is repeated on every row. It costs almost nothing on disk
  (parquet dictionary-encodes the duplicate: 1.2% of file bytes against 98.8%
  for ``image``), but decoding it per frame would be pure waste. Decoded once
  per episode.
- Instructions are not in the parquet. They come from the stage-annotation
  sidecar in the HUGE-Bench source repo, which also carries the stage labels
  the phase head is supervised on.
"""
from __future__ import annotations

import collections
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Official protocol constants (openpi/scripts/action_infer.py). Changing these
# silently makes our numbers incomparable to the published baselines.
ACTION_HORIZON = 20
EXEC_STEPS = 10

STAGE_IGNORE = -1


# ── task families ────────────────────────────────────────────────────────────
# Every result is reported split by family, because the Orbit family is where
# the aliasing and the partial observability live (2.3-4.8% aliased frames
# against 0.0% elsewhere) and Inspect behaves in the opposite direction.
#
# Grouping is by ``task_id`` from the annotations, not by instruction text. A
# keyword classifier gets this wrong: ``building`` reads "Orbit the building in
# the upper left of the view once" and is an orbit motion, but the paper's
# 34%-of-episodes Orbit group counts only hl / orbit / orbit_multi. The table
# below is the measured composition of all 5,175 train episodes.
#
#   task_id      eps%   frames%  stages  motion
#   0            33.5    10.0      3     fly to N metres above a named target
#   building     10.1    17.6      6     orbit a referred building, no radius
#   hl           13.4    11.1      6     orbit at a named altitude
#   orbit        13.4    11.9      6     orbit at a named altitude and radius
#   orbit_multi   7.1    11.3      6     spiral down around a target
#   road         11.7    17.8      5     follow a road in a given direction
#   farm          2.2    10.1      4     boustrophedon mapping sweep
#   obstacle      8.5    10.3     1-2    reach a pose behind/between obstacles

TASK_FAMILY = {
    "0": "inspect",
    "building": "orbit",
    "hl": "orbit",
    "orbit": "orbit",
    "orbit_multi": "orbit",
    "road": "road",
    "farm": "survey",
    "obstacle": "obstacle",
}

# The three ids the published Orbit share (34% of episodes) is computed over.
PAPER_ORBIT_IDS = ("hl", "orbit", "orbit_multi")


def task_family(task_id: str) -> str:
    return TASK_FAMILY.get(str(task_id), "other")


@dataclass
class Episode:
    index: int
    path: Path
    env_id: str
    task_id: str
    instruction: str
    length: int
    family: str
    # per-frame stage id, STAGE_IGNORE outside any annotated segment
    stages: np.ndarray

    @property
    def num_stages(self) -> int:
        v = self.stages[self.stages >= 0]
        return int(v.max()) + 1 if v.size else 0


def _episode_path(root: Path, split: str, idx: int) -> Path:
    return root / split / "data" / f"chunk-{idx // 1000:03d}" / f"episode_{idx:06d}.parquet"


def build_index(data_root: str | Path, anno_root: str | Path, split: str,
                require_stages: bool = True,
                families: Sequence[str] | None = None,
                require_file: bool = True) -> list[Episode]:
    """Index episodes from the sidecar annotations.

    ``require_file=False`` indexes the full annotated split regardless of what
    has been downloaded. Distribution statistics must use that: the release is
    ordered so episode index correlates with ``task_id``, which makes any
    prefix of the download a badly skewed sample of families and lengths.

    The annotations are the authority for length and instruction. ``info.json``
    is not: it reports ``total_tasks: 109`` against 5,175 episodes and 1,102
    distinct instructions, so anything indexing off it lands on the wrong rows.
    """
    data_root, anno_root = Path(data_root), Path(anno_root)
    seg_file = anno_root / "stage_segments" / f"{split}.jsonl"
    map_file = anno_root / "episode_mapping" / f"{split}.jsonl"
    for f in (seg_file, map_file):
        if not f.exists():
            raise FileNotFoundError(f"stage annotation missing: {f}")

    segs: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    with open(seg_file, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            segs[r["episode_index"]].append(
                (r["frame_start"], r["frame_end"], r["subtask_id"]))

    out: list[Episode] = []
    missing = 0
    with open(map_file, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            idx, T = r["episode_index"], int(r["length"])
            p = _episode_path(data_root, split, idx)
            if not p.exists():
                missing += 1
                if require_file:
                    continue
            stages = np.full(T, STAGE_IGNORE, dtype=np.int16)
            for a, b, s in segs.get(idx, []):
                stages[a:min(b, T - 1) + 1] = s
            if require_stages and not (stages >= 0).any():
                continue
            fam = task_family(r["task_id"])
            if families and fam not in families:
                continue
            out.append(Episode(idx, p, r["env_id"], str(r["task_id"]),
                               r["instruction"], T, fam, stages))
    if missing:
        print(f"[hugebench] {split}: {missing} annotated episodes not on disk",
              flush=True)
    return out


def split_by_episode(eps: list[Episode], val_frac: float = 0.03,
                     seed: int = 0) -> tuple[list[Episode], list[Episode]]:
    """Hold out whole episodes. Frames inside an episode are near-duplicates,
    so a frame-level split reports a validation number that is mostly memory."""
    rng = random.Random(seed)
    order = list(eps)
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_frac))
    return order[n_val:], order[:n_val]


# ── parquet access ───────────────────────────────────────────────────────────

_COLS = ["state", "actions", "image", "first_image"]


def read_episode(path: Path, with_images: bool = True):
    import pyarrow.parquet as pq
    cols = _COLS if with_images else ["state", "actions"]
    t = pq.read_table(path, columns=cols)
    state = np.asarray(t["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(t["actions"].to_pylist(), dtype=np.float32)
    if not with_images:
        return state, actions, None, None
    imgs = t["image"].combine_chunks().field("bytes").to_pylist()
    first = t["first_image"].combine_chunks().field("bytes")[0].as_py()
    return state, actions, imgs, first


def _decode(buf: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(buf)).convert("RGB")


# ── action normalisation ─────────────────────────────────────────────────────

@dataclass
class ActionStats:
    q01: np.ndarray
    q99: np.ndarray

    def to(self, device):
        return (torch.as_tensor(self.q01, device=device),
                torch.as_tensor(self.q99, device=device))

    def normalise(self, a: np.ndarray) -> np.ndarray:
        span = np.maximum(self.q99 - self.q01, 1e-6)
        return np.clip(2.0 * (a - self.q01) / span - 1.0, -5.0, 5.0)

    def denormalise(self, a: np.ndarray) -> np.ndarray:
        span = np.maximum(self.q99 - self.q01, 1e-6)
        return (a + 1.0) * 0.5 * span + self.q01

    def save(self, path):
        json.dump({"q01": self.q01.tolist(), "q99": self.q99.tolist()},
                  open(path, "w"))

    @staticmethod
    def load(path) -> "ActionStats":
        d = json.load(open(path))
        return ActionStats(np.asarray(d["q01"], np.float32),
                           np.asarray(d["q99"], np.float32))


def compute_action_stats(eps: Sequence[Episode], max_episodes: int = 400,
                         seed: int = 0) -> ActionStats:
    """Quantile stats over per-step actions. Recompute whenever the horizon or
    the episode subset changes -- the v2 line lost a run to stale stats."""
    rng = random.Random(seed)
    pick = list(eps)
    rng.shuffle(pick)
    acc = [read_episode(e.path, with_images=False)[1] for e in pick[:max_episodes]]
    a = np.concatenate(acc, axis=0)
    return ActionStats(np.quantile(a, 0.01, axis=0).astype(np.float32),
                       np.quantile(a, 0.99, axis=0).astype(np.float32))


# ── windowed iterable dataset ────────────────────────────────────────────────

class HugeBenchWindows(IterableDataset):
    """Streams windows of W inference steps.

    ``windows_per_episode`` amortises the ~28 MB parquet read. Set it to 1 for
    strict i.i.d. sampling at the cost of throughput.
    """

    def __init__(self, episodes: list[Episode], stats: ActionStats,
                 transform: Callable, window: int = 4,
                 stride: int = EXEC_STEPS, horizon: int = ACTION_HORIZON,
                 windows_per_episode: int = 4, seed: int = 0,
                 vision_rates: Sequence[int] = (1, 1, 2, 4, 0),
                 infinite: bool = True):
        super().__init__()
        self.eps = episodes
        self.stats = stats
        self.transform = transform
        self.W = window
        self.stride = stride
        self.H = horizon
        self.wpe = windows_per_episode
        self.seed = seed
        # 0 means "never refresh after the first step". Sampling the rate per
        # episode makes it a first-class input, so the L2.5 frequency ablation
        # is inference-only and costs no extra training run.
        self.vision_rates = list(vision_rates)
        self.infinite = infinite
        self.min_len = stride * (window - 1) + horizon + 1

    def _windows(self, ep: Episode, rng: random.Random):
        span = self.stride * (self.W - 1)
        hi = ep.length - span - self.H - 1
        if hi <= 1:
            return []
        return [rng.randint(1, hi) for _ in range(self.wpe)]

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 9973 + wid)
        pool = [e for i, e in enumerate(self.eps)
                if i % nw == wid and e.length >= self.min_len]
        if not pool:
            return
        epoch = 0
        while True:
            rng.shuffle(pool)
            for ep in pool:
                starts = self._windows(ep, rng)
                if not starts:
                    continue
                try:
                    state, actions, imgs, first = read_episode(ep.path)
                except Exception as e:  # corrupt shard: skip loudly, do not mask
                    print(f"[hugebench] skip {ep.path.name}: "
                          f"{type(e).__name__}", flush=True)
                    continue
                first_px = self.transform(_decode(first))
                rate = rng.choice(self.vision_rates)
                for t0 in starts:
                    s = self._make(ep, state, actions, imgs, first_px, t0, rate)
                    if s is not None:
                        yield s
            epoch += 1
            if not self.infinite:
                return

    def _make(self, ep, state, actions, imgs, first_px, t0, rate):
        T = min(len(state), len(actions))
        frames = [t0 + k * self.stride for k in range(self.W)]
        if frames[-1] + self.H >= T:
            return None

        px, upd, chunks, chunk_mask, stages, prog, poses = [], [], [], [], [], [], []
        last = None
        for k, f in enumerate(frames):
            fresh = (k == 0) or (rate > 0 and k % rate == 0)
            if fresh:
                last = self.transform(_decode(imgs[f]))
            px.append(last)
            upd.append(1.0 if fresh else 0.0)

            a = actions[f:f + self.H]
            m = np.ones(len(a), dtype=np.float32)
            if len(a) < self.H:  # tail pad; masked out of the loss
                pad = self.H - len(a)
                a = np.concatenate([a, np.zeros((pad, a.shape[1]), np.float32)])
                m = np.concatenate([m, np.zeros(pad, np.float32)])
            chunks.append(self.stats.normalise(a))
            chunk_mask.append(m)
            stages.append(int(ep.stages[min(f, len(ep.stages) - 1)]))
            prog.append(f / max(T - 1, 1))
            poses.append(state[f])

        return {
            "pixel_values": torch.stack(px),                       # (W,3,H,W)
            "first_pixel_values": first_px,                        # (3,H,W)
            "vision_update": torch.tensor(upd),                    # (W,)
            "pose": torch.from_numpy(np.stack(poses)),             # (W,4)
            "pose0": torch.from_numpy(state[0].copy()),            # (4,)
            "action": torch.from_numpy(np.stack(chunks)).float(),  # (W,H,4)
            "action_mask": torch.from_numpy(np.stack(chunk_mask)), # (W,H)
            "stage": torch.tensor(stages, dtype=torch.long),       # (W,)
            "progress": torch.tensor(prog, dtype=torch.float),     # (W,)
            "instruction": ep.instruction,
            "family": ep.family,
            "env_id": ep.env_id,
            "episode_index": ep.index,
        }


def collate(batch: list[dict], tokenizer=None, max_len: int = 48) -> dict:
    out = {}
    for k in ("pixel_values", "first_pixel_values", "vision_update", "pose",
              "pose0", "action", "action_mask", "stage", "progress"):
        out[k] = torch.stack([b[k] for b in batch])
    for k in ("instruction", "family", "env_id", "episode_index"):
        out[k] = [b[k] for b in batch]
    if tokenizer is not None:
        tok = tokenizer(out["instruction"], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        out["input_ids"] = tok["input_ids"]
        out["attention_mask"] = tok["attention_mask"]
    return out
