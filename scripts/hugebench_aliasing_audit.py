"""Observation-aliasing audit for HUGE-Bench.

The scene is static, so the official two-frame observation (first frame +
current frame) is a deterministic function of (env_id, initial pose, current
pose). Aliasing is therefore exactly computable from the 4-D state column
alone -- no images required.
"""
import json, io, re, time, os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests, pyarrow.parquet as pq

EP  = "https://hf-mirror.com"
RID = "yu781986168/HUGE_Dataset_v0"
COLS = ["actions", "state", "task_index", "env_id"]
S = requests.Session()
def url(rel): return f"{EP}/datasets/{RID}/resolve/main/{rel}"

class HttpFile(io.RawIOBase):
    """Seekable HTTP file. readinto (not read) is what BufferedReader calls."""
    def __init__(self, u):
        self.u, self.pos = u, 0
        r = S.head(u, allow_redirects=True, timeout=60); r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos+off if whence == 1 else self.size+off)
        return self.pos
    def tell(self):     return self.pos
    def seekable(self): return True
    def readable(self): return True
    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size: return 0
        end = min(self.pos + n, self.size) - 1
        for _ in range(4):
            try:
                r = S.get(self.u, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=90)
                r.raise_for_status(); d = r.content
                b[:len(d)] = d; self.pos += len(d); return len(d)
            except Exception: time.sleep(2)
        raise IOError("range read failed")

def get_text(rel):
    for _ in range(5):
        try:
            r = S.get(url(rel), timeout=60); r.raise_for_status(); return r.text
        except Exception as e: last = e; time.sleep(2)
    raise last

def read_ep(args):
    split, idx = args
    rel = f"{split}/data/chunk-{idx//1000:03d}/episode_{idx:06d}.parquet"
    try:
        t = pq.read_table(io.BufferedReader(HttpFile(url(rel)), 1 << 20), columns=COLS)
        return dict(st=np.array(t["state"].to_pylist(), dtype=np.float64),
                    ac=np.array(t["actions"].to_pylist(), dtype=np.float64),
                    ti=int(t["task_index"][0].as_py()),
                    env=str(t["env_id"][0].as_py()), idx=idx)
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}

def load(split, n):
    tm = {j["task_index"]: j["task"] for j in
          (json.loads(l) for l in get_text(f"{split}/meta/tasks.jsonl").splitlines() if l.strip())}
    eps = [json.loads(l) for l in get_text(f"{split}/meta/episodes.jsonl").splitlines() if l.strip()]
    idxs = [e["episode_index"] for e in eps]
    rng = np.random.default_rng(0)
    if n < len(idxs): idxs = [idxs[i] for i in sorted(rng.choice(len(idxs), n, replace=False))]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(read_ep, [(split, i) for i in idxs]))
    ok  = [o for o in out if "err" not in o]
    bad = [o for o in out if "err" in o]
    print(f"[{split}] {len(ok)}/{len(idxs)} ok in {time.time()-t0:.0f}s", flush=True)
    if bad: print("   first error:", bad[0]["err"][:150], flush=True)
    return tm, ok

# ---- sanity: are actions world-frame pose deltas? ----
def check_action_convention(eps):
    errs = []
    for e in eps[:40]:
        st, ac = e["st"], e["ac"]
        pred = st[0, :3] + np.cumsum(ac[:-1, :3], axis=0)
        errs.append(np.abs(pred - st[1:, :3]).max())
    print(f"[sanity] max|state[0]+cumsum(actions) - state[1:]| over 40 eps: "
          f"median={np.median(errs):.4f} m  max={np.max(errs):.4f} m", flush=True)
    print("         -> actions are WORLD-FRAME pose deltas" if np.median(errs) < 0.05
          else "         -> WARNING: actions are NOT world-frame deltas", flush=True)

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

K = 20                 # HUGE-Bench action horizon
POS_TOL, YAW_TOL = 2.0, np.deg2rad(15)
ARC_MIN = 20.0         # metres of path travelled between the two visits

def aliasing(eps, tm):
    n_frames = n_alias = 0
    div_alias, div_rand, gt_mag = [], [], []
    per_task = {}
    for e in eps:
        st, T = e["st"], len(e["st"])
        if T <= K + 5: continue
        step = np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1)
        arc  = np.concatenate([[0.0], np.cumsum(step)])
        fut  = st[K:, :3] - st[:-K, :3]          # future K-step displacement
        M    = len(fut)
        n_frames += M
        gt_mag.append(np.linalg.norm(fut, axis=1))
        task = tm.get(e["ti"], "?")
        tkey = ' '.join(re.sub(r'\d+(\.\d+)?', 'N', task).split()[:4])
        hit = 0
        for t in range(0, M, 3):                 # stride 3 for speed
            d   = np.linalg.norm(st[:M, :3] - st[t, :3], axis=1)
            dy  = np.abs(wrap(st[:M, 3] - st[t, 3]))
            far = np.abs(arc[:M] - arc[t]) > ARC_MIN
            m   = (d < POS_TOL) & (dy < YAW_TOL) & far
            if m.any():
                hit += 1
                div_alias.append(np.linalg.norm(fut[m] - fut[t], axis=1).mean())
        n_alias += hit * 3
        per_task.setdefault(tkey, [0, 0])
        per_task[tkey][0] += hit * 3; per_task[tkey][1] += M
        # random same-instruction control pairs
        r = np.random.default_rng(e["idx"]).integers(0, M, size=min(40, M))
        div_rand.append(np.linalg.norm(fut[r] - fut[r[::-1]], axis=1).mean())
    gt = np.concatenate(gt_mag)
    print(f"\n===== observation aliasing (K={K} steps, tol {POS_TOL} m / 15 deg, "
          f"arc > {ARC_MIN} m) =====")
    print(f"  frames analysed                     : {n_frames}")
    print(f"  frames with an aliased revisit      : {n_alias}  ({100*n_alias/max(n_frames,1):.1f}%)")
    print(f"  GT |future disp| over K             : {gt.mean():.3f} m")
    if div_alias:
        da = np.mean(div_alias)
        print(f"  divergence at ALIASED pose pairs    : {da:.3f} m  "
              f"({100*da/gt.mean():.0f}% of GT magnitude)")
    print(f"  divergence at RANDOM same-ep pairs  : {np.mean(div_rand):.3f} m  (control)")
    print(f"\n  --- aliasing rate by task ---")
    for k, (a, m) in sorted(per_task.items(), key=lambda x: -x[1][0]/max(x[1][1],1)):
        if m > 500: print(f"    {100*a/m:5.1f}%   {k}   (n={m})")

tm, eps = load("train", 600)
if eps:
    check_action_convention(eps)
    aliasing(eps, tm)
print("ALIAS_OK", flush=True)