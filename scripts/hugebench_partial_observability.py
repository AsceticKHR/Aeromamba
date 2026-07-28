"""Is HUGE-Bench partially observable under its own observation protocol?

Two independent halves; memory is required only if BOTH hold:
  H1  the stage label carries action information beyond the observation
  H2  the stage label cannot be inferred from the observation

Observation O = (env_id, instruction, initial pose, current pose).  The scene is
static, so O is exactly what the official "first frame + current frame + text"
input encodes -- no images needed.
"""
import json, io, time, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests, pyarrow.parquet as pq

EP, RID = "https://hf-mirror.com", "yu781986168/HUGE_Dataset_v0"
K, KNN = 20, 8
S = requests.Session()
def url(rel): return f"{EP}/datasets/{RID}/resolve/main/{rel}"

class HttpFile(io.RawIOBase):
    def __init__(self, u):
        self.u, self.pos = u, 0
        r = S.head(u, allow_redirects=True, timeout=60); r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos+o if w == 1 else self.size+o); return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size: return 0
        end = min(self.pos+n, self.size) - 1
        for _ in range(4):
            try:
                r = S.get(self.u, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=90)
                r.raise_for_status(); d = r.content; b[:len(d)] = d
                self.pos += len(d); return len(d)
            except Exception: time.sleep(2)
        raise IOError("range read failed")

def read_ep(idx):
    rel = f"train/data/chunk-{idx//1000:03d}/episode_{idx:06d}.parquet"
    try:
        t = pq.read_table(io.BufferedReader(HttpFile(url(rel)), 1 << 20), columns=["state"])
        return idx, np.array(t["state"].to_pylist(), dtype=np.float64)
    except Exception:
        return idx, None

# ---- stage sidecar ----
segs = collections.defaultdict(list)
for l in open("/root/autodl-tmp/tmp/stage_segments_train.jsonl", encoding="utf-8"):
    r = json.loads(l); segs[r["episode_index"]].append((r["frame_start"], r["frame_end"], r["subtask_id"]))
emap = {}
for l in open("/root/autodl-tmp/tmp/episode_mapping_train.jsonl", encoding="utf-8"):
    r = json.loads(l); emap[r["episode_index"]] = r

groups = collections.defaultdict(list)
for i, r in emap.items(): groups[(r["env_id"], r["instruction"])].append(i)
# dense, genuinely multi-stage groups
sel = [(k, v) for k, v in groups.items()
       if len(v) >= 16 and np.median([emap[i]["num_stages"] for i in v]) >= 3]
sel.sort(key=lambda kv: -len(kv[1]))
sel = sel[:14]
need = sorted({i for _, v in sel for i in v})
print(f"groups={len(sel)}  episodes={len(need)}", flush=True)

t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    ST = dict(ex.map(read_ep, need))
ST = {k: v for k, v in ST.items() if v is not None}
print(f"fetched {len(ST)}/{len(need)} in {time.time()-t0:.0f}s", flush=True)

def phases(idx, T):
    p = np.full(T, -1, dtype=int)
    for a, b, s in segs.get(idx, []): p[a:min(b, T-1)+1] = s
    return p

W = np.array([1.0, 1.0, 1.0, 10.0])       # yaw radians -> ~10 m equivalent
rows = []
for (env, ins), idxs in sel:
    F = []
    for i in idxs:
        st = ST.get(i)
        if st is None or len(st) <= K + 2: continue
        T = len(st); ph = phases(i, T)
        fut = st[K:, :3] - st[:-K, :3]
        for t in range(T - K):
            if ph[t] < 0: continue
            F.append((i, st[0], st[t], fut[t], ph[t]))
    if len(F) < 400: continue
    epi = np.array([f[0] for f in F])
    X   = np.concatenate([np.array([f[1] for f in F]) * W,
                          np.array([f[2] for f in F]) * W], axis=1)
    Y   = np.array([f[3] for f in F]); P = np.array([f[4] for f in F])
    rng = np.random.default_rng(0)
    q   = rng.choice(len(F), size=min(1500, len(F)), replace=False)
    eO, eOS, accS, hit = [], [], [], 0
    for j in q:
        other = epi != epi[j]                      # exclude the query's own episode
        if other.sum() < KNN * 3: continue
        d  = np.linalg.norm(X[other] - X[j], axis=1)
        oi = np.where(other)[0]
        nn = oi[np.argsort(d)[:KNN]]
        eO.append(np.linalg.norm(Y[nn].mean(0) - Y[j]))
        vote = collections.Counter(P[nn]).most_common(1)[0][0]
        accS.append(vote == P[j])
        same = nn[P[nn] == P[j]]
        if len(same) == 0:                         # fall back to same-stage in group
            cand = oi[P[oi] == P[j]]
            if len(cand) == 0: continue
            same = cand[np.argsort(np.linalg.norm(X[cand] - X[j], axis=1))[:KNN]]
        eOS.append(np.linalg.norm(Y[same].mean(0) - Y[j])); hit += 1
    if not eO: continue
    maj = collections.Counter(P).most_common(1)[0][1] / len(P)
    rows.append(dict(env=env, ins=ins, n=len(F), nst=len(set(P.tolist())),
                     gt=float(np.linalg.norm(Y, axis=1).mean()),
                     eO=float(np.mean(eO)), eOS=float(np.mean(eOS)),
                     acc=float(np.mean(accS)), maj=float(maj)))
    r = rows[-1]
    print(f"  [{r['nst']}st n={r['n']:6d}] E(O)={r['eO']:6.2f}  E(O,S)={r['eOS']:6.2f}  "
          f"drop={100*(1-r['eOS']/r['eO']):5.1f}%  acc(S|O)={r['acc']:.3f} (chance {r['maj']:.3f})"
          f"  {r['ins'][:52]}", flush=True)

print("\n================ AGGREGATE ================")
if rows:
    w  = np.array([r["n"] for r in rows], dtype=float); w /= w.sum()
    eO = float(np.sum(w*[r["eO"] for r in rows])); eOS = float(np.sum(w*[r["eOS"] for r in rows]))
    ac = float(np.sum(w*[r["acc"] for r in rows])); mj = float(np.sum(w*[r["maj"] for r in rows]))
    gt = float(np.sum(w*[r["gt"] for r in rows]))
    print(f"  GT |future displacement| over K={K}     : {gt:.2f} m")
    print(f"  H1  error WITHOUT stage   E(O)         : {eO:.2f} m")
    print(f"      error WITH stage      E(O,S)       : {eOS:.2f} m")
    print(f"      -> stage buys                      : {100*(1-eOS/eO):.1f}% error reduction")
    print(f"  H2  stage inferable from observation   : {100*ac:.1f}%  (chance {100*mj:.1f}%)")
    print(f"      -> residual stage uncertainty      : {100*(1-ac):.1f}%")
print("STAGE_TEST_DONE", flush=True)