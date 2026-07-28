import json, re, time, os, io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests, pyarrow.parquet as pq

EP  = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
RID = "yu781986168/HUGE_Dataset_v0"
K, COLS = 8, ["actions","state","task_index","frame_index","env_id"]
S = requests.Session()
S.headers["User-Agent"] = "hugeqc/1.0"

def url(rel): return f"{EP}/datasets/{RID}/resolve/main/{rel}"

def get_text(rel):
    for _ in range(5):
        try:
            r = S.get(url(rel), timeout=60); r.raise_for_status(); return r.text
        except Exception as e:
            last=e; time.sleep(2)
    raise last

def meta(split):
    tm = {j["task_index"]: j["task"] for j in
          (json.loads(l) for l in get_text(f"{split}/meta/tasks.jsonl").splitlines() if l.strip())}
    eps = [json.loads(l) for l in get_text(f"{split}/meta/episodes.jsonl").splitlines() if l.strip()]
    return tm, eps

class HttpFile(io.RawIOBase):
    """minimal seekable HTTP file so pyarrow fetches only the columns it needs"""
    def __init__(self, u):
        self.u = u; self.pos = 0
        r = S.head(u, allow_redirects=True, timeout=60); r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else (self.pos+off if whence==1 else self.size+off)
        return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        for _ in range(4):
            try:
                r = S.get(self.u, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=90)
                r.raise_for_status(); b = r.content; self.pos += len(b); return b
            except Exception: time.sleep(2)
        raise IOError("range read failed")

def read_ep(args):
    split, idx = args
    rel = f"{split}/data/chunk-{idx//1000:03d}/episode_{idx:06d}.parquet"
    try:
        t = pq.read_table(io.BufferedReader(HttpFile(url(rel)), 1<<20), columns=COLS)
        return (np.array(t["actions"].to_pylist(), dtype=np.float32),
                np.array(t["state"].to_pylist(),   dtype=np.float32),
                int(t["task_index"][0].as_py()), str(t["env_id"][0].as_py()))
    except Exception as e:
        return None

def load(split, n):
    tm, eps = meta(split)
    idxs = [e["episode_index"] for e in eps]
    rng = np.random.default_rng(0)
    if n < len(idxs): idxs = [idxs[i] for i in sorted(rng.choice(len(idxs), n, replace=False))]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        out = [r for r in ex.map(read_ep, [(split,i) for i in idxs]) if r is not None]
    print(f"[{split}] {len(out)}/{len(idxs)} episodes in {time.time()-t0:.0f}s", flush=True)
    return tm, out

def chunks(tm, eps):
    Xs, Y, I, E = [], [], [], []
    for acts, st, ti, env in eps:
        if len(acts) <= K: continue
        cum = np.cumsum(acts, axis=0)
        for t in range(len(acts)-K):
            Y.append(cum[t+K]-cum[t]); Xs.append(st[t]); I.append(tm.get(ti,"?")); E.append(env)
    return (np.array(Xs,dtype=np.float32), np.array(Y,dtype=np.float32),
            np.array(I,dtype=object), np.array(E,dtype=object))

res = {}
for split, n, pre in [("train",700,"tr"),("test_seen",250,"ts"),("test_unseen",250,"tu")]:
    tm, eps = load(split, n)
    s,y,i,e = chunks(tm, eps)
    res[f"{pre}_s"],res[f"{pre}_y"],res[f"{pre}_i"],res[f"{pre}_e"] = s,y,i,e
    print(f"  -> {len(y)} frames, GT |disp|={np.linalg.norm(y[:,:3],axis=1).mean():.3f} m", flush=True)
np.savez("/root/autodl-tmp/tmp/huge_qc.npz", **res)
print("COLLECT_OK", flush=True)