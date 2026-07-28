#!/usr/bin/env python
"""Fetch HUGE-Bench assets in the order the experiment ladder needs them.

Order matters: train is the long pole (~6 h at observed throughput) and blocks
W2, so it goes first. The remaining 3DGS tars only gate closed-loop eval, and
1_office is already on disk, which is enough for the E1.2 render-fidelity gate.

Never fetches HUGE_PI0 — we cite pi0's published numbers instead of running it,
and its params + train_state are 40 GB of nothing.
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

ROOT = "/root/autodl-tmp/huge/dl"
DATA = "yu781986168/HUGE_Dataset_v0"
ENVS = "yu781986168/3DGS_Mesh_Envs"

JOBS = [
    ("train",        DATA, ["train/data/**", "train/meta/**"]),
    ("envs_rest",    ENVS, [f"archives/3DGS_Mesh_Envs_{n}.tar" for n in
                            ("2_city", "3_road", "4_lake", "no1_building",
                             "no3_door", "overhead_bridge")]),
    ("test_unseen",  DATA, ["test_unseen/data/**", "test_unseen/meta/**"]),
    ("test_seen",    DATA, ["test_seen/data/**", "test_seen/meta/**"]),
]


def free_gb() -> float:
    st = os.statvfs("/root/autodl-tmp")
    return st.f_bavail * st.f_frsize / 1e9


def run(name, repo, pats, tries=6):
    for k in range(tries):
        try:
            print(f"[{name}] attempt {k + 1}/{tries}  free={free_gb():.0f} GB",
                  flush=True)
            snapshot_download(repo_id=repo, repo_type="dataset",
                              allow_patterns=pats,
                              local_dir=f"{ROOT}/{repo.split('/')[-1]}",
                              max_workers=8)
            print(f"[{name}] OK  free={free_gb():.0f} GB", flush=True)
            return True
        except Exception as e:
            print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(20)
    return False


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for name, repo, pats in JOBS:
        if only and name not in only:
            continue
        if free_gb() < 15:
            print(f"[abort] only {free_gb():.0f} GB left, refusing to continue",
                  flush=True)
            break
        run(name, repo, pats)
    print("FETCH_DONE", flush=True)
