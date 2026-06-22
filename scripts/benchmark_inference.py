"""
AeroMamba Inference Latency Benchmark

Measures single-step inference latency for different Mamba backbone sizes.
Useful for selecting the right model variant for real-time UAV control.

Real-time requirement: < 100 ms per control step (10 Hz).

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --mamba_type mamba-370m --n_runs 100
    python scripts/benchmark_inference.py --all   # benchmark all variants
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA, MAMBA_PRESETS


# ─────────────────────────────────────────────────────────────────────────────

def benchmark_single(
    mamba_type:  str = "mamba-130m",
    vision_type: str = "siglip_so_384",
    batch_size:  int = 1,
    n_warmup:    int = 5,
    n_runs:      int = 50,
) -> dict:
    """
    Benchmark inference latency for one (mamba_type, vision_type) combination.

    Returns a result dict with timing statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Loading {mamba_type} + {vision_type} …", flush=True)
    model = AeroMambaVLA(
        mamba_type=mamba_type,
        vision_type=vision_type,
        chunk_size=5,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # ── Dummy inputs (single-encoder path) ────────────────────────────────────
    pixels = torch.randn(batch_size, 3, 384, 384, device=device)
    ids    = torch.randint(0, 50277, (batch_size, 64), device=device)
    prop   = torch.randn(batch_size, 4, device=device)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    for _ in range(n_warmup):
        with torch.inference_mode():
            model(pixels, ids, prop)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Timed runs ────────────────────────────────────────────────────────────
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(pixels, ids, prop)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    latencies_t = torch.tensor(latencies)
    mean_ms  = latencies_t.mean().item()
    std_ms   = latencies_t.std().item()
    p50_ms   = latencies_t.quantile(0.50).item()
    p95_ms   = latencies_t.quantile(0.95).item()

    # ── GPU memory ────────────────────────────────────────────────────────────
    gpu_mb = 0.0
    if device.type == "cuda":
        gpu_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        torch.cuda.reset_peak_memory_stats(device)

    # ── Verdict ───────────────────────────────────────────────────────────────
    realtime_ok = mean_ms < 100.0

    result = {
        "mamba_type":  mamba_type,
        "vision_type": vision_type,
        "device":      str(device),
        "params_M":    total_params,
        "mean_ms":     mean_ms,
        "std_ms":      std_ms,
        "p50_ms":      p50_ms,
        "p95_ms":      p95_ms,
        "gpu_mb":      gpu_mb,
        "realtime_ok": realtime_ok,
        "action_shape": tuple(out["action"].shape),
    }

    _print_result(result)
    return result


def _print_result(r: dict) -> None:
    status = "✅" if r["realtime_ok"] else "⚠️ "
    print(f"\n  {'─' * 56}")
    print(f"  Config : {r['mamba_type']} + {r['vision_type']}")
    print(f"  Device : {r['device']}   Params: {r['params_M']:.0f}M")
    print(f"  Latency: mean={r['mean_ms']:.1f}ms  std={r['std_ms']:.1f}ms  "
          f"p50={r['p50_ms']:.1f}ms  p95={r['p95_ms']:.1f}ms")
    if r["gpu_mb"] > 0:
        print(f"  VRAM   : {r['gpu_mb']:.0f} MB")
    print(f"  Output : {r['action_shape']}  (batch × K=5 × 4-DOF)")
    print(f"  Status : {status} {'Real-time OK (<100ms)' if r['realtime_ok'] else 'Exceeds 100ms target'}")
    print(f"  {'─' * 56}")


def _print_summary(results: list[dict]) -> None:
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY — Inference Benchmark Results")
    print(f"{'═' * 70}")
    print(f"  {'Model':<20} {'Params':>8} {'Mean(ms)':>10} {'P95(ms)':>10} {'VRAM(MB)':>10}  RT?")
    print(f"  {'─' * 66}")
    for r in results:
        rt = "✅" if r["realtime_ok"] else "❌"
        print(
            f"  {r['mamba_type']:<20} {r['params_M']:>7.0f}M "
            f"{r['mean_ms']:>10.1f} {r['p95_ms']:>10.1f} "
            f"{r['gpu_mb']:>10.0f}  {rt}"
        )
    print(f"{'═' * 70}\n")

    best = min(results, key=lambda x: x["mean_ms"])
    print(f"  💡 Fastest: {best['mamba_type']}  ({best['mean_ms']:.1f} ms avg)")
    suitable = [r for r in results if r["realtime_ok"]]
    if suitable:
        # Pick largest model that still meets real-time requirement
        recommended = max(suitable, key=lambda x: x["params_M"])
        print(f"  💡 Recommended for real-time UAV control: {recommended['mamba_type']}")
    else:
        print(f"  ⚠️  No variant meets <100ms on this device. Consider CPU→GPU upgrade.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AeroMamba Inference Latency Benchmark")
    p.add_argument("--mamba_type",  default="mamba-130m",
                   choices=list(MAMBA_PRESETS.keys()),
                   help="Mamba backbone variant to benchmark")
    p.add_argument("--vision_type", default="siglip_so_384",
                   help="Vision encoder type")
    p.add_argument("--batch",       type=int,   default=1,
                   help="Batch size (default 1 for single-step inference)")
    p.add_argument("--n_runs",      type=int,   default=50,
                   help="Number of timed inference runs")
    p.add_argument("--all",         action="store_true",
                   help="Benchmark all Mamba variants sequentially")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'═' * 60}")
    print(f"  AeroMamba Inference Latency Benchmark")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU (no GPU detected)'}")
    print(f"{'═' * 60}")

    if args.all:
        variants = ["mamba-130m", "mamba-370m", "mamba-790m", "mamba-1.4b"]
        results  = []
        for variant in variants:
            try:
                r = benchmark_single(
                    mamba_type=variant,
                    vision_type=args.vision_type,
                    batch_size=args.batch,
                    n_runs=args.n_runs,
                )
                results.append(r)
            except Exception as e:
                print(f"\n  ⚠️  Skipping {variant}: {e}")
        if results:
            _print_summary(results)
    else:
        benchmark_single(
            mamba_type=args.mamba_type,
            vision_type=args.vision_type,
            batch_size=args.batch,
            n_runs=args.n_runs,
        )
