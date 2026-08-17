#!/usr/bin/env python3
"""Week 5 - measure matmul speed and memory for FP32, FP16, and BF16.

Times a square matrix multiply on the GPU for three precisions with warmup and
CUDA synchronization, then compares one CPU run with one GPU run at FP32. All
timings use time.perf_counter around synchronized GPU work. Results print as a
table and optionally save to JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def time_matmul(a: torch.Tensor, b: torch.Tensor, iters: int,
                warmup: int) -> float:
    cuda = a.device.type == "cuda"
    for _ in range(warmup):
        a @ b
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        a @ b
        if cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def gflops(size: int, seconds: float) -> float:
    return (2 * size ** 3) / seconds / 1e9


def main() -> None:
    args = parse_args()
    size = args.size
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    rows = []
    print(f"matmul {size}x{size}, iters={args.iters}, device={device}\n")
    print(f"{'precision':10} {'median_ms':>10} {'gflops':>10} "
          f"{'peak_mib':>10}")
    for name, dtype in DTYPES.items():
        if not has_cuda:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        seconds = time_matmul(a, b, args.iters, args.warmup)
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        row = {
            "precision": name,
            "median_ms": round(seconds * 1000, 3),
            "gflops": round(gflops(size, seconds), 1),
            "peak_mib": round(peak_mib, 1),
        }
        rows.append(row)
        print(f"{name:10} {row['median_ms']:>10.3f} {row['gflops']:>10.1f} "
              f"{row['peak_mib']:>10.1f}")
        del a, b

    print("\n=== CPU versus GPU at FP32 ===")
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    cpu_iters = max(3, args.iters // 4)
    cpu_s = time_matmul(a_cpu, b_cpu, cpu_iters, 1)
    print(f"cpu_fp32   {cpu_s * 1000:10.3f} ms   {gflops(size, cpu_s):8.1f} "
          f"gflops")
    speedup = None
    if has_cuda:
        gpu_fp32 = next(r for r in rows if r["precision"] == "fp32")
        speedup = round(cpu_s * 1000 / gpu_fp32["median_ms"], 1)
        print(f"gpu_fp32   {gpu_fp32['median_ms']:10.3f} ms   "
              f"{gpu_fp32['gflops']:8.1f} gflops")
        print(f"gpu_is     {speedup}x faster than cpu at fp32")

    if args.output:
        payload = {
            "size": size,
            "warmup": args.warmup,
            "gpu_iters": args.iters,
            "cpu_iters": cpu_iters,
            "device": device,
            "precisions": rows,
            "cpu_fp32_ms": round(cpu_s * 1000, 3),
            "gpu_speedup_fp32": speedup,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
