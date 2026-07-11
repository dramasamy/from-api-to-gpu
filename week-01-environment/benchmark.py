#!/usr/bin/env python3
"""Week 1 - CPU vs GPU matmul benchmark, writes results.json.

Run on the DGX Spark inside the venv from setup.sh:
    ~/venvs/w1/bin/python benchmark.py

Measures average time for an NxN float32 matrix multiply on CPU and GPU.
Uses GPU warmup + cuda.synchronize() so timings are accurate (CUDA kernels
launch asynchronously, so you must synchronize before stopping the clock).
"""
import json
import time

import torch

N = 4096
ITERS = 20


def bench(a: torch.Tensor, b: torch.Tensor, sync: bool = False) -> float:
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = a @ b  # noqa: F841
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main() -> None:
    a = torch.rand(N, N)
    b = torch.rand(N, N)

    cpu_s = bench(a, b)
    result = {
        "matrix": f"{N}x{N}",
        "dtype": "float32",
        "iters": ITERS,
        "cpu_ms": round(cpu_s * 1000, 2),
    }

    if torch.cuda.is_available():
        ag, bg = a.cuda(), b.cuda()
        for _ in range(3):  # warmup: pay one-time kernel/load cost first
            _ = ag @ bg
        gpu_s = bench(ag, bg, sync=True)
        flops = 2 * N**3
        result.update(
            {
                "gpu_ms": round(gpu_s * 1000, 2),
                "cpu_gflops": round(flops / cpu_s / 1e9, 1),
                "gpu_gflops": round(flops / gpu_s / 1e9, 1),
                "speedup": round(cpu_s / gpu_s, 1),
                "device": torch.cuda.get_device_name(0),
            }
        )

    print(json.dumps(result, indent=2))
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
