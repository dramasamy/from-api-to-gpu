#!/usr/bin/env python3
"""Week 1 - CPU vs GPU matmul benchmark, writes results.json.

Run on the DGX Spark inside the venv from setup.sh:
    ~/venvs/w1/bin/python benchmark.py

Measures average time for an NxN float32 matrix multiply on CPU and GPU.
Uses GPU warmup + cuda.synchronize() so timings are accurate (CUDA kernels
launch asynchronously, so you must synchronize before stopping the clock).

Also records:
- the cold first-call cost (one-time kernel load), and
- TF32 tensor-core matmul for comparison (PyTorch defaults matmul TF32 off,
  so the main number is true FP32).
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
    flops = 2 * N**3

    cpu_s = bench(a, b)
    result = {
        "matrix": f"{N}x{N}",
        "dtype": "float32",
        "iters": ITERS,
        "cpu_ms": round(cpu_s * 1000, 2),
        "cpu_gflops": round(flops / cpu_s / 1e9, 1),
    }

    if torch.cuda.is_available():
        ag, bg = a.cuda(), b.cuda()

        # Cold first call: pays the one-time kernel-load cost.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ag @ bg
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000

        # True FP32 (TF32 disabled; this is PyTorch's default for matmul).
        torch.backends.cuda.matmul.allow_tf32 = False
        for _ in range(3):  # warmup
            _ = ag @ bg
        fp32_s = bench(ag, bg, sync=True)

        # TF32 tensor cores (lower precision, much faster).
        torch.backends.cuda.matmul.allow_tf32 = True
        for _ in range(3):
            _ = ag @ bg
        tf32_s = bench(ag, bg, sync=True)

        result.update(
            {
                "gpu_cold_ms": round(cold_ms, 2),
                "gpu_fp32_ms": round(fp32_s * 1000, 2),
                "gpu_fp32_tflops": round(flops / fp32_s / 1e12, 1),
                "gpu_fp32_speedup": round(cpu_s / fp32_s, 1),
                "gpu_tf32_ms": round(tf32_s * 1000, 2),
                "gpu_tf32_tflops": round(flops / tf32_s / 1e12, 1),
                "gpu_tf32_speedup": round(cpu_s / tf32_s, 1),
                "device": torch.cuda.get_device_name(0),
            }
        )

    print(json.dumps(result, indent=2))
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
