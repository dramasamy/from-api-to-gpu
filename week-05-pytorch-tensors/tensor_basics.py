#!/usr/bin/env python3
"""Week 5 - tensor basics: rank, shape, dtype, bytes, and device.

Shows a scalar, vector, matrix, and 3D tensor, then one element-wise
operation, one broadcast, and one CPU-to-GPU transfer. Every printed size and
byte count comes from PyTorch, not from a hand estimate.
"""

from __future__ import annotations

import torch


def describe(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name:8} rank={tensor.ndim} "
        f"shape={tuple(tensor.shape)} "
        f"dtype={str(tensor.dtype).replace('torch.', '')} "
        f"elem={tensor.element_size()}B "
        f"total={tensor.nbytes}B "
        f"dev={tensor.device}"
    )


def main() -> None:
    scalar = torch.tensor(3.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.zeros(2, 3)
    tensor3d = torch.zeros(2, 3, 4)

    print("=== rank, shape, dtype, bytes, device ===")
    describe("scalar", scalar)
    describe("vector", vector)
    describe("matrix", matrix)
    describe("tensor3d", tensor3d)

    print("\n=== same shape, three precisions ===")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        describe(str(dtype).replace("torch.", ""), torch.zeros(1024, 1024,
                                                                dtype=dtype))

    print("\n=== element-wise operation ===")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    print("a + b =", (a + b).tolist())

    print("\n=== broadcasting ===")
    matrix = torch.ones(2, 3)
    row = torch.tensor([1.0, 2.0, 3.0])
    print("matrix shape", tuple(matrix.shape), "+ row shape", tuple(row.shape))
    print("result:\n", (matrix + row).tolist())

    print("\n=== device transfer ===")
    if torch.cuda.is_available():
        cpu_tensor = torch.ones(3)
        gpu_tensor = cpu_tensor.to("cuda")
        print("cpu_tensor.device", cpu_tensor.device)
        print("gpu_tensor.device", gpu_tensor.device)
    else:
        print("CUDA not available; skipping GPU transfer")


if __name__ == "__main__":
    main()
