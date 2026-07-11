#!/usr/bin/env python3
"""Week 1 - prove the GPU is usable from PyTorch.

Run on the DGX Spark inside the venv from setup.sh:
    ~/venvs/w1/bin/python validate_gpu.py
"""
import torch


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)

    if not torch.cuda.is_available():
        print("No CUDA GPU visible to PyTorch.")
        return

    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

    # Put a real tensor on the GPU to prove end-to-end use.
    x = torch.rand(3, 3, device="cuda")
    print("Tensor on:", x.device)
    print("Allocated MiB:", round(torch.cuda.memory_allocated(0) / 1024**2, 2))


if __name__ == "__main__":
    main()
