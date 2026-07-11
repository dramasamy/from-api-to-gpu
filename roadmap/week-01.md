# Week 1 — DGX Spark and GPU environment

**Phase 1: Become comfortable running local models**

---

## Learn

- What DGX OS provides
- CPU versus GPU
- Unified memory versus traditional discrete VRAM
- NVIDIA driver
- CUDA runtime
- CUDA toolkit
- PyTorch CUDA support
- Containers and NVIDIA container runtime
- GPU utilization, memory usage, temperature, and power
- ARM64 implications for packages and container images

## Hands-on

Collect a complete machine inventory:

```bash
uname -a
cat /etc/os-release
lscpu
free -h
lsblk
nvidia-smi
nvcc --version
python3 --version
docker version
```

Create a Python validation script:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print(
        "Allocated memory:",
        round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "GiB",
    )
```

## Deliverable

`00-environment/system-report.md`

## CUDA track

Start **CUDA Level 1: User** in parallel from around now — see [cuda-track.md](cuda-track.md).

## Reference

- DGX Spark User Guide — https://docs.nvidia.com/dgx/dgx-spark/index.html
