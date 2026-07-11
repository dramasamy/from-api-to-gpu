# Week 1 — DGX Spark system report

Captured output from `./inventory.sh` on my DGX Spark (2026-07-11). This is a
results artifact, not a tutorial — see the Week 1 blog post for what each value
means and why it matters.

Reproduce with:

```bash
./inventory.sh          # needs an SSH alias `spark` (see README.md)
```

## Summary

| Component | Value |
| --------- | ----- |
| OS        | Ubuntu 24.04.4 LTS (DGX OS), kernel 6.17.0-1026-nvidia |
| Arch      | aarch64 (ARM64) |
| CPU       | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| Memory    | 121 GiB unified (CPU + GPU shared) |
| Storage   | 3.7 TB NVMe |
| GPU       | NVIDIA GB10 (Grace-Blackwell) |
| Driver    | 580.159.03 |
| CUDA      | 13.0 (toolkit 13.0.88 at /usr/local/cuda-13; nvcc not on PATH) |
| Python    | 3.12.3 (system; PyTorch installed in a venv, see below) |
| Docker    | 29.2.1 (arm64; daemon needs group/sudo) |

## Captured output (trimmed)

```text
$ uname -a
Linux spark-66b9 6.17.0-1026-nvidia ... aarch64 aarch64 aarch64 GNU/Linux

$ cat /etc/os-release
PRETTY_NAME="Ubuntu 24.04.4 LTS"   VERSION_CODENAME=noble

$ lscpu
Architecture: aarch64   CPU(s): 20
Model name: Cortex-X925 (10)   Model name: Cortex-A725 (10)

$ free -h
Mem: 121Gi total   6.2Gi used   115Gi available

$ lsblk
nvme0n1 3.7T   |-p1 298M /boot/efi   |-p2 3.7T /

$ nvidia-smi
NVIDIA-SMI 580.159.03   Driver 580.159.03   CUDA Version: 13.0
GPU 0: NVIDIA GB10   35C   P8   4W   Memory-Usage: Not Supported   0% util

$ nvcc --version            # not on PATH
bash: nvcc: command not found
$ /usr/local/cuda/bin/nvcc --version
Cuda compilation tools, release 13.0, V13.0.88

$ ls -d /usr/local/cuda*
/usr/local/cuda  /usr/local/cuda-13  /usr/local/cuda-13.0

$ python3 --version
Python 3.12.3
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'   # system Python; use a venv
# after ./setup.sh -> ~/venvs/w1: torch 2.13.0+cu130, CUDA available: True

$ docker version
Client: Docker Engine - Community 29.2.1 (linux/arm64)
permission denied ... /var/run/docker.sock   # daemon needs group/sudo
```

## Notes

- `Memory-Usage: Not Supported` is expected on the Spark — memory is unified,
  so there is no separate VRAM figure to report. NVIDIA documents this as known
  behavior. Track model memory via `free -h` or PyTorch counters instead.
- The CUDA toolkit is installed; `nvcc` just is not on `PATH`. Running models
  never needs `nvcc` (PyTorch ships its own CUDA runtime).
- Deep GPU monitoring (utilization / bandwidth profiling) is deferred to the
  CUDA track (~Week 5).
