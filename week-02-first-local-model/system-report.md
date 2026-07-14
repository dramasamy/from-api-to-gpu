# System report

Captured on 2026-07-14 at 04:26 UTC.

```text
$ uname -m
aarch64

$ ollama --version
ollama version is 0.31.2

$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVIDIA GB10, 580.159.03

$ free -h
               total        used        free      shared  buff/cache   available
Mem:           121Gi       7.4Gi        47Gi        43Mi        67Gi       114Gi
Swap:           15Gi        96Ki        15Gi

$ ~/venvs/w1/bin/python -c \
    'import torch; print(torch.__version__, torch.version.cuda)'
2.13.0+cu130 13.0
```

The benchmark measurements are stored in `results.json`.
