# System report

Captured on 2026-07-14.

```text
$ uname -m
aarch64

$ ~/venvs/w3/bin/python --version
Python 3.12.3

$ ~/venvs/w3/bin/python -m pip show huggingface-hub | grep -E '^(Name|Version):'
Name: huggingface_hub
Version: 1.23.0
```

The scripts ran on an NVIDIA DGX Spark. This metadata-only lab did not load a
model or use the GPU.
