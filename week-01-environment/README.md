# Week 1 — DGX Spark and GPU environment

Reproducible lab for Week 1 of *From API to GPU*. This directory holds the
**commands and scripts** to inventory a DGX Spark and validate its GPU. The
full explanation of every concept (CPU vs GPU, unified memory, driver vs CUDA
runtime vs toolkit, ARM64) lives in the blog post — this README stays high-level
so nothing is duplicated.

> Blog post (concepts + walkthrough): *From API to GPU, Week 1: Understanding My
> NVIDIA DGX Spark Environment*.
> Curriculum: [`../roadmap/week-01.md`](../roadmap/week-01.md)

## What this week does

- Take a complete machine inventory (CPU, memory, storage, GPU, driver, CUDA).
- Learn to read `nvidia-smi` as an inventory tool.
- Understand the DGX Spark's unified-memory architecture.
- (Next) Install CUDA-enabled PyTorch and prove the GPU is usable from Python.

## Prerequisites

Reach your DGX Spark as `spark` over SSH so every command below is copy-pasteable.
In `~/.ssh/config` on your control machine:

```text
Host spark
    HostName <spark-ip-or-hostname>
    User <your-user>
```

Verify before running anything else:

```bash
ssh spark 'nvidia-smi'
```

## Files

| File | What it does |
| ---- | ------------ |
| `inventory.sh`     | Runs the full machine inventory on the Spark over SSH. |
| `system-report.md` | Captured real output from my run (the Week 1 deliverable). |
| `setup.sh`         | Creates an isolated venv on the Spark and installs PyTorch. |
| `requirements.txt` | Pinned Python deps (`torch`, `numpy`). |
| `validate_gpu.py`  | Proves the GPU is usable from PyTorch. |
| `benchmark.py`     | CPU-vs-GPU matmul benchmark -> `results.json`. |
| `results.json`     | Measured benchmark numbers from my run. |

## How to reproduce

Inventory (from your control machine; needs the `spark` SSH alias):

```bash
./inventory.sh
```

GPU validation + benchmark (on the Spark, inside a venv):

```bash
ssh spark                       # log into the Spark
git clone <this-repo> && cd from-api-to-gpu/week-01-environment
./setup.sh                      # creates ~/venvs/w1 and installs torch
~/venvs/w1/bin/python validate_gpu.py
~/venvs/w1/bin/python benchmark.py
```

Compare your output to `system-report.md` and `results.json`. Values (GPU model,
driver, memory, speedup) will differ by machine — that's expected.
