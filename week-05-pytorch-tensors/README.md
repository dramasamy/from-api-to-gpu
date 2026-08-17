# Week 5 — PyTorch tensors

Reproducible lab for Week 5 of *From API to GPU*. It shows what a tensor is
(rank, shape, dtype, bytes, device), then measures matrix-multiply speed and
memory for FP32, FP16, and BF16 on the GPU, plus a CPU-versus-GPU comparison.

> Blog post: *From API to GPU, Week 5: Tensors, the Data Structure Behind Every
> Model* (see `blogs/week-05-blog.md` in the parent repository).
>
> Roadmap: [`../roadmap/week-05.md`](../roadmap/week-05.md)

## Prerequisites

- An NVIDIA GPU box reachable over SSH as `spark`.
- A Python environment with CUDA-enabled PyTorch. This lab reuses the Week 1
  environment `~/venvs/w1` (torch 2.13.0+cu130). Confirm with:

  ```bash
  ssh spark '~/venvs/w1/bin/python -c "import torch; \
      print(torch.__version__, torch.cuda.is_available())"'
  ```

## Files

| File | Purpose |
| ---- | ------- |
| `tensor_basics.py` | Prints rank, shape, dtype, bytes, device; element-wise, broadcast, transfer. |
| `benchmark_matmul.py` | Times FP32/FP16/BF16 matmul and compares CPU with GPU. |
| `requirements.txt` | Notes the pinned PyTorch build used. |
| `results.json` | Stores the captured measurements. |
| `observations.md` | Summarizes what the numbers show. |
| `system-report.md` | Records the verified runtime and device facts. |
| `model-run.yaml` | Records the environment and benchmark command. |
| `troubleshooting.md` | Covers CUDA timing and out-of-memory caveats. |

## Run

```bash
ssh spark '~/venvs/w1/bin/python -' < tensor_basics.py
ssh spark '~/venvs/w1/bin/python - --size 4096 --iters 20' < benchmark_matmul.py
```

FP16 and BF16 are both 2 bytes but split those bits differently. Read the range
(largest value) and step size (smallest gap near 1) straight from PyTorch:

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
for name, dt in (("fp32", torch.float32),
                 ("fp16", torch.float16),
                 ("bf16", torch.bfloat16)):
    fi = torch.finfo(dt)
    print(f"{name:5} max={fi.max:.3e} smallest_step_near_1={fi.eps:.3e}")
PY'
```

To save machine-readable results, run the benchmark on the Spark with
`--output results.json` and copy the file back, or run it in a local CUDA
environment. `results.json` in this lab is the captured run, lightly annotated.

GPU timings vary run to run. The benchmark reports the median of many iterations
after a warmup, which is why the numbers are stable enough to compare.
