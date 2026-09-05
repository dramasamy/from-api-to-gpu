# Week 6 — Neural-network basics

Reproducible lab for Week 6 of *From API to GPU*. It trains the smallest useful
neural network, a single linear layer that learns to convert Celsius to
Fahrenheit, so the weight, bias, loss, gradient, optimizer, learning rate,
epoch, and batch become concrete instead of abstract.

> Blog posts (two parts): *From API to GPU, Week 6 (Part 1): A Model That
> Predicts, and How Wrong It Is* and *Week 6 (Part 2): Watching a Neural Network
> Learn* (see `blogs/week-06-part-1-blog.md` and `blogs/week-06-part-2-blog.md`
> in the parent repository). Part 1 covers the forward pass and the loss; Part 2
> covers gradients, the optimizer, and the training loop that `train_tiny.py`
> runs.
>
> Roadmap: [`../roadmap/week-06.md`](../roadmap/week-06.md)

## Prerequisites

- A machine with PyTorch. This lab reuses the Week 1 environment `~/venvs/w1`
  (torch 2.13.0+cu130). It runs on the CPU; no GPU is required.
- Confirm with:

  ```bash
  ssh spark '~/venvs/w1/bin/python -c "import torch; print(torch.__version__)"'
  ```

## Files

| File | Purpose |
| ---- | ------- |
| `train_tiny.py` | Trains Celsius to Fahrenheit, prints the history, and saves the checkpoint. |
| `predict.py` | Loads the saved `.pt` checkpoint and runs one standalone inference. |
| `requirements.txt` | Notes the pinned PyTorch version used. |
| `results.json` | Stores the captured training run (regenerate with `--output`). |
| `observations.md` | Summarizes what the numbers show. |
| `system-report.md` | Records the verified runtime facts. |
| `model-run.yaml` | Records the training configuration. |
| `troubleshooting.md` | Covers input scaling and reproducibility. |

## Run

```bash
ssh spark '~/venvs/w1/bin/python -' < train_tiny.py
```

Training saves `celsius_to_fahrenheit.pt` to the Spark's current directory. Then
load that checkpoint and run one inference (from the same directory, so it finds
the `.pt`):

```bash
ssh spark '~/venvs/w1/bin/python -' < predict.py
```

Regenerate `results.json` (writes to the Spark's current directory):

```bash
ssh spark '~/venvs/w1/bin/python - --output results.json' < train_tiny.py
```

The run is seeded, so the same seed gives the same numbers. The inputs are raw
Celsius, so the printed weight is the model's real parameter. Large raw inputs
make the first gradient large, which is why the learning rate is small (0.0003);
a large rate diverges. If you cloned only the public repo, drop the `public/`
prefix and run `train_tiny.py` from inside this directory.
