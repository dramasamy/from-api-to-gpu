# Week 3 — Understand model files and Hugging Face

Reproducible lab for Week 3 of *From API to GPU*. It inspects a Hugging Face
repository without downloading the 6.17 GB model weights, then compares the base
and instruction-tuned checkpoints.

> Blog post: *From API to GPU, Week 3: Reading a Hugging Face Model Repository*
>
> Roadmap: [`../roadmap/week-03.md`](../roadmap/week-03.md)

## Prerequisites

- A Linux machine reachable over SSH as `spark`.
- Python 3.12 or newer with `venv` support.
- Internet access to the public Hugging Face Hub.
- No Hugging Face token is required. A read-only `HF_TOKEN` raises API rate
  limits if you already have one.

## Files

| File | Purpose |
| ---- | ------- |
| `setup.sh` | Creates `~/venvs/w3` and installs the pinned Hub client. |
| `model-inspector.py` | Prints architecture, tokenizer, license, and shard metadata. |
| `compare-models.py` | Compares the Qwen base and instruct repositories. |
| `requirements.txt` | Pins the Python dependency. |
| `results.json` | Stores the inspected repository data. |
| `observations.md` | Summarizes what the commands showed. |
| `system-report.md` | Records the verified runtime and machine facts. |
| `model-run.yaml` | Records the selected model and inspection command. |
| `troubleshooting.md` | Covers downloads, ARM64, and Hub rate limits. |

## Run

From the parent repository:

```bash
ssh spark 'bash -s' < public/week-03-model-files/setup.sh
ssh spark '~/venvs/w3/bin/python -' \
  < public/week-03-model-files/model-inspector.py
ssh spark '~/venvs/w3/bin/python -' \
  < public/week-03-model-files/compare-models.py
```

To inspect another public repository:

```bash
ssh spark '~/venvs/w3/bin/python - --model google/gemma-3-1b-it' \
  < public/week-03-model-files/model-inspector.py
```

The inspector downloads only small metadata files. It lists weight sizes and
hashes through the Hub API without downloading the Safetensors shards.
