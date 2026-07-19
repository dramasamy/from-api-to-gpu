# Week 3 — Understand model files and Hugging Face

Reproducible lab for Week 3 of *From API to GPU*. The blog post reads a model's
Hugging Face page in a browser; this lab is the **API way**: the same answers
via `curl`, `jq`, and the Hub client, pinned to an exact revision and without
downloading the 6.17 GB model weights. It also compares the base and
instruction-tuned checkpoints by hashing their weight shards.

> Blog post: *From API to GPU, Week 3: Reading a Hugging Face Model Repository*
>
> Roadmap: [`../roadmap/week-03.md`](../roadmap/week-03.md)

## Prerequisites

- Any macOS or Linux machine with internet access, `curl`, and `jq` for direct
  repository inspection.
- An SSH target named `spark` only if you want to reproduce the remote commands
  exactly as shown in the series.
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
| `troubleshooting.md` | Covers downloads, ARM64, Hub rate limits, and API-only snags. |

## Run: the API way

### 1. Overview from the Hub API

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct
curl -s "https://huggingface.co/api/models/$MODEL" | jq '{
  repository: .id,
  revision: .sha,
  base_model: .cardData.base_model,
  license_metadata: .cardData.license,
  architecture: .config.architectures[0],
  parameters: .safetensors.total,
  precision_groups: .safetensors.parameters,
  checkpoint_files: [
    .siblings[].rfilename | select(endswith(".safetensors"))
  ]
}'
```

### 2. Pin a revision and read the raw files

Record the commit so a later change to `main` cannot alter your results, then
read each file directly:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct
REV=$(curl -s "https://huggingface.co/api/models/$MODEL" | jq -r '.sha')
BASE="https://huggingface.co/$MODEL/raw/$REV"

curl -sL "$BASE/config.json"
curl -sL "$BASE/generation_config.json"
curl -sL "$BASE/tokenizer_config.json" | jq -r '.chat_template'
curl -sL "$BASE/model.safetensors.index.json" | jq '.metadata.total_size'
curl -sL "$BASE/LICENSE" | grep -E 'FOR NON-COMMERCIAL PURPOSES ONLY'
```

### 3. Automate with the Python inspector

Run it from the parent repository:

```bash
ssh spark 'bash -s' < public/week-03-model-files/setup.sh
ssh spark '~/venvs/w3/bin/python -' \
  < public/week-03-model-files/model-inspector.py
ssh spark '~/venvs/w3/bin/python -' \
  < public/week-03-model-files/compare-models.py
```

The inspector downloads only small metadata files. It lists weight sizes and
hashes through the Hub API without downloading the Safetensors shards.
`compare-models.py` hashes the first weight shard of each repository to prove the
base and instruct checkpoints differ. The verified target is
`Qwen/Qwen2.5-3B-Instruct`. Other model families may use different files or
configuration keys.
