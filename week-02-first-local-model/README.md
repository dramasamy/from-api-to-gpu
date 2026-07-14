# Week 2 — Run your first model with Ollama

Reproducible lab for Week 2 of *From API to GPU*. These scripts run a local model
with Ollama, call its HTTP API, chat from Python, benchmark latency, and inspect
tokenization. The full explanation lives in the blog post; this README stays
high-level.

> Blog post (concepts + walkthrough): *From API to GPU, Week 2: What Actually
> Happens Behind the API*.
> Curriculum: [`../roadmap/week-02.md`](../roadmap/week-02.md)

## Prerequisites

- An NVIDIA GPU box reachable over SSH as `spark` (see Week 1's README for the
  SSH alias). A cloud GPU instance works too.
- Ollama installed and running on that box. Check with:

  ```bash
  ssh spark 'ollama --version; systemctl is-active ollama'
  ```

- A model pulled. This lab uses `phi4` (14.7B, ~9 GB):

  ```bash
  ssh spark 'ollama pull phi4'
  ```

- The `MODEL` variable in the scripts defaults to `phi4`; override with
  `MODEL=<name> ./generate.sh` to try another model you have pulled.
- Python packages for the tokenizer inspection scripts. From a copy of this lab
  directory on the Spark, install the pinned versions with:

  ```bash
  ~/venvs/w1/bin/python -m pip install -r requirements.txt
  ```

## Files

| File | What it does |
| ---- | ------------ |
| `setup.sh` | Pulls the model used throughout the week. |
| `cli_run.sh` | Sends one prompt through Ollama's terminal-aware CLI. |
| `inspect_model.sh` | Shows the model list, model metadata, and actual Ollama manifest. |
| `generate.sh`      | Calls the `/api/generate` endpoint and pretty-prints the full JSON. |
| `timing.sh`        | Does a cold-load call and prints load time, tokens, and tokens/sec. |
| `temperature.sh`   | Compares two replies at the selected temperature. |
| `chat_client.py`   | Provides one-shot and interactive chat through `/api/chat`. |
| `benchmark.py`     | Measures cold/warm TTFT, total latency, and tokens/sec. |
| `decode_tokens.py` | Uses `tiktoken` to encode the prompt and decode token IDs. |
| `inspect_tokenizer.py` | Reads tokenizer metadata from the GGUF model layer. |
| `requirements.txt` | Pins the Python packages used by the inspection scripts. |
| `results.json`     | Stores the measurements captured on the DGX Spark. |
| `model-run.yaml`   | Records the model, runtime, hardware, and run settings. |
| `system-report.md` | Captures the verified machine and runtime versions. |
| `observations.md` | Summarizes the measured behavior. |
| `troubleshooting.md` | Records the SSH CLI and unified-memory caveats. |

## How to run

These call `localhost:11434`, so they run **on the Spark**. Either SSH in and run
them there, or pipe a script over SSH:

```bash
ssh spark 'bash -s' < setup.sh
ssh spark 'bash -s' < generate.sh
ssh spark 'bash -s' < timing.sh
ssh spark 'bash -s' < inspect_model.sh
ssh spark '~/venvs/w1/bin/python -' < decode_tokens.py
ssh spark '~/venvs/w1/bin/python -' < inspect_tokenizer.py
ssh spark '~/venvs/w1/bin/python - --once "What is a pod?"' < chat_client.py
ssh spark '~/venvs/w1/bin/python - --runs 3' < benchmark.py
```

Ollama's CLI uses terminal-aware output. Run `cli_run.sh` after copying the lab
directory to the Spark, or force a pseudo-terminal for a direct remote call:

```bash
ssh -tt spark \
  'ollama run --nowordwrap phi4 "Reply with exactly: local model ready"'
```

Your numbers (load time, tokens/sec) will differ by machine. That is expected.
