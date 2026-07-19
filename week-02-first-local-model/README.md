# Week 2 — Run your first model with Ollama

Reproducible lab for Week 2 of *From API to GPU*. Simple checks are direct shell
commands so you can see and learn the real Ollama interface. Python files are
used only where the program itself is part of the lesson: chat state, streamed
timing, benchmarking, and token inspection.

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

- Pull the model used in this lab directly:

  ```bash
  ssh spark 'ollama pull phi4'
  ```

- Python packages for the tokenizer inspection scripts. From a copy of this lab
  directory on the Spark, install the pinned versions with:

  ```bash
  ~/venvs/w1/bin/python -m pip install -r requirements.txt
  ```

## Files

| File | What it does |
| ---- | ------------ |
| `chat_client.py`   | Provides one-shot and interactive chat through `/api/chat`. |
| `benchmark.py`     | Measures cold/warm TTFT, total latency, and tokens/sec. |
| `decode_tokens.py` | Uses `tiktoken` to encode the prompt and decode token IDs. |
| `inspect_tokenizer.py` | Reads tokenizer metadata from the GGUF model layer. |
| [llama.cpp note](ollama-model-in-llamacpp.md) | Optional cross-runtime investigation. |
| `setup_llamacpp.sh` | Builds the pinned CUDA runtime used by that investigation. |
| `inspect_model.sh` | Verifies Ollama's manifest-to-blob chain for that investigation. |
| `run_ollama_model_llamacpp.sh` | Runs `llama.cpp` against Ollama's exact blob. |
| `requirements.txt` | Pins the Python packages used by the inspection scripts. |
| `results.json`     | Stores the measurements captured on the DGX Spark. |
| `model-run.yaml`   | Records the model, runtime, hardware, and run settings. |
| `system-report.md` | Captures the verified machine and runtime versions. |
| `observations.md` | Summarizes the measured behavior. |
| `troubleshooting.md` | Records the SSH CLI and unified-memory caveats. |

## Direct commands

Open a shell on the Spark. The API commands use `localhost:11434`, which means
they must run on the machine where Ollama is running:

```bash
ssh spark
```

List the local models:

```bash
ollama list
```

Ask Ollama for focused Phi-4 metadata:

```bash
curl -s http://localhost:11434/api/show -d '{"model": "phi4"}' | jq '{
  format: .details.format,
  architecture: .details.family,
  parameters: .details.parameter_size,
  context_length: (.model_info | to_entries
    | map(select(.key | endswith(".context_length"))) | first.value),
  quantization: .details.quantization_level,
  capabilities
}'
```

Call the generate API and inspect its full response:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "phi4",
  "prompt": "Explain Kubernetes scheduling in three sentences.",
  "stream": false
}' | jq
```

Change `temperature` and run the same command twice to compare the answers:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "phi4",
  "prompt": "Give me a one-sentence tagline for a coffee shop.",
  "stream": false,
  "options": {"temperature": 0}
}' | jq -r .response
```

Use a fixed seed to make repeated sampling more comparable under the same
conditions:

```bash
for RUN in 1 2 3 4; do
  printf "run=%s seed=42: " "$RUN"
  curl -s http://localhost:11434/api/generate -d '{
    "model": "phi4",
    "prompt": "Reply with one invented coffee shop name and nothing else.",
    "stream": false,
    "options": {"temperature": 1.2, "seed": 42, "num_predict": 16}
  }' | jq -r .response | tr '\n' ' '
  printf "\n"
done
```

Inspect the files in the local Phi-4 package:

```bash
MANIFEST=/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/phi4/latest
jq '{schemaVersion, layers: [.layers[] | {mediaType, digest, size}]}' \
  "$MANIFEST"
```

Check which models are currently loaded and their configured context sizes:

```bash
ollama ps
```

Send a role-tagged chat request. Add earlier `user` and `assistant` messages to
the array when the model needs conversation history:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "phi4",
  "stream": false,
  "messages": [
    {"role": "system", "content": "Reply in one short sentence."},
    {"role": "user", "content": "My cluster is called Atlas."},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "What is my cluster called?"}
  ]
}' | jq .message
```

Exit the Spark shell when you finish the local commands:

```bash
exit
```

Ollama's CLI uses terminal-aware output. For a one-shot remote call, force SSH
to allocate a pseudo-terminal:

```bash
ssh -tt spark \
  'ollama run --nowordwrap phi4 "Reply with exactly: local model ready"'
```

## Python programs

These files contain behavior worth reading and reusing. Run them from the parent
repository by piping the program to the Spark's Week 1 Python environment:

```bash
ssh spark '~/venvs/w1/bin/python -' \
  < public/week-02-first-local-model/decode_tokens.py
ssh spark '~/venvs/w1/bin/python -' \
  < public/week-02-first-local-model/inspect_tokenizer.py
ssh spark '~/venvs/w1/bin/python - --once "What is a pod?"' \
  < public/week-02-first-local-model/chat_client.py
ssh spark '~/venvs/w1/bin/python - --runs 3' \
  < public/week-02-first-local-model/benchmark.py
```

## Optional llama.cpp investigation

Read [`ollama-model-in-llamacpp.md`](ollama-model-in-llamacpp.md) before running
this path. The shell scripts remain here because building a pinned CUDA runtime,
resolving a content-addressed model file, and extracting verbose runtime evidence
are genuinely multi-step operations:

This optional path needs Git, CMake, a C++ compiler, and the CUDA toolkit. The
setup script builds a pinned commit under the current user's home directory and
does not need root access.

```bash
ssh spark 'bash -s' \
  < public/week-02-first-local-model/setup_llamacpp.sh
ssh spark 'bash -s' \
  < public/week-02-first-local-model/inspect_model.sh
ssh spark 'bash -s' \
  < public/week-02-first-local-model/run_ollama_model_llamacpp.sh
```

Your numbers (load time, tokens/sec) will differ by machine. That is expected.
