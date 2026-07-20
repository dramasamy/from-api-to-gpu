# Week 4: Parameters, weights, and memory

Reproducible lab for Week 4 of *From API to GPU*. It turns a parameter count and
a precision into a weight-memory estimate, checks that estimate against a real
model file, and measures how runtime memory grows past the weights.

> Blog post: *From API to GPU, Week 4: What Model Parameters Actually Cost in
> Memory* (draft, publication link forthcoming).
>
> Roadmap: [`../roadmap/week-04.md`](../roadmap/week-04.md)

## Prerequisites

- Any macOS or Linux machine with Python 3.10 or newer for the calculator.
- An SSH target named `spark` running Ollama for the real memory measurements.
- No extra Python packages are required. The calculator uses the standard
  library only.

## Files

| File | Purpose |
| ---- | ------- |
| `model_memory.py` | Estimates weight memory from a parameter count and precision. |
| `quantized-file-size.md` | Verifies why a Q4 package exceeds four bits per parameter. |
| `requirements.txt` | Notes that no third-party packages are needed. |
| `results.json` | Stores the captured measurements. |
| `observations.md` | Summarizes what the numbers show. |
| `system-report.md` | Records the verified machine and model facts. |
| `model-run.yaml` | Records the model and measurement context. |
| `troubleshooting.md` | Covers unified memory and quantized-size caveats. |

## Run

The calculator runs anywhere:

```bash
python3 model_memory.py --parameters 70 --precision int4
python3 model_memory.py --count 3085938688 --precision bf16
```

The second example uses the Qwen count and precision reported by the Hugging
Face API. These direct checks reproduce both source fields and the Safetensors
index total:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct
REV=aa8e72537993ba99e69dfaafa59ed015b17504d1
curl -s "https://huggingface.co/api/models/$MODEL/revision/$REV" | jq '{
  revision: .sha,
  parameters: .safetensors.total,
  precision_groups: .safetensors.parameters
}'
curl -sL "https://huggingface.co/$MODEL/raw/$REV/model.safetensors.index.json" \
  | jq '.metadata.total_size'
```

Connect to the Spark once, then run the runtime comparison directly. Each
generation request unloads the existing model first so the requested context
takes effect:

```bash
ssh spark
ollama list | awk 'NR == 1 || $1 == "llama3.2:3b"'

ollama stop llama3.2:3b >/dev/null 2>&1 || true
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' \
  http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:3b","prompt":"hi","stream":false,
  "options":{"num_ctx":4096,"num_predict":1}}'
ollama ps

ollama stop llama3.2:3b >/dev/null 2>&1 || true
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' \
  http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:3b","prompt":"hi","stream":false,
  "options":{"num_ctx":16384,"num_predict":1}}'
ollama ps
```

The weight formula is exact for one uniform precision. Mixed quantization such
as `Q4_K_M` uses slightly more than its nominal bit count, so the clean estimate
is a lower bound for those files. See
[`quantized-file-size.md`](quantized-file-size.md) for the complete
manifest-to-blob verification.
