# From API to GPU — Curriculum

A software engineer's hands-on journey from **AI API consumer** to **local LLM systems architect**: local deployment, inference optimization, RAG, agents, fine-tuning, CUDA, and production AI architecture on an NVIDIA DGX Spark.

> This `curriculum/` directory is the **public roadmap** — what I plan to learn and build, week by week. Reproducible code, benchmarks, and hands-on labs will land in top-level folders (one-to-one with each week) as I progress.

---

## Who this plan is for (my starting point)

- Software engineer + TPM. Comfortable with **Linux, Python, Docker, Kubernetes**.
- I use frontier models via API to write code in VS Code.
- I have **never** deployed a model myself.
- New to: ML, PyTorch/TensorFlow, transformers, llama, LM Studio, Ollama, MoE, MTP, quantization internals, CUDA programming.
- Hardware: **NVIDIA DGX Spark** (DGX OS, ARM64, unified memory).

Nothing in this plan assumes prior ML knowledge.

---

## Target role

> An **LLM Systems Architect / AI Infrastructure Engineer** who can select, deploy, optimize, integrate, evaluate, fine-tune, secure, and operate language models.

Five connected skill areas:

1. **Model literacy** — parameters, weights, tensors, tokens, embeddings, attention, transformers, MoE, context, KV cache, quantization.
2. **Inference engineering** — loading models, GPU memory, batching, throughput, latency, serving engines.
3. **Application integration** — APIs, structured output, tool calling, RAG, agents, memory, guardrails, evaluation.
4. **Adaptation** — prompting, RAG, LoRA, QLoRA, SFT, dataset prep, evaluation.
5. **Infrastructure & acceleration** — CUDA, PyTorch, containers, Kubernetes, monitoring, multi-GPU, production architecture.

My Linux/containers/K8s/TPM background is a head start — the goal is to **connect that infrastructure knowledge to model behavior**.

---

## Core learning principle

Every week, run this loop:

> **Run → observe → measure → understand → modify → document**

Example: run a model → measure memory + tokens/sec → change quantization → observe quality/perf → read why it changed → publish the experiment. This makes abstract terms (weights, KV cache, precision, batching) concrete.

**Do not** spend weeks on transformer equations before running anything.

---

## Progression

```text
API consumer
→ local model user
→ model-literate developer
→ inference engineer
→ RAG and agent integrator
→ fine-tuning practitioner
→ LLM platform engineer
→ LLM systems and solution architect
```

## First milestone (by end of Week 4)

> Deploy a local model on DGX Spark, access it via an API, inspect its model config, calculate its theoretical memory footprint, and publish repeatable performance results.

At that point I've crossed from **AI API consumer** to **beginner LLM deployment engineer**.

---

## 32-week schedule (~6–10 hrs/week)

| Phase | Weeks | Theme |
| ----- | ----- | ----- |
| 1 | 1–4 | Comfortable running local models |
| 2 | 5–8 | Enough ML to understand inference |
| 3 | 9–13 | Transformers and model terminology |
| 4 | 14–16 | Quantization and model formats |
| 5 | 17–20 | Inference engineering |
| 6 | 21–24 | Production-style model services |
| 7 | 25–28 | RAG and application integration |
| 8 | 29–32 | Fine-tuning and model adaptation |

A **parallel CUDA track** starts around Week 5 — see [cuda-track.md](cuda-track.md).

Week files: [week-01.md](week-01.md) … [week-32.md](week-32.md).

---

## Repository structure (public repo)

```text
from-api-to-gpu/
├── README.md
├── ROADMAP.md
├── GLOSSARY.md
├── HARDWARE.md
├── BLOG_INDEX.md
├── 00-environment/
├── 01-first-local-model/
├── 02-model-anatomy/
├── 03-pytorch-foundations/
├── 04-transformers/
├── 05-huggingface/
├── 06-quantization/
├── 07-inference-engines/
├── 08-benchmarking/
├── 09-serving/
├── 10-rag/
├── 11-agents/
├── 12-fine-tuning/
├── 13-cuda/
├── 14-production/
├── projects/
│   ├── local-chat-api/
│   ├── model-benchmark-dashboard/
│   ├── robotics-rag-assistant/
│   └── robotics-fine-tuned-model/
├── notebooks/
├── scripts/
├── results/
├── diagrams/
└── docs/
```

**Never** commit large model files. Store only: download scripts, model IDs, configuration, checksums (when needed), benchmark results, and adapter weights (only when licensing and file size permit).

---

## Weekly GitHub folder standard

Every weekly folder should contain the same predictable files:

```text
week-XX-topic/
├── README.md
├── concepts.md
├── setup.sh
├── requirements.txt
├── experiment.py
├── benchmark.py
├── results.json
├── observations.md
├── troubleshooting.md
└── diagrams/
```

Each `README.md` answers:

1. What did I learn?
2. Why does it matter?
3. What did I run?
4. What hardware/software did I use?
5. How can someone reproduce it?
6. What results did I observe?
7. What confused me?
8. What would I test next?
9. What production implications did I identify?

---

## Model strategy

Pick **one model family** and test several sizes within it (isolates parameter-count effects).

| Model class | Purpose |
| ----------- | ------- |
| 1B–4B instruct | Fast experiments, learning mechanics |
| 7B–14B | Realistic local app development |
| 20B–40B | Memory, throughput, serving experiments |
| Larger / MoE | Quantization, offloading, architecture analysis |

Record for every model run (`model-run.yaml`):

```yaml
model:
  repository:
  architecture:
  total_parameters:
  active_parameters:
  precision:
  quantization:
  context_limit:
  license:
runtime:
  engine:
  version:
  command:
  context_configured:
  batch_configured:
hardware:
  device:
  memory:
  driver:
  cuda:
  pytorch:
results:
  load_time_seconds:
  idle_memory_gb:
  peak_memory_gb:
  prompt_tokens:
  output_tokens:
  time_to_first_token_ms:
  tokens_per_second:
```

---

## Portfolio evidence (matters more than certificates)

1. Reproducible local-model benchmark suite.
2. OpenAI-compatible inference service.
3. Production-style Kubernetes deployment.
4. RAG system with measurable retrieval quality.
5. LoRA/QLoRA adapter with before/after evaluation.
6. Profiling report showing CUDA / inference bottlenecks.
7. Complete solution-architecture document.
8. 30+ public technical posts showing progression.

TPM differentiators to weave in: requirements, capacity planning, security, multi-tenancy, SLOs, cost, rollout strategy, evaluation gates, operational readiness, failure handling, governance.

---

## Concepts I should eventually explain without buzzwords

- **Model:** parameter, weight, bias, tensor, layer, checkpoint, architecture, base vs instruct vs reasoning vs multimodal.
- **Transformer:** token, tokenizer, embedding, attention, Q/K/V, transformer block, logit, sampling, temperature, context window, RoPE, GQA/MQA, KV cache.
- **Architecture:** dense, MoE, expert, router, total vs active parameters, MTP, speculative decoding, distillation.
- **Runtime:** inference engine, prefill, decode, TTFT, inter-token latency, throughput, batch, continuous batching, quantization, offloading, tensor/pipeline/data parallelism.
- **Adaptation:** prompt engineering, RAG, fine-tuning, SFT, LoRA, QLoRA, adapter, continued pretraining, preference tuning, RLHF/DPO, evaluation.

---

## References

- DGX Spark User Guide — https://docs.nvidia.com/dgx/dgx-spark/index.html
- Ollama API — https://docs.ollama.com/api/introduction
- PyTorch basics — https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Transformers quantization — https://huggingface.co/docs/transformers/main_classes/quantization
- Ollama context length — https://docs.ollama.com/context-length
- Ollama tool calling — https://docs.ollama.com/capabilities/tool-calling
- PEFT quantization — https://huggingface.co/docs/peft/developer_guides/quantization
- Ollama import — https://docs.ollama.com/import
