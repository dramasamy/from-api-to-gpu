# From API to GPU

A software engineer's hands-on journey from **AI API consumer** to **local LLM systems architect** — local deployment, inference optimization, RAG, agents, fine-tuning, CUDA, and production AI architecture on an **NVIDIA DGX Spark**.

I've used frontier models via API for years but never deployed one myself. This repo documents me learning that from scratch — no prior ML background assumed — week by week, with reproducible code and measured results.

---

## What's here

- **[`curriculum/`](curriculum/README.md)** — the public 32-week roadmap: what I plan to learn and build each week, plus a parallel [CUDA track](curriculum/cuda-track.md).
- **`week-NN-topic/`** (coming as I go) — hands-on labs, one per week: runnable code, benchmarks, and observations.

## The approach

Every week follows one loop:

> **Run → observe → measure → understand → modify → document**

Always quantifying memory (GB), tokens/sec, time-to-first-token (ms), and latency — not another `pip install` tutorial.

## Roadmap at a glance

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

Start here → **[curriculum/README.md](curriculum/README.md)**

## Hardware

NVIDIA DGX Spark — DGX OS, ARM64, unified memory.

## License

See [LICENSE](LICENSE).
