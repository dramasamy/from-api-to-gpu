# Glossary

Plain-language definitions for the terms in this journey, written for a software engineer with no ML background. Infra analogies are included where they help. This grows as new terms appear week to week.

> Convention: define a term here the first time it shows up in a lab or post.

## Model fundamentals

- **Parameter** — a learnable number inside a model. "7B" means ~7 billion of them. Roughly analogous to all the tuned config values a system learned instead of a human setting them.
- **Weight** — the main kind of parameter; the strength of a connection. Stored as numbers in tensors.
- **Bias** — a second kind of parameter added after a weighted sum; shifts the result.
- **Tensor** — a typed, multi-dimensional array (scalar → vector → matrix → higher). The core data structure models are built from.
- **Layer** — one processing stage; a model is a stack of layers.
- **Checkpoint** — a saved snapshot of a model's parameters at a point in time (like a build artifact).
- **Architecture** — the structural design of a model (how layers/attention/MLPs are wired), independent of the trained weights.
- **Base model** — pretrained on raw text; predicts next tokens but isn't tuned to follow instructions.
- **Instruct model** — a base model further tuned to follow instructions / chat.
- **Reasoning model** — tuned to produce intermediate reasoning steps before answering.
- **Multimodal model** — handles more than text (e.g. images + text).

## Tokens & transformers

- **Token** — a chunk of text (often a subword), the unit a model actually reads/writes. Not the same as a word.
- **Tokenizer** — converts text ↔ token IDs. Different models tokenize the same text differently.
- **Vocabulary** — the fixed set of tokens a model knows.
- **Special tokens** — control tokens like BOS (begin), EOS (end), and padding.
- **Embedding** — a vector representing a token's meaning; similar meanings sit closer together.
- **Attention** — the mechanism that lets each token "look at" other tokens to build context.
- **Query / Key / Value (Q/K/V)** — the three projections attention uses to decide what to focus on and what to pull in.
- **Transformer block** — the repeated unit: attention + feed-forward (MLP) + normalization + residual connections.
- **Logit** — the raw score for each possible next token before turning into probabilities.
- **Sampling** — choosing the next token from the probability distribution (greedy, top-k, top-p, temperature).
- **Temperature** — randomness knob for sampling; higher = more varied output.
- **Context window** — max tokens the model can consider at once (prompt + generated so far).
- **RoPE** — rotary positional encoding; a common way to tell the model token order and extend context.
- **GQA / MQA** — Grouped/Multi-Query Attention; share key/value heads to shrink the KV cache and speed decoding.
- **KV cache** — stored keys/values from previous tokens so each new token doesn't recompute the whole sequence. Grows with context length and eats memory — like a per-request cache that scales with conversation length.

## Architecture terminology

- **Dense model** — every parameter is used for every token.
- **MoE (Mixture of Experts)** — only a subset of "expert" sub-networks runs per token.
- **Expert** — one of the parallel sub-networks in an MoE.
- **Router / gating network** — picks which experts handle each token.
- **Total vs active parameters** — total = all weights that must be stored; active = the fraction actually computed per token. "100B total / 12B active" costs like 12B to compute but like 100B to store.
- **MTP (Multi-Token Prediction)** — predicting several future tokens at once to speed generation.
- **Speculative decoding** — a small model drafts tokens that a big model verifies, speeding output.
- **Distillation** — training a smaller model to mimic a larger one.

## Precision & quantization

- **FP32 / FP16 / BF16 / FP8** — floating-point formats with fewer bits going down the list; fewer bits = less memory, less precision.
- **INT8 / INT4** — integer formats used for quantized weights; big memory savings.
- **Precision vs range** — precision = how finely values are represented; range = how large/small they can be. BF16 trades precision for range vs FP16.
- **Quantization** — representing weights (and sometimes activations) with fewer bits to cut memory/compute.
- **Dequantization** — converting quantized values back to higher precision for computation.
- **Calibration** — measuring value distributions to choose good quantization scales.
- **PTQ / QAT** — Post-Training Quantization (after training) vs Quantization-Aware Training (during).
- **Group size / per-channel / per-tensor** — how finely quantization scales are applied; finer = better accuracy, more overhead.

## Model formats

- **Safetensors** — a safe, fast tensor storage format (no arbitrary code execution).
- **GGUF** — the llama.cpp/Ollama format; often quantized; runs on CPU and GPU.
- **GPTQ / AWQ** — GPU-oriented quantized weight formats/methods, primarily for inference.
- **ONNX / TensorRT engine** — portable / hardware-optimized inference formats.
- **Adapter** — small add-on weights (e.g. LoRA) layered on a base model.

## Runtime & serving

- **Inference engine** — the software that loads a model and generates tokens (Ollama, llama.cpp, Transformers, vLLM, TensorRT-LLM).
- **Prefill** — processing the whole input prompt (parallel, compute-heavy).
- **Decode** — generating output tokens one at a time (sequential, memory-bandwidth-heavy).
- **TTFT (time to first token)** — latency until the first output token appears.
- **Inter-token latency** — time between successive output tokens.
- **Throughput** — total tokens/sec across all requests.
- **Batch / continuous batching** — grouping requests to use the GPU efficiently; continuous batching adds/removes requests mid-flight.
- **Offloading** — moving some weights/cache to CPU RAM or disk when they don't fit in GPU memory.
- **Tensor / pipeline / data parallelism** — ways to split a model or workload across multiple GPUs.

## Adaptation

- **Prompt engineering** — steering behavior via the prompt alone.
- **RAG (Retrieval-Augmented Generation)** — fetch relevant documents and put them in the prompt so the model answers from your data. Use for changing facts.
- **Fine-tuning** — adjusting model weights on your data. Use for behavior/style/format, not fast-changing facts.
- **SFT (Supervised Fine-Tuning)** — training on input→output examples.
- **LoRA** — Low-Rank Adaptation; trains small adapter matrices instead of all weights.
- **QLoRA** — LoRA on top of a quantized (e.g. 4-bit) base model to save training memory.
- **Rank / alpha / target modules** — LoRA hyperparameters: adapter size, scaling, and which layers get adapters.
- **Continued pretraining** — more next-token training on domain text before instruction tuning.
- **Preference tuning (RLHF / DPO)** — aligning outputs to preferred responses.
- **Evaluation** — measuring quality: exact match, semantic similarity, groundedness, hallucination rate, task success.

## Hardware / platform

- **GPU utilization** — fraction of time the GPU is doing work.
- **Unified memory** — CPU and GPU share one memory pool (as on DGX Spark), instead of separate discrete VRAM.
- **CUDA** — NVIDIA's platform for running code on the GPU.
- **Driver vs CUDA runtime vs toolkit** — the kernel driver, the runtime libraries programs link against, and the developer/compiler toolkit, respectively.
