# Week 4 — Parameters, weights, and memory

**Phase 1: Become comfortable running local models**

> This is where "7B," "14B," and "70B" should finally become meaningful.

---

## Learn

- A parameter is a learnable numerical value.
- A weight is a major category of parameter.
- Bias
- Tensor / matrix
- Layer
- Shape
- Parameter count
- Precision
- Weight memory
- Runtime memory
- Activation memory
- KV-cache memory
- Framework overhead

## Core calculation

Approximate weight storage:

```text
weight memory ≈ number of parameters × bits per parameter / 8
```

For a 7-billion-parameter model:

```text
FP32: 7B × 4 bytes  ≈ 28 GB
FP16: 7B × 2 bytes  ≈ 14 GB
INT8: 7B × 1 byte   ≈ 7 GB
INT4: 7B × 0.5 byte ≈ 3.5 GB
```

Actual runtime usage is higher: caches, temporary tensors, runtime metadata, and other buffers.

## Hands-on

Write a calculator:

```bash
python model_memory.py --parameters 70 --precision int4
```

Output:

```text
Raw weight estimate: 35.0 GB
Estimated operational range: ...
```

## Milestone check

By the end of this week I should have: deployed a local model, accessed it via API, inspected its config, calculated its theoretical memory footprint, and published repeatable performance results — i.e. **crossed from AI API consumer to beginner LLM deployment engineer.**
