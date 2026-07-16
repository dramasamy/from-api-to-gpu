# Week 3 — Understand model files and Hugging Face

**Phase 1: Become comfortable running local models**

---

## Learn

- Base model versus instruct model
- Model family
- Architecture
- Checkpoint
- Repository
- Model card
- License
- `config.json`
- Tokenizer files
- Safetensors
- Shards
- Generation configuration
- Chat template
- Special tokens

## Scope boundary

Week 3 teaches where these fields and files live, not the full math behind every
value. Record the real values now and teach them in depth later:

- Week 4: parameter count, `torch_dtype`, precision, and weight memory
- Week 9: vocabulary, tokenizer files, BOS/EOS, special tokens, chat templates
- Week 10: attention heads, query, key, and value
- Week 11: transformer blocks, hidden size, intermediate size, architecture class
- Week 12: temperature, top-k, top-p, repetition penalty, seed, and sampling
- Week 13: GQA, RoPE, and long-context extensions
- Week 18: KV cache and model/tokenizer/runtime context limits

## Hands-on

Pick one small model repository and inspect every major file.

Write a script that prints:

- Architecture
- Parameter-related configuration
- Number of layers
- Hidden size
- Attention heads
- Vocabulary size
- Maximum position length
- Data type
- EOS and BOS token IDs

## Deliverable

`model-inspector.py`
