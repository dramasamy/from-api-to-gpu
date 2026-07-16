# Week 10 — Attention

**Phase 3: Understand transformers and model terminology**

---

## Learn

- Query
- Key
- Value
- Attention score
- Self-attention
- Causal masking
- Attention heads
- Multi-head attention
- How `num_attention_heads` and `num_key_value_heads` map to query, key, and
	value paths in `config.json`
- Why attention cost grows with sequence length
- Prefill versus decoding

No need to memorize every equation — be able to explain what each matrix is doing.

## Hands-on

Implement a simplified single-head attention operation in PyTorch. Then connect
its tensor shapes to the Week 3 model's attention-head config fields.
