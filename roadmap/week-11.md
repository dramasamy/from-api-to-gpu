# Week 11 — Transformer architecture

**Phase 3: Understand transformers and model terminology**

---

## Learn

- Transformer block
- How `num_hidden_layers` becomes a stack of transformer blocks
- How `hidden_size` sets the main vector width
- How `intermediate_size` sets the MLP / feed-forward width
- How the `architectures` class ties these config fields to an implementation
- Token embedding
- Positional information
- Attention layer
- MLP / feed-forward layer
- Normalization
- Residual connection
- Output / logit layer
- Decoder-only architecture
- Encoder-only architecture
- Encoder-decoder architecture

## Hands-on

Draw the path:

```text
Prompt
 → tokenizer
 → token IDs
 → embeddings
 → transformer blocks
 → logits
 → sampler
 → next token
```

Annotate the path with the Week 3 model's real layer count, hidden size, and
intermediate size. Explain each field here rather than expecting it to be
understood during repository inspection.
