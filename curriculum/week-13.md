# Week 13 — Model architectures and modern terminology

**Phase 3: Understand transformers and model terminology**

---

## Learn

- Dense model
- Mixture of Experts (MoE)
- Total parameters
- Active parameters
- Router / gating network
- Expert
- Shared expert
- Sparse activation
- Multi-Token Prediction (MTP)
- Grouped Query Attention (GQA)
- Multi-Query Attention (MQA)
- Sliding-window attention
- RoPE
- Long-context extension
- Reasoning model
- Multimodal model

## Particularly important distinction

A model described as:

```text
100B total parameters, 12B active parameters
```

does **not** have the same memory requirement as a dense 12B model.

It may execute only a subset of its experts per token, but the inactive expert weights generally still need to be stored or made available. Therefore:

- **Compute per generated token** may resemble a smaller model.
- **Weight-storage requirement** may resemble the much larger total model.
- Communication and routing can become important in distributed deployments.
