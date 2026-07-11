# Week 31 — LoRA and QLoRA

**Phase 8: Fine-tuning and model adaptation**

---

## Learn

- Frozen base weights
- Low-rank matrices
- Trainable adapter parameters
- Quantized base model
- Adapter merge
- Adapter serving
- Training memory versus inference memory
- Checkpointing
- Gradient accumulation
- Gradient checkpointing

Hugging Face's PEFT docs include training LoRA adapters on 4-bit quantized models — the central idea behind common QLoRA-style workflows.

## Hands-on

Fine-tune a modest model on a small dataset. Compare:

- Base model
- Base model + RAG
- Base model + adapter
- Base model + adapter + RAG

Ollama can also import supported Safetensors adapters by referencing the base model and adapter from a Modelfile.

## References

- PEFT quantization — https://huggingface.co/docs/peft/developer_guides/quantization
- Ollama import — https://docs.ollama.com/import
