# Week 29 — Fine-tuning concepts

**Phase 8: Fine-tuning and model adaptation**

---

## Learn

- Pretraining
- Continued pretraining
- Supervised fine-tuning
- Instruction tuning
- Preference tuning
- Full fine-tuning
- Parameter-efficient fine-tuning
- LoRA
- QLoRA
- Adapter
- Rank
- Alpha
- Target modules
- Learning rate
- Epoch
- Overfitting
- Catastrophic forgetting

## Critical architectural rule

- **RAG** for changing facts and documents
- **Fine-tuning** for behavior, style, task format, and specialized patterns
- **Tools / APIs** for live state and actions

Do **not** fine-tune the model merely to memorize a database that changes frequently.

For the robotics platform:

| Information                       | Preferred approach        |
| --------------------------------- | ------------------------- |
| Stable electrical concepts        | Base model or fine-tuning |
| Platform response style           | Fine-tuning               |
| Pin mappings and board revisions  | RAG / tool lookup         |
| Current lesson assignment         | LMS / API tool            |
| Student progress                  | Database / API tool       |
| Troubleshooting dialogue patterns | Fine-tuning               |
| Current curriculum content        | RAG with access controls  |
