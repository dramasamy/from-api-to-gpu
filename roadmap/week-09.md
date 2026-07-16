# Week 9 — Tokenization

**Phase 3: Understand transformers and model terminology**

---

## Learn

- Character
- Word
- Subword
- Token
- Vocabulary
- BPE / SentencePiece-style concepts
- Token IDs
- Special tokens
- BOS, EOS, and padding
- Chat templates
- How `vocab_size`, BOS/EOS/padding IDs, and `model_max_length` appear in model
	repository tokenizer files
- Why token counts differ between models
- Why languages may tokenize differently

## Hands-on

Compare how several tokenizers encode:

- English
- Tamil
- Python
- Kubernetes YAML
- JSON
- Robotics-platform technical documentation

Inspect `vocab.json`, `merges.txt`, `tokenizer_config.json`, and
`tokenizer.json` from the Week 3 model. Connect each file to the tokenization
steps above.
