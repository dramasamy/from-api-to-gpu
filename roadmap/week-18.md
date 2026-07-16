# Week 18 — KV cache and context length

**Phase 5: Inference engineering**

---

## Learn

- Prompt / context window
- Input tokens
- Output tokens
- KV cache
- Prefill
- Decode
- Why the first token is slower
- Why long context consumes more memory
- Why maximum advertised context may not be practical
- Difference between model `max_position_embeddings`, tokenizer
	`model_max_length`, runtime-configured context, and advertised family limits
- Context quality degradation
- Prefix caching

Increasing context length generally increases memory consumption — treat context length as a resource and performance setting, not just a model feature.

## Hands-on

Run the same model with 1K / 4K / 16K / 32K prompts (where supported). Measure:

- Memory
- Time to first token
- Decode speed
- Answer accuracy

Start from the Week 3 model's conflicting 32,768 model-config limit and 131,072
tokenizer limit. Verify the actual supported runtime settings instead of
assuming the larger number works.

## Reference

- Ollama context length — https://docs.ollama.com/context-length
