# Week 2 — Run your first model with Ollama

**Phase 1: Become comfortable running local models**

---

## Learn

- What a model runtime is
- Model versus runtime
- Model manifest
- Prompt / system prompt
- Generation parameters
- Temperature
- Top-p
- Context length
- Tokens per second
- Time to first token

Ollama gives an easy first layer: CLI, local HTTP API, model configuration, and OpenAI-compatible access. Its default local API is on port `11434`, and its Modelfile supports base models, runtime parameters, prompt templates, system messages, and adapters.

## Hands-on

- Install and run a small instruct model.
- Chat through the CLI.
- Call it using `curl`.
- Call it using Python.
- Change temperature.
- Change context size.
- Compare deterministic vs creative output.
- Measure response time.

Example API test:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-model>",
    "messages": [
      { "role": "user", "content": "Explain Kubernetes scheduling in three sentences." }
    ],
    "stream": false
  }'
```

## Deliverable

A Python command-line chat client and a benchmark script.

## Reference

- Ollama API — https://docs.ollama.com/api/introduction
