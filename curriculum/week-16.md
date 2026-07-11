# Week 16 — Model packaging formats

**Phase 4: Quantization and model formats**

---

## Learn

- PyTorch checkpoint
- Safetensors
- GGUF
- GPTQ
- AWQ
- ONNX
- TensorRT engine
- Adapter files
- Why a model format and a quantization method are not always the same thing
- Why runtimes support different formats

## Hands-on

Create a matrix:

| Format      | Typical runtime         | CPU/GPU | Quantized? | Fine-tunable?       |
| ----------- | ----------------------- | ------- | ---------- | ------------------- |
| Safetensors | Transformers            | GPU     | Possibly   | Yes                 |
| GGUF        | llama.cpp / Ollama      | CPU/GPU | Often      | Usually inference   |
| AWQ         | Compatible GPU runtimes | GPU     | Yes        | Primarily inference |
| GPTQ        | Compatible GPU runtimes | GPU     | Yes        | Primarily inference |
