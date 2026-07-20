# Observations

- The weight formula `parameters x bits / 8` is exact for one uniform precision.
  Qwen2.5-3B-Instruct at BF16 gives 6,171,877,376 bytes, matching the value in
  `model.safetensors.index.json` from Week 3.
- Mixed quantization costs more on disk than its nominal bit count. `llama3.2:3b`
  is labeled Q4_K_M, and its tensor list has 168 Q4_K, 29 Q6_K, and 58 F32
  tensors. Its 2,019,377,376-byte weight layer works out to about 5.03 on-disk
  bits per parameter, not 4. The blob's size and SHA-256 digest match its
  manifest record. The clean INT4 estimate is a lower bound.
- Runtime memory is larger than the weights. `llama3.2:3b` is a 2.0 GB package
  but Ollama reports 2.6 GB loaded with a 4,096-token context and 4.1 GB with a
  16,384-token context. `ollama ps` reports one total and does not split the KV
  cache, activations, and framework buffers.
- Context length is a memory decision. Quadrupling the context from 4,096 to
  16,384 raised the reported loaded size well past the package size.
- The Spark has 121 GiB of unified memory, so these small models leave large
  headroom. The formula matters most when planning larger models against a fixed
  memory budget.
