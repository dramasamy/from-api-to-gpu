# Troubleshooting

## The clean formula under-estimates quantized files

`parameters x bits / 8` is exact only for one uniform precision such as FP32,
FP16, or BF16. Formats like `Q4_K_M` mix bit widths across tensors, so the real
file is larger than the nominal 4-bit estimate. Treat the INT4 number as a lower
bound and read the real weight-layer size from the model file when it matters.

## Unified memory has no separate VRAM figure

The DGX Spark shares one memory pool between CPU and GPU, so `nvidia-smi` does
not print a normal VRAM total. Use `free -h` for the whole pool and `ollama ps`
for the loaded model size, including its KV cache.

## Loaded size changes with context

`ollama ps` reports the loaded size for the context you requested. A larger
`num_ctx` reserves a larger KV cache, so the same model shows a larger loaded
size. Set context from measured need, not the model maximum.
