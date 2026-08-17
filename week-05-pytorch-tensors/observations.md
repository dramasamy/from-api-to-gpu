# Observations

- A tensor's byte size matches Week 4's math exactly: element count times bytes
  per element. A 1024x1024 FP32 tensor is 4,194,304 bytes, and the same shape at
  FP16 or BF16 is 2,097,152 bytes, exactly half.
- FP16 and BF16 both report 2 bytes per element, so they use the same memory. The
  difference is how they split those 16 bits: `torch.finfo` shows FP16 maxes at
  about 65,500 with a finer step, while BF16 keeps near-FP32 range with a coarser
  step.
- On a 4096x4096 matmul, the GPU at FP32 reached about 18,473 GFLOP/s versus
  about 817 GFLOP/s on the CPU. That is roughly a 22.6x speedup for this shape.
- FP16 and BF16 matmul ran in about 1.55 ms versus 7.44 ms for FP32, roughly 4.8x
  faster. The three matrices halve from 192 MiB to 96 MiB of payload, while
  PyTorch's measured peak fell from 224 MiB to 128 MiB (about 43 percent, since
  the peak includes extra working allocation).
- Timings vary run to run, so the benchmark reports the median of 20 GPU
  iterations and 5 CPU iterations after a warmup. FP16 and BF16 can swap order,
  but 16-bit beating FP32 and the GPU beating the CPU are stable.
