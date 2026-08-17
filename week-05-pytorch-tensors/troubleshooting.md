# Troubleshooting

## GPU timing needs warmup and synchronization

CUDA work is asynchronous. If you call `time.perf_counter()` right after a matmul
without `torch.cuda.synchronize()`, you time the launch, not the compute, and get
numbers that are far too small. The benchmark runs a warmup first, then wraps each
timed iteration between a start time and a synchronize call.

## Timings vary between runs

Clocks, thermal state, and scheduling make single timings noisy. The benchmark
reports the median of 20 iterations. Expect the absolute milliseconds to shift a
little between runs while the ordering (FP16/BF16 faster than FP32, GPU faster
than CPU) stays the same.

## Out-of-memory on larger sizes

Raising `--size` grows memory as the square of the dimension. A size that fits in
FP16 may not fit in FP32, because FP32 uses twice the bytes. If PyTorch raises a
CUDA out-of-memory error, lower `--size` or use a smaller precision. On the DGX
Spark the memory is unified, so this shares the same pool as the rest of the
system.
