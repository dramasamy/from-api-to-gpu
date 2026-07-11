# CUDA Track (parallel)

Run this **alongside** the main 32-week plan. Do **not** wait until the end to learn CUDA — but also do **not** start by writing complex kernels. Begin around **Week 5** and progress one level at a time.

> Goal: you don't need to become a CUDA compiler engineer to be an LLM solution architect. You **do** need to profile an inference workload and explain whether it is **compute-bound, memory-capacity-bound, memory-bandwidth-bound, CPU-bound, serialization-bound, or network-bound.**

Public repo folder: `13-cuda/` (`gpu-architecture/`, `kernels/`, `memory/`, `profiling/`, `custom-extension/`).

---

## Level 1 — User (start ~Week 5)

Learn:

- What CUDA is
- Driver versus toolkit
- CUDA runtime
- CUDA-enabled PyTorch
- Device selection
- Device memory
- Host-to-device transfer
- Synchronization
- CUDA out-of-memory errors

## Level 2 — Observer

Tools:

- `nvidia-smi`
- PyTorch profiler
- Nsight Systems
- Nsight Compute
- CUDA memory statistics

Understand:

- Kernel launch
- Memory copy
- GPU utilization
- Compute utilization
- Memory bandwidth
- Synchronization stalls

## Level 3 — Performance engineer

Learn:

- Threads
- Blocks
- Grids
- Warps
- Streaming multiprocessors
- Global memory
- Shared memory
- Registers
- Coalesced access
- Kernel fusion
- Tensor cores
- Arithmetic intensity

## Level 4 — Developer

Build:

- A vector-add kernel
- Matrix operation
- Simple reduction
- PyTorch CUDA extension
- Triton kernel
- Compare custom code with optimized PyTorch operations

## Level 5 — LLM optimization specialist

Understand:

- Fused attention
- FlashAttention concepts
- Fused MLP
- Quantized matrix multiplication
- Tensor parallel collectives
- NCCL
- CUDA graphs
- Speculative decoding
- Kernel bottleneck analysis
- Why TensorRT-LLM can outperform generic execution

---

## Suggested pairing with main track

| Main-track weeks | CUDA level |
| ---------------- | ---------- |
| 5–8   | Level 1 — User |
| 9–13  | Level 2 — Observer |
| 14–16 | Level 3 — Performance engineer |
| 17–20 | Level 4 — Developer |
| 21–32 | Level 5 — LLM optimization specialist |
