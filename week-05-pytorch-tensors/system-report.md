# System report

Captured on 2026-07-16.

```text
$ ~/venvs/w1/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_name", torch.cuda.get_device_name(0))
for dt in (torch.float32, torch.float16, torch.bfloat16):
    print(dt, "element_size", torch.zeros(1, dtype=dt).element_size())
PY
torch 2.13.0+cu130
cuda_available True
device_name NVIDIA GB10
torch.float32 element_size 4
torch.float16 element_size 2
torch.bfloat16 element_size 2
```

The dtype byte sizes match the precision table from Week 4. The benchmark ran on
an NVIDIA GB10 with 121 GiB of unified memory.
