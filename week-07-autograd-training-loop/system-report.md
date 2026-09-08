# System report

## Toy weight choice: 2.0 versus 3.0

Read-only Spark check with fixed input 3 and bias 1, against target 10.
This does not change the training experiment or its measurements.

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
for start in (2.0, 3.0):
  w = torch.tensor([start], requires_grad=True)
  b = torch.tensor([1.0], requires_grad=True)
  x = torch.tensor([3.0])
  pred = w * x + b
  loss = (pred - 10.0)**2
  loss.backward()
  print(f'w={start}: pred={pred.item()}, loss={loss.item()}, '
      f'w.grad={w.grad.item()}, b.grad={b.grad.item()}')
PY
```

```text
w=2.0: pred=7.0, loss=9.0, w.grad=-18.0, b.grad=-6.0
w=3.0: pred=10.0, loss=0.0, w.grad=0.0, b.grad=0.0
```

## Current verified run

Commands were issued from the parent repository root on the control Mac.
SSH runs Python on Spark; the source redirection reads the local public lab.
The source's training arithmetic is unchanged. Reporting now retains raw floats
and measures final loss after the last update.

### Runtime and isolated directory

```bash
ssh spark 'uname -m; ~/venvs/w1/bin/python -c "import torch; print(torch.__version__)"'
```

```text
aarch64
2.13.0+cu130
```

```bash
ssh spark 'mktemp -d /tmp/week07-reviewed.XXXXXXXX'
```

```text
/tmp/week07-reviewed.c56DPELh
```

### Training

This is the exact invocation. The long line is retained for command provenance.
Both generated files are inside the newly allocated directory on Spark.

```bash
ssh spark 'cd /tmp/week07-reviewed.c56DPELh && ~/venvs/w1/bin/python - --epochs 20000 --lr 0.0003 --seed 0 --checkpoint manual_celsius_to_fahrenheit.pt --output results.json' < public/week-07-autograd-training-loop/train_manual.py
```

Complete stdout:

```text
device=cpu
before training: weight=1.540996 bias=-0.293429
loss.grad_fn type=MeanBackward0
first grad backward(): weight=-2314.6404 bias=-73.8247
first grad by hand   : weight=-2314.6404 bias=-73.8247
epoch     1 loss=    965.8369 weight=2.2354 bias=-0.2713
epoch    50 loss=    849.9656 weight=2.0501 bias=0.5317
epoch  1000 loss=    319.4210 weight=1.9533 bias=12.7090
epoch  5000 loss=      5.1848 weight=1.8195 bias=29.5422
epoch 10000 loss=      0.0300 weight=1.8015 bias=31.8129
epoch 20000 loss=      0.0000 weight=1.8000 bias=31.9981
after training:  weight=1.8000 bias=31.9981 (true weight 1.8, bias 32)
final loss (post-update, unrounded)=2.9371251457632752e-06
saved checkpoint to manual_celsius_to_fahrenheit.pt
inference: 25 C -> 76.9985 F (true 77)
```

### Independent checkpoint load

The new process reads the checkpoint from disk, without importing the training
program or receiving its in-memory variables. `weight` and `bias` are the saved
dictionary keys; each value is a detached one-element tensor. The expression
uses those values and a new input of shape `(1, 1)` to produce one prediction.

```bash
ssh spark 'cd /tmp/week07-reviewed.c56DPELh && ~/venvs/w1/bin/python -' <<'PY'
import torch

# Read only the checkpoint: no training program or in-memory parameters.
state = torch.load("manual_celsius_to_fahrenheit.pt", weights_only=True)
print(f"keys={list(state.keys())}")
for name, value in state.items():
    # Each saved parameter is a one-element tensor, without gradient tracking.
    print(f"{name}: shape={tuple(value.shape)} requires_grad={value.requires_grad}")
with torch.no_grad():
    # One row with one Celsius input: shape (1, 1).
    test_c = torch.tensor([[25.0]])
    prediction = state["weight"] * test_c + state["bias"]
print(f"prediction: shape={tuple(prediction.shape)} "
      f"requires_grad={prediction.requires_grad}")
print(f"inference: 25 C -> {prediction.item()} F (true 77)")
PY
```

Complete stdout:

```text
keys=['weight', 'bias']
weight: shape=(1,) requires_grad=False
bias: shape=(1,) requires_grad=False
prediction: shape=(1, 1) requires_grad=False
inference: 25 C -> 76.99851989746094 F (true 77)
```

### Generated JSON

The local `results.json` preserves this remote artifact byte-for-byte. These
are values from the new run, not precision reconstructed from the old rounded
baseline. `history` contains selected post-update measurements; `final_loss`
measures the final saved parameters, so it agrees with the last history row.

```bash
ssh spark 'cat /tmp/week07-reviewed.c56DPELh/results.json'
```

```json
{
  "seed": 0,
  "epochs": 20000,
  "lr": 0.0003,
  "initial_weight": 1.5409960746765137,
  "initial_bias": -0.293428897857666,
  "first_loss": 1491.7762451171875,
  "first_weight_grad_backward": -2314.640380859375,
  "first_weight_grad_by_hand": -2314.640380859375,
  "first_bias_grad_backward": -73.82466125488281,
  "first_bias_grad_by_hand": -73.82466125488281,
  "learned_weight": 1.800014615058899,
  "learned_bias": 31.998149871826172,
  "final_loss": 2.9371251457632752e-06,
  "history": [
    {
      "epoch": 1,
      "loss": 965.8368530273438,
      "weight": 2.2353882789611816,
      "bias": -0.27128151059150696
    },
    {
      "epoch": 50,
      "loss": 849.9656372070312,
      "weight": 2.0500855445861816,
      "bias": 0.5316590070724487
    },
    {
      "epoch": 1000,
      "loss": 319.4210205078125,
      "weight": 1.9533096551895142,
      "bias": 12.70899486541748
    },
    {
      "epoch": 5000,
      "loss": 5.184848308563232,
      "weight": 1.8195323944091797,
      "bias": 29.5422306060791
    },
    {
      "epoch": 10000,
      "loss": 0.030038340017199516,
      "weight": 1.8014867305755615,
      "bias": 31.81292724609375
    },
    {
      "epoch": 20000,
      "loss": 2.9371251457632752e-06,
      "weight": 1.800014615058899,
      "bias": 31.998149871826172
    }
  ],
  "inference_25c_f": 76.99851989746094
}
```

## Second verification run (latest blog commands)

The main agent executed the following training block and fresh-process loader
on Spark using the same `run_dir`. Their complete output is captured in the
latest blog and copied here unchanged. This is a separate run from
`/tmp/week07-reviewed.c56DPELh` above; neither its JSON nor the historical
baseline is replaced here.

### Isolated directory and training

```bash
run_dir=$(ssh spark 'mktemp -d /tmp/week07-reviewed.XXXXXXXX')
printf '%s\n' "$run_dir"
ssh spark "cd '$run_dir' && ~/venvs/w1/bin/python - \
    --epochs 20000 --lr 0.0003 --seed 0 \
    --checkpoint manual_celsius_to_fahrenheit.pt --output results.json" \
    < public/week-07-autograd-training-loop/train_manual.py
```

```text
/tmp/week07-reviewed.RXvW4Ygo
device=cpu
before training: weight=1.540996 bias=-0.293429
loss.grad_fn type=MeanBackward0
first grad backward(): weight=-2314.6404 bias=-73.8247
first grad by hand   : weight=-2314.6404 bias=-73.8247
epoch     1 loss=    965.8369 weight=2.2354 bias=-0.2713
epoch    50 loss=    849.9656 weight=2.0501 bias=0.5317
epoch  1000 loss=    319.4210 weight=1.9533 bias=12.7090
epoch  5000 loss=      5.1848 weight=1.8195 bias=29.5422
epoch 10000 loss=      0.0300 weight=1.8015 bias=31.8129
epoch 20000 loss=      0.0000 weight=1.8000 bias=31.9981
after training:  weight=1.8000 bias=31.9981 (true weight 1.8, bias 32)
final loss (post-update, unrounded)=2.9371251457632752e-06
saved checkpoint to manual_celsius_to_fahrenheit.pt
inference: 25 C -> 76.9985 F (true 77)
```

### Fresh-process checkpoint load

```bash
ssh spark "cd '$run_dir' && ~/venvs/w1/bin/python -" <<'PY'
import torch

# Read only the checkpoint: no training program or in-memory parameters.
state = torch.load("manual_celsius_to_fahrenheit.pt", weights_only=True)
print(f"keys={list(state.keys())}")
for name, value in state.items():
    # Each saved parameter is a one-element tensor, without gradient tracking.
    print(f"{name}: shape={tuple(value.shape)} requires_grad={value.requires_grad}")
with torch.no_grad():
    # One row with one Celsius input: shape (1, 1).
    test_c = torch.tensor([[25.0]])
    prediction = state["weight"] * test_c + state["bias"]
print(f"prediction: shape={tuple(prediction.shape)} "
      f"requires_grad={prediction.requires_grad}")
print(f"inference: 25 C -> {prediction.item()} F (true 77)")
PY
```

```text
keys=['weight', 'bias']
weight: shape=(1,) requires_grad=False
bias: shape=(1,) requires_grad=False
prediction: shape=(1, 1) requires_grad=False
inference: 25 C -> 76.99851989746094 F (true 77)
```

### Deterministic mode and gradient lifecycle check

The main agent also executed this exact combined command for the blog. It was
independently rerun on Spark during this lab sync, with identical full stdout.
It does not read the checkpoint or change any files.

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
from torch import nn

# Mode and autograd are independent controls; use no randomness in this check.
drop = nn.Dropout(p=1.0)  # Drop every input in training; pass through in eval.
x = torch.ones(3, requires_grad=True)
drop.eval()
y = drop(x)
print("eval:", "training=", drop.training, "requires_grad=", y.requires_grad)
print("values:", y)
drop.train()
with torch.no_grad():
    y = drop(x)
print("train + no_grad:", "training=", drop.training,
      "requires_grad=", y.requires_grad)
print("values:", y)

# A fresh forward rebuilds the graph; zero_ clears only accumulated gradients.
w = torch.tensor([2.0], requires_grad=True)
for step in range(1, 3):
    loss = (w * 3.0) ** 2
    print("forward", step, "grad_fn:", type(loss.grad_fn).__name__)
    loss.backward()
    print("after backward:", w.grad.item())
    w.grad.zero_()
    print("after zero_:", w.grad.item(), "w:", w.item())
PY
```

```text
eval: training= False requires_grad= True
values: tensor([1., 1., 1.], requires_grad=True)
train + no_grad: training= True requires_grad= False
values: tensor([0., 0., 0.])
forward 1 grad_fn: PowBackward0
after backward: 36.0
after zero_: 0.0 w: 2.0
forward 2 grad_fn: PowBackward0
after backward: 36.0
after zero_: 0.0 w: 2.0
```

## Inline mini-experiment verification

Each exact blog command below was rerun on Spark during this lab sync, in its
own fresh Python process. All complete outputs matched the blog. These are
new captures, not older evidence relabeled as rerun. No files were written.

### Current computational graph

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])
pred = w * x + b
loss = (pred - 10.0)**2
print("pred:", pred)
print("loss:", loss)
print("x.requires_grad:", x.requires_grad, " w.requires_grad:", w.requires_grad)
PY'
```

```text
pred: tensor([7.], grad_fn=<AddBackward0>)
loss: tensor([9.], grad_fn=<PowBackward0>)
x.requires_grad: False  w.requires_grad: True
```

### Leaves and stored gradients

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
torch.manual_seed(0)
C=torch.tensor([[-40.],[-10.],[0.],[20.],[37.],[100.]]); F=C*1.8+32
w=torch.randn(1, requires_grad=True); b=torch.randn(1, requires_grad=True)
pred=w*C+b; loss=((pred-F)**2).mean(); loss.backward()
for name, t in [("weight ", w), ("bias   ", b), ("celsius", C)]:
    print(name, "is_leaf", t.is_leaf, "req_grad", t.requires_grad, "grad", t.grad)
print("pred   ", "is_leaf", pred.is_leaf, "grad_fn", type(pred.grad_fn).__name__)
PY'
```

```text
weight  is_leaf True req_grad True grad tensor([-2314.6404])
bias    is_leaf True req_grad True grad tensor([-73.8247])
celsius is_leaf True req_grad False grad None
pred    is_leaf False grad_fn AddBackward0
```

### Hand derivative check

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])
pred = w * x + b
loss = (pred - 10.0)**2
loss.backward()
print("w.grad:", w.grad, " b.grad:", b.grad)
print("hand dL/dw:", 2*(pred.item()-10)*x.item(),
      " hand dL/db:", 2*(pred.item()-10))
PY'
```

```text
w.grad: tensor([-18.])  b.grad: tensor([-6.])
hand dL/dw: -18.0  hand dL/db: -6.0
```

### Mean of the per-example gradients

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
torch.manual_seed(0)
C=torch.tensor([[-40.],[-10.],[0.],[20.],[37.],[100.]]); F=C*1.8+32
w=torch.randn(1, requires_grad=True); b=torch.randn(1, requires_grad=True)
pred=w*C+b
contrib=(2.0*(pred-F)*C)   # per-example weight-gradient contribution
print("per-example weight-grad contributions:")
print(contrib.detach().flatten())
print("mean =", contrib.mean().item())
PY'
```

```text
per-example weight-grad contributions:
tensor([  1754.6619,    594.0678,     -0.0000,  -1498.9403,  -3098.8667,
        -11638.7637])
mean = -2314.640380859375
```

### Update failure without `no_grad()`

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
w = torch.tensor([2.0], requires_grad=True)
loss = (w * 3.0 - 1.0)**2
loss.backward()
try:
    w -= 0.1 * w.grad
except RuntimeError as e:
    print("without no_grad:")
    print(str(e))
with torch.no_grad():
    w -= 0.1 * w.grad
print("with no_grad, new w:", w.item())
print("w.requires_grad still:", w.requires_grad)
PY'
```

```text
without no_grad:
a leaf Variable that requires grad is being used in an in-place operation.
with no_grad, new w: -1.0
w.requires_grad still: True
```

### Gradient accumulation and clearing

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
w = torch.tensor([2.0], requires_grad=True)
for i in range(1, 3):
    loss = w * 3.0                   # dL/dw = 3
    loss.backward()
    print("backward", i, "-> w.grad:", w.grad.item())
w.grad.zero_()                       # fill the gradient tensor with zeros
loss = w * 3.0
loss.backward()
print("after zero_ and fresh backward -> w.grad:", w.grad.item())
PY'
```

```text
backward 1 -> w.grad: 3.0
backward 2 -> w.grad: 6.0
after zero_ and fresh backward -> w.grad: 3.0
```

### Original seeded dropout check

```bash
ssh spark '~/venvs/w1/bin/python - <<PY
import torch
from torch import nn
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.ones(8)
drop.train()
print("train mode :", drop(x))
drop.eval()
print("eval  mode :", drop(x))
PY'
```

```text
train mode : tensor([0., 0., 2., 0., 0., 0., 2., 2.])
eval  mode : tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

## Historical run (original output, unchanged)

The transcript below belongs to the original script and its rounded
`results-historical.json`. The old raw floats were not retained. Its object
address is historical output, not a value to reproduce. Do not combine this
transcript with the new run's raw JSON or treat displayed `0.0000` as zero loss.

```text
$ ~/venvs/w1/bin/python -c 'import torch; print(torch.__version__)'
2.13.0+cu130

$ ~/venvs/w1/bin/python - < train_manual.py
device=cpu
before training: weight=1.540996 bias=-0.293429
loss.grad_fn=<MeanBackward0 object at 0xed88a92c6470>
first grad backward(): weight=-2314.6404 bias=-73.8247
first grad by hand   : weight=-2314.6404 bias=-73.8247
epoch     1 loss=    965.8369 weight=2.2354 bias=-0.2713
epoch    50 loss=    849.9656 weight=2.0501 bias=0.5317
epoch  1000 loss=    319.4210 weight=1.9533 bias=12.7090
epoch  5000 loss=      5.1848 weight=1.8195 bias=29.5422
epoch 10000 loss=      0.0300 weight=1.8015 bias=31.8129
epoch 20000 loss=      0.0000 weight=1.8000 bias=31.9981
after training:  weight=1.8000 bias=31.9981 (true weight 1.8, bias 32)
saved checkpoint to manual_celsius_to_fahrenheit.pt
inference: 25 C -> 76.9985 F (true 77)
```
