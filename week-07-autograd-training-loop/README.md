# Week 7: Autograd and the training loop

Reproducible lab for Week 7 of *From API to GPU*. It rewrites the Week 6
Celsius-to-Fahrenheit training by hand: no `nn.Linear` and no `torch.optim`. The
weight and bias are plain tensors with `requires_grad=True`, and the optimizer
step is one line written directly. The point is to show that `backward()` and the
training loop are not magic.

> Companion article: *From API to GPU, Week 7: The Training Loop, Written by Hand*.
> Public article link forthcoming; this lab contains the runnable commands and evidence.
>
> Roadmap: [`../roadmap/week-07.md`](../roadmap/week-07.md)

## Prerequisites

- A machine with PyTorch. This lab reuses the Week 1 environment `~/venvs/w1`
  (torch 2.13.0+cu130). It runs on the CPU; no GPU is required.
- From the parent repository root, confirm the remote architecture and runtime:

  ```bash
  ssh spark 'uname -m; ~/venvs/w1/bin/python -c "import torch; print(torch.__version__)"'
  ```

  The captured run returned `aarch64` and `2.13.0+cu130`. The CUDA build was
  installed, but this training program runs on the CPU.

## Files

| File | Purpose |
| ---- | ------- |
| `train_manual.py` | Trains Celsius to Fahrenheit with a hand-written loop. |
| `requirements.txt` | Notes the pinned PyTorch version used. |
| `results.json` | Raw floats from the new captured run, including final loss. |
| `results-historical.json` | Unchanged rounded results from the original run. |
| `observations.md` | Summarizes what the numbers show. |
| `system-report.md` | Full current command/output evidence and historical transcript. |
| `model-run.yaml` | Records the training configuration. |
| `troubleshooting.md` | Covers requires_grad, no_grad, and gradient clearing. |

## Run

Run these commands on the control machine from the **parent repository root**.
For a public-only clone, run from its root and drop `public/` from local paths.
The local source is sent over SSH; Python, training, and checkpoint writes all
run on Spark. There is no need to copy the repository onto Spark.

Create a unique remote directory first, so no previous checkpoint or result is
overwritten. Keep the shell variable for the subsequent commands:

```bash
run_dir=$(ssh spark 'mktemp -d /tmp/week07-annotations.XXXXXXXX')
printf '%s\n' "$run_dir"
ssh spark "cd '$run_dir' && ~/venvs/w1/bin/python - \
    --epochs 20000 --lr 0.0003 --seed 0 \
    --checkpoint manual_celsius_to_fahrenheit.pt --output results.json" \
    < public/week-07-autograd-training-loop/train_manual.py
```

Expect the loss to decrease and the weight and bias to approach 1.8 and 32.
The same mathematical forward rule and plain-SGD update are used as in Week 6;
floating-point bitwise identity is not promised. The seed controls the random
start in the chosen environment. Console values are formatted for reading;
the generated JSON retains unrounded floats, including the final post-update
loss. See `system-report.md` for the captured run and exact temporary path.

## Load the checkpoint in a fresh process

This starts a separate Python process that reads only the saved tensor dictionary.
It prints both keys, tensor shapes, gradient flags, and the prediction for 25 C:

This code runs on Spark:

```bash
ssh spark "cd '$run_dir' && ~/venvs/w1/bin/python -" <<'PY'
import torch

# Load the saved parameter dictionary, restricting loaded types.
# Only load a checkpoint you trust.
state = torch.load("manual_celsius_to_fahrenheit.pt", weights_only=True)
# Verify the checkpoint contains the expected weight and bias entries.
print(f"keys={list(state.keys())}")
# Inspect each saved parameter's shape and gradient flag.
for name, value in state.items():
    print(f"{name}: shape={tuple(value.shape)} requires_grad={value.requires_grad}")
with torch.no_grad():  # Prediction needs no gradient graph.
    # One new Celsius input, shape (1, 1).
    test_c = torch.tensor([[25.0]])
    # Compute the model prediction from input, weight, and bias.
    # Use the checkpoint's learned tensors rather than training-process variables.
    prediction = state["weight"] * test_c + state["bias"]
# Verify the prediction's shape and that it has no gradient tracking.
print(f"prediction: shape={tuple(prediction.shape)} "
      f"requires_grad={prediction.requires_grad}")
# Compare the reloaded model's prediction with the known answer.
print(f"inference: 25 C -> {prediction.item()} F (true 77)")
PY
```

## Retrieve results without replacing the captured baseline

Inspect the remote JSON directly:

```bash
ssh spark "cat '$run_dir/results.json'"
```

Reserve a fresh local filename, then copy the remote JSON into it. This leaves
both checked-in result files untouched, even if you repeat the retrieval:

```bash
local_result=$(mktemp public/week-07-autograd-training-loop/results-rerun.XXXXXXXX)
scp "spark:$run_dir/results.json" "$local_result"
python3 -m json.tool "$local_result"
```

The checkpoint stays in `run_dir` on Spark; it is not added to the repository.
Temporary files can be removed by the operating system, so retain any artifacts
you need before that happens. The JSON file is evidence, not a checkpoint.

## Inline mini-experiments

Run these direct commands from the control machine. Each starts a fresh Python
process on Spark and writes no files. The blog explains the concepts;
[`system-report.md`](system-report.md) captures the commands and full output.

### Current computational graph

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
# Create a one-value weight, shape (1,), with gradient tracking enabled.
w = torch.tensor([2.0], requires_grad=True)
# Make a separate one-value bias with tracking enabled too.
b = torch.tensor([1.0], requires_grad=True)
# Omitted requires_grad defaults to False: this one-value input is fixed data.
x = torch.tensor([3.0])
# Compute the model prediction from input, weight, and bias.
pred = w * x + b
# Choose 10 as this tiny example target, unrelated to the Fahrenheit task.
# Score the squared prediction error against the target: (7 - 10)**2 = 9.
loss = (pred - 10.0)**2
# Show the prediction and loss with their recorded operations.
print("pred:", pred)
print("loss:", loss)
# Compare gradient tracking for the fixed input and trainable weight.
print("x.requires_grad:", x.requires_grad, " w.requires_grad:", w.requires_grad)
PY
```

### Leaves and stored gradients

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
torch.manual_seed(0)  # Repeat the random start in this PyTorch environment.
# Six Celsius inputs, one temperature per row: shape (6, 1).
C = torch.tensor([[-40.],[-10.],[0.],[20.],[37.],[100.]])
F = C * 1.8 + 32  # Build the six target Fahrenheit answers, shape (6, 1).
# Draw a one-value weight, shape (1,), with gradient tracking enabled.
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)  # A separate random bias.
# Compute the model prediction from input, weight, and bias.
pred = w * C + b  # Reuse each parameter across all six rows, shape (6, 1).
# Score the squared prediction error against each target, then average.
loss = ((pred - F) ** 2).mean()
loss.backward()  # Compute gradients and accumulate them in leaf .grad fields.
# Compare which leaves require gradients and actually received them.
for name, t in [("weight ", w), ("bias   ", b), ("celsius", C)]:
    print(name, "is_leaf", t.is_leaf, "req_grad", t.requires_grad, "grad", t.grad)
# Contrast those leaves with the prediction's recorded producing operation.
print("pred   ", "is_leaf", pred.is_leaf, "grad_fn", type(pred.grad_fn).__name__)
PY
```

### Hand derivative check

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
# Create one-value parameters, shape (1,), with gradient tracking enabled.
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)  # Bias also needs a gradient.
x = torch.tensor([3.0])  # Fixed input; requires_grad defaults to False.
# Compute the model prediction from input, weight, and bias.
pred = w * x + b
# 10 is the deliberately chosen toy target, not a Fahrenheit label.
loss = (pred - 10.0)**2  # Score the squared prediction error against the target.
loss.backward()  # Follow the graph to fill w.grad and b.grad.
print("w.grad:", w.grad, " b.grad:", b.grad)
# Apply the chain-rule formulas using the input and prediction values.
print("hand dL/dw:", 2*(pred.item()-10)*x.item(),
      " hand dL/db:", 2*(pred.item()-10))
PY
```

### Mean of the per-example gradients

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
# Repeat this environment's random starting parameters.
torch.manual_seed(0)
# Six Celsius inputs, one temperature per row: shape (6, 1).
C = torch.tensor([[-40.],[-10.],[0.],[20.],[37.],[100.]])
F = C * 1.8 + 32  # Build the six target Fahrenheit answers, shape (6, 1).
w = torch.randn(1, requires_grad=True)  # One random weight, tracking enabled.
b = torch.randn(1, requires_grad=True)  # One random bias, tracking enabled.
# Compute the model prediction from input, weight, and bias.
pred = w * C + b  # The shape-(1,) parameters serve all six rows.
# Calculate each example's weight-gradient contribution: 2 * error * input.
contrib = (2.0 * (pred - F) * C)
print("per-example weight-grad contributions:")
# detach() drops tracking for this view; flatten() lays six rows out as (6,).
print(contrib.detach().flatten())
# Average the contributions to compare with the full-batch weight gradient.
print("mean =", contrib.mean().item())
PY
```

### Update failure without `no_grad()`

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
w = torch.tensor([2.0], requires_grad=True)  # One tracked value, shape (1,).
loss = (w * 3.0 - 1.0)**2  # Score squared prediction error for input 3, target 1.
loss.backward()  # Compute the gradient and store it in w.grad.
try:  # Demonstrate the failure when updating a tracked leaf without no_grad.
    w -= 0.1 * w.grad  # Attempt an in-place gradient update with learning rate 0.1.
except RuntimeError as e:
    print("without no_grad:")  # Label the failed attempt.
    print(str(e))  # Show PyTorch's reason for rejecting the update.
with torch.no_grad():  # Keep the parameter update out of the gradient graph.
    w -= 0.1 * w.grad  # The same update is now allowed.
print("with no_grad, new w:", w.item())
# Verify the weight still requires gradients for future training steps.
print("w.requires_grad still:", w.requires_grad)
PY
```

### Gradient accumulation and clearing

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
w = torch.tensor([2.0], requires_grad=True)  # One tracked value, shape (1,).
# Run two backward passes without clearing gradients or updating the weight.
for i in range(1, 3):
    loss = w * 3.0  # Use a simple loss whose weight gradient is always 3.
    loss.backward()  # Add the new gradient to whatever w.grad already holds.
    print("backward", i, "-> w.grad:", w.grad.item())
w.grad.zero_()  # Clear the accumulated gradient before the next backward pass.
loss = w * 3.0  # Make a fresh forward calculation with the unchanged weight.
loss.backward()  # Accumulate onto zero this time.
print("after zero_ and fresh backward -> w.grad:", w.grad.item())
PY
```

### Original seeded dropout check

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
from torch import nn
torch.manual_seed(0)  # Repeat this environment's random dropout choices.
# Give each input value a 50% chance of being zeroed during training.
drop = nn.Dropout(p=0.5)
x = torch.ones(8)  # Eight ones, shape (8,); requires_grad defaults to False.
drop.train()  # Enable random dropping and scaling of surviving values.
print("train mode :", drop(x))  # Show which values survived and their scaling.
drop.eval()  # Switch that same layer to evaluation behavior.
print("eval  mode :", drop(x))  # Run it again on the unchanged input.
PY
```

### Deterministic mode and gradient lifecycle check

```bash
ssh spark '~/venvs/w1/bin/python -' <<'PY'
import torch
from torch import nn

# Drop every value during training to make the mode comparison deterministic.
drop = nn.Dropout(p=1.0)
x = torch.ones(3, requires_grad=True)  # Three ones, shape (3,), tracking on.
drop.eval()  # Switch the layer to evaluation behavior.
y = drop(x)  # Evaluation passes the input through unchanged.
# Check module mode and gradient tracking separately.
print("eval:", "training=", drop.training, "requires_grad=", y.requires_grad)
print("values:", y)  # Verify evaluation preserved the input values.
drop.train()  # Switch the same layer back to training behavior.
with torch.no_grad():  # Disable gradient recording without changing module mode.
    y = drop(x)  # Training dropout still zeros the same input.
# Check that no_grad left training mode enabled but suppressed tracking.
print("train + no_grad:", "training=", drop.training,
      "requires_grad=", y.requires_grad)
print("values:", y)  # Show whether training behavior still zeroed the values.

w = torch.tensor([2.0], requires_grad=True)  # One tracked value, shape (1,).
# Repeat with the same weight, clearing gradients between fresh graphs.
for step in range(1, 3):
    loss = (w * 3.0) ** 2  # Build a fresh graph for the same squared loss.
    print("forward", step, "grad_fn:", type(loss.grad_fn).__name__)
    loss.backward()  # Compute and accumulate the new weight gradient.
    print("after backward:", w.grad.item())  # Report the newly computed gradient.
    w.grad.zero_()  # Clear the gradient in place, without changing w.
    print("after zero_:", w.grad.item(), "w:", w.item())  # Show both values.
PY
```
