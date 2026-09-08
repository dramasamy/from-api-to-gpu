# Troubleshooting

## The update must run inside `torch.no_grad()`

The line `weight -= lr * weight.grad` changes a leaf tensor in place. Here the
weight is a leaf (not the result of a tracked operation) with
`requires_grad=True`. In normal gradient mode PyTorch rejects this in-place
update. Wrapping it in `torch.no_grad()` permits the update without recording
the update arithmetic in the gradient graph.

## Gradients must be cleared each step

`backward()` adds new gradients onto whatever is already in `param.grad`.
Recomputing the same loss at unchanged parameters and calling `backward()` again
without clearing adds the same contribution twice. Calling `backward()` twice
on the very same graph normally fails because the first call frees saved
intermediates, unless that graph was retained. The loop calls
`weight.grad.zero_()` and `bias.grad.zero_()` after each update so the next
`backward()` starts from zero. A direct demonstration is in the blog.

## Leaf tensors receive `.grad` by default

The Celsius inputs have `requires_grad=False`, so they never receive a gradient.
PyTorch stores `.grad` for leaf tensors (not results of tracked operations) with
`requires_grad=True`, which here is the weight and bias. Intermediate tensors like
`pred` are non-leaf and do not keep a `.grad` unless you call `retain_grad()`. If
no tensor in the expression tracks gradients, `backward()` raises an error rather
than quietly leaving `.grad` as `None`.

## Reproducibility

The script calls `torch.manual_seed(seed)` to control its random starting
parameters. The captured run used Spark's CPU with torch 2.13.0+cu130 on aarch64.
A seed alone does not guarantee bitwise-identical results across PyTorch
versions, hardware, or different implementations of the same SGD mathematics.
Compare raw values in `results.json`, not just four-decimal console output.

## Checkpoint paths and loading

SSH runs the program on Spark, so relative checkpoint and output paths belong
to Spark's working directory, not the local lab directory. Follow the README's
fresh temporary-directory workflow to avoid overwriting an earlier run.
The checkpoint contains detached parameter tensors, not the training graph or
an `nn.Module`. Use `weights_only=True` when loading this tensor dictionary;
load only checkpoints from a trusted source. `torch.no_grad()` disables gradient
recording for inference. There is no module here on which to call `eval()`.
