# Observations

- The weight and bias are plain tensors with `requires_grad=True`, not an
  `nn.Linear` module. There is no `torch.optim` optimizer either.
- The loss carries a `grad_fn` (`MeanBackward0`), which is PyTorch remembering
  how the loss was built. That record is the computational graph `backward()`
  walks.
- In the new run, `backward()` and the hand-computed derivative agree in the raw
  JSON: weight -2314.640380859375 and bias -73.82466125488281. This checks these
  two derivatives at the initial parameters, not all floating-point executions.
- The hand-written update `param -= lr * param.grad`, wrapped in
  `torch.no_grad()`, implements the same mathematical plain-SGD rule as Week 6.
- The new run's final post-update loss is 2.9371251457632752e-06, not zero.
  The console's four-decimal `0.0000` is display rounding. The parameters ended
  at weight 1.800014615058899 and bias 31.998149871826172.
- The before/after is explicit: weight started at 1.540996 and ended at 1.8000;
  bias started at -0.293429 and ended at 31.9981.
- A fresh process loaded only the checkpoint's `weight` and `bias`, both shape
  `(1,)` with `requires_grad=False`. Under `torch.no_grad()`, a `(1, 1)` input
  for 25 C produced 76.99851989746094 F, close to the known answer of 77 F.
- `results.json` contains the new Spark run's unrounded values. The original
  rounded baseline is preserved in `results-historical.json`; its transcript
  remains in the historical section of `system-report.md`. The displayed
  measurements agree at the original precision. The old raw floats were not
  saved, so bitwise agreement with the old run cannot be established.
