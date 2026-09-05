# Observations

- The model started with a random weight near -0.007 and a bias near 0.54,
  nowhere close to the true 1.8 and 32. Those come from `torch.manual_seed(0)`.
- Every one of the six initial predictions was far off (at 100 C the model
  guessed -0.212 instead of 212). Squaring and averaging those errors gives the
  first loss, 10352.21, in squared Fahrenheit units.
- The first gradients were large and negative (weight -9237.2, bias -127.4). The
  weight gradient is large because the raw Celsius inputs are large.
- The first update is exact arithmetic: weight = -0.007487 - 0.0003 * -9237.2129
  = 2.7637, which matches epoch 1's weight.
- The loss (recomputed after each update) fell fast then slowed: 1992 at epoch 1,
  806 at epoch 50, 303 at epoch 1000, 4.92 at epoch 5000, and about 0 by 20000.
- The weight moved close to its target within the first handful of updates; the
  bias was the slow one, crawling from 0.57 up to 31.9981 over thousands of
  epochs.
- The final weight and bias were 1.8000 and 31.9981 (final loss 2.9e-06), close
  to the true rule but not perfectly exact. Running to 40000 or 100000 epochs
  gives the same values because the updates round away in FP32.
- The saved `state_dict` reproduced the result: 25 C predicted 76.9985 F, which
  rounds to the true 77.
- Raw inputs force a small learning rate. At learning rate 0.1 the loss explodes
  (1e4, 1.9e9, 3.8e14, up to 6.2e35 by epoch 7, then `inf` by epoch 8).
