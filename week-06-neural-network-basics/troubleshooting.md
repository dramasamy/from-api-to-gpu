# Troubleshooting

## A large learning rate diverges on raw Celsius values

The raw Celsius inputs range up to 100. With mean-squared-error loss, those large
inputs make the first gradient large (about -9237 for the weight). At learning
rate 0.1 the optimizer overshoots so far the loss explodes:

```bash
ssh spark '~/venvs/w1/bin/python - --lr 0.1 --epochs 6' \
    < train_tiny.py 2>&1 | grep 'epoch     1'
```

```text
epoch     1 loss=1907324800.0000 weight=923.7138 bias=13.2759
```

By epoch 6 the loss is about 6e35 and the values eventually become `inf` or
`nan`. The fix used in the lab is a small learning rate (0.0003). In real
training the more common fix is to scale the inputs to a small range so the
gradients stay small and a normal learning rate works.

## An earlier version scaled the inputs and confused two weights

A first version divided the inputs by 100 to keep SGD stable, then multiplied the
learned weight back by 100 for display. That worked numerically but created two
meanings of "weight": the model's internal parameter and the display value. The
printed gradient belonged to the internal parameter, so the first-update
arithmetic did not line up with the printed weight. Switching to raw inputs with
a small learning rate removed the confusion; the printed weight is now the real
parameter.

## Reproducibility

The script calls `torch.manual_seed(seed)`, so the initial random weight and
bias, and therefore the whole run, are the same each time for a given seed. On
this Spark with torch 2.13.0+cu130, seed 0 twice gave the same displayed values,
and seeds 1 and 2 reached the same displayed 1.8 and 32 because this task has a
single best fit.

## The final loss is tiny but not zero

The final loss is about 2.9e-06, and the learned bias is 31.9981, not exactly 32.
The run stops at 20000 epochs, but running to 40000 or 100000 epochs gives the
exact same values: the remaining SGD updates are smaller than the spacing FP32
can represent, so the parameters stop changing. The task can be represented
exactly by a linear layer, but this run plateaus just short of it in FP32.
