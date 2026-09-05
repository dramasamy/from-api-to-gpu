# System report

```text
$ ~/venvs/w1/bin/python -c 'import torch; print(torch.__version__)'
2.13.0+cu130

$ ~/venvs/w1/bin/python - < train_tiny.py
device=cpu
initial weight=-0.007487 bias=0.536444
initial predictions vs targets:
  C= -40.0  pred=   0.836  target=  -40.0
  C= -10.0  pred=   0.611  target=   14.0
  C=   0.0  pred=   0.536  target=   32.0
  C=  20.0  pred=   0.387  target=   68.0
  C=  37.0  pred=   0.259  target=   98.6
  C= 100.0  pred=  -0.212  target=  212.0
first loss=10352.2070 weight_grad=-9237.2129 bias_grad=-127.3941
first weight update: -0.007487 - 0.0003 * -9237.2129 = 2.763677
epoch     1 loss=   1992.1444 weight=2.7637 bias=0.5747
epoch    50 loss=    806.2087 weight=2.0436 bias=1.3524
epoch   200 loss=    690.7745 weight=2.0255 bias=3.6312
epoch  1000 loss=    302.9773 weight=1.9493 bias=13.2121
epoch  5000 loss=      4.9179 weight=1.8190 bias=29.6063
epoch 10000 loss=      0.0285 weight=1.8014 bias=31.8178
epoch 20000 loss=      0.0000 weight=1.8000 bias=31.9981

learned weight=1.8000 bias=31.9981 final_loss=2.9334e-06 (true weight 1.8, bias 32)
saved state_dict to celsius_to_fahrenheit.pt
inference: 25 C -> 76.9985 F (true 77)
```

The script prints `device=cpu`, so this run was intentionally on the CPU. The
task is small enough that the GPU is not needed here.
