#!/usr/bin/env python3
"""Week 6 - train the smallest useful neural network: Celsius to Fahrenheit.

The true rule is F = C * 1.8 + 32. A single linear layer y = w*x + b should
learn a weight near 1.8 and a bias near 32. The point is not the model. It is to
watch the weight and bias start random, the loss fall, the gradients drive the
change, and the trained parameters get saved to a .pt checkpoint. A separate
predict.py loads that checkpoint to run inference.

The inputs are raw Celsius, so the printed weight is the model's real parameter.
Because the raw inputs are large, the first gradient is large too, which is why
the learning rate has to be small (0.0003). A large learning rate diverges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)  # learning rate 0.0003
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("celsius_to_fahrenheit.pt"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Six training inputs in Celsius. Each inner [ ] is one example with one
    # value, so the shape is 6 rows by 1 column (six examples, one feature).
    celsius = torch.tensor([[-40.0], [-10.0], [0.0], [20.0],
                            [37.0], [100.0]])
    # The correct answer for each input. This is the only place the true rule
    # appears, and it only builds the labels; the model never sees it.
    fahrenheit = celsius * 1.8 + 32.0

    model = nn.Linear(1, 1)  # one layer: prediction = weight * C + bias
    loss_fn = nn.MSELoss()  # scores how wrong the predictions are
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # updates w, b

    print(f"device={model.weight.device}")
    init_w = model.weight.item()
    init_b = model.bias.item()
    print(f"initial weight={init_w:.6f} bias={init_b:.6f}")

    # Show where the first loss comes from: predict, then average the
    # squared errors across the six examples.
    with torch.no_grad():
        first_preds = model(celsius)
    print("initial predictions vs targets:")
    for c, p, f in zip(celsius.tolist(), first_preds.tolist(),
                       fahrenheit.tolist()):
        print(f"  C={c[0]:>6.1f}  pred={p[0]:>8.3f}  target={f[0]:>7.1f}")

    # One manual step to expose the first gradient before the optimizer moves.
    optimizer.zero_grad()
    first_loss_t = loss_fn(model(celsius), fahrenheit)
    first_loss_t.backward()
    first_loss = first_loss_t.item()
    w_grad = model.weight.grad.item()
    b_grad = model.bias.grad.item()
    print(f"first loss={first_loss:.4f} "
          f"weight_grad={w_grad:.4f} bias_grad={b_grad:.4f}")
    print(f"first weight update: {init_w:.6f} - {args.lr} * {w_grad:.4f} "
          f"= {init_w - args.lr * w_grad:.6f}")

    history = []
    # log_at picks the handful of epochs to print, so the output stays short.
    log_at = {1, 50, 200, 1000, 5000, 10000, args.epochs}
    for epoch in range(1, args.epochs + 1):
        # One pass of the four steps from the intro (plus a housekeeping reset):
        optimizer.zero_grad()  # reset: clear gradients left from the last step
        loss = loss_fn(model(celsius), fahrenheit)  # steps 1-2: predict, then score
        loss.backward()  # step 3: backprop fills each parameter's gradient
        optimizer.step()  # step 4: nudge weight and bias down the loss
        if epoch in log_at:
            # This block only logs progress; the learning already happened above.
            # Recompute the loss after the step so the printed loss and the
            # printed weight and bias all describe the same post-update model.
            with torch.no_grad():
                post_loss = loss_fn(model(celsius), fahrenheit).item()
            w = model.weight.item()  # current weight, as a plain Python float
            b = model.bias.item()  # current bias
            print(f"epoch {epoch:>5} loss={post_loss:>12.4f} "
                  f"weight={w:.4f} bias={b:.4f}")
            # Keep this snapshot so results.json can chart the run later.
            history.append({"epoch": epoch, "loss": round(post_loss, 4),
                            "weight": round(w, 4), "bias": round(b, 4)})

    # Training is finished. Read the final learned weight and bias, and the
    # loss of the finished model, which is now tiny.
    final_w = model.weight.item()
    final_b = model.bias.item()
    with torch.no_grad():
        final_loss = loss_fn(model(celsius), fahrenheit).item()
    print(f"\nlearned weight={final_w:.4f} bias={final_b:.4f} "
          f"final_loss={final_loss:.4e} (true weight 1.8, bias 32)")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"saved state_dict to {args.checkpoint}")

    if args.output:
        args.output.write_text(json.dumps({
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "initial_weight": round(init_w, 6),
            "initial_bias": round(init_b, 6),
            "first_loss": round(first_loss, 4),
            "first_weight_grad": round(w_grad, 4),
            "first_bias_grad": round(b_grad, 4),
            "learned_weight": round(final_w, 4),
            "learned_bias": round(final_b, 4),
            "final_loss": final_loss,
            "history": history,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
