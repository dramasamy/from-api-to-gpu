#!/usr/bin/env python3
"""Week 7 - the training loop written by hand for Celsius to Fahrenheit.

Week 6 used nn.Linear and torch.optim.SGD. This version removes both. The weight
and bias are plain tensors with requires_grad=True, and the parameter update is
one line I write myself: param -= lr * param.grad. It uses the same mathematical
forward rule and plain-SGD update as Week 6, from a different random start.
This is not a claim of bitwise-identical floating-point results.

The problem is identical to Week 6: learn F = C * 1.8 + 32 from six examples,
with raw Celsius inputs and a small learning rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    # Configure the training run and where to save its checkpoint and results.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("manual_celsius_to_fahrenheit.pt"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Make the random starting parameters repeatable in this environment.
    torch.manual_seed(args.seed)

    # Six Celsius inputs, one temperature per row: shape (6, 1).
    celsius = torch.tensor([[-40.0], [-10.0], [0.0], [20.0],
                            [37.0], [100.0]])
    # Labels are the six correct answers, also shape (6, 1). Only label creation
    # uses the known formula; training receives inputs and labels, not the rule.
    # Inputs and labels are fixed data; neither needs gradients.
    fahrenheit = celsius * 1.8 + 32.0

    # Each parameter has shape (1,): one randomly drawn number. requires_grad
    # asks PyTorch to track operations so backward() can compute derivatives.
    weight = torch.randn(1, requires_grad=True)
    # Draw a separate one-value bias with tracking enabled too.
    bias = torch.randn(1, requires_grad=True)
    print(f"device={weight.device}")
    # item() extracts a one-element tensor as a Python number for reporting.
    init_w = weight.item()
    init_b = bias.item()
    print(f"before training: weight={init_w:.6f} bias={init_b:.6f}")

    # One manual forward and backward to show two things:
    #   1. loss carries a grad_fn, so it remembers how it was built.
    #   2. backward() and a hand-written derivative can be compared numerically.
    # Compute the model prediction from input, weight, and bias.
    # Broadcasting reuses each shape-(1,) parameter across all six input rows.
    pred = weight * celsius + bias
    # Score the squared prediction error against each target, then average
    # all six errors into one scalar loss, shape ().
    loss = ((pred - fahrenheit) ** 2).mean()
    print(f"loss.grad_fn type={type(loss.grad_fn).__name__}")
    # Follow the recorded graph to fill the two parameters' shape-(1,) .grad.
    loss.backward()
    # Store the starting loss as a Python number for JSON, before any updates.
    first_loss = loss.item()
    # Check the MSE derivatives without building another gradient graph:
    # dloss/dweight = mean(2 * error * input), dloss/dbias = mean(2 * error).
    with torch.no_grad():
        # Average the six weight contributions, then extract the scalar number.
        hand_w_grad = (2.0 * (pred - fahrenheit) * celsius).mean().item()
        # Bias contributions omit the input multiplier; average those too.
        hand_b_grad = (2.0 * (pred - fahrenheit)).mean().item()
    print(f"first grad backward(): weight={weight.grad.item():.4f} "
          f"bias={bias.grad.item():.4f}")
    print(f"first grad by hand   : weight={hand_w_grad:.4f} "
          f"bias={hand_b_grad:.4f}")
    # Save both gradients before clearing them; .grad holds a tensor.
    first_w_grad = weight.grad.item()
    first_b_grad = bias.grad.item()
    # Discard this demonstration's gradients before the actual training loop.
    weight.grad.zero_()
    bias.grad.zero_()

    # Keep selected post-update measurements, not every training step.
    history = []
    # Report progress at these epochs, including the final requested epoch.
    log_at = {1, 50, 1000, 5000, 10000, args.epochs}
    # Each epoch uses all six examples for one parameter update.
    for epoch in range(1, args.epochs + 1):
        # Compute the model prediction from input, weight, and bias.
        # The forward pass produces six predictions, shape (6, 1).
        pred = weight * celsius + bias
        # Score the squared prediction error against each target, then average.
        loss = ((pred - fahrenheit) ** 2).mean()
        # Backward pass fills weight.grad and bias.grad.
        loss.backward()
        # Move each parameter opposite its gradient, scaled by the learning
        # rate. This is the mathematical plain-SGD rule, using the full batch.
        # no_grad keeps the in-place update out of the gradient graph.
        with torch.no_grad():
            # Adjust the weight using its gradient and the learning rate.
            weight -= args.lr * weight.grad
            # Apply the same update rule to the bias, using its own gradient.
            bias -= args.lr * bias.grad
        # Clear the gradients, or the next backward() would add onto these.
        weight.grad.zero_()
        # Both accumulators must be cleared, not just the weight's.
        bias.grad.zero_()
        # Capture progress only at the selected reporting epochs.
        if epoch in log_at:
            # Measure again after the update, using the new weight and bias.
            with torch.no_grad():
                post_loss = ((weight * celsius + bias - fahrenheit) ** 2
                             ).mean().item()
            print(f"epoch {epoch:>5} loss={post_loss:>12.4f} "
                  f"weight={weight.item():.4f} bias={bias.item():.4f}")
            # Format stdout for reading, but retain unrounded floats in JSON.
            history.append({"epoch": epoch, "loss": post_loss,
                            "weight": weight.item(), "bias": bias.item()})

    final_w = weight.item()
    final_b = bias.item()
    # The loop's loss precedes its last update. Recompute the final measurement
    # from the saved parameters, without changing those parameters.
    with torch.no_grad():
        final_loss = ((weight * celsius + bias - fahrenheit) ** 2).mean().item()

    print(f"after training:  weight={final_w:.4f} bias={final_b:.4f} "
          f"(true weight 1.8, bias 32)")
    # Show the raw final loss so display rounding does not make it look zero.
    print(f"final loss (post-update, unrounded)={final_loss}")

    # Save a dictionary with two named shape-(1,) tensors. detach() removes
    # gradient tracking; the checkpoint contains values, not the training graph.
    # torch.save writes this dictionary to the path supplied by --checkpoint.
    torch.save({"weight": weight.detach(), "bias": bias.detach()},
               args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")

    # Tiny same-process reload smoke test. The README also tests a fresh process.
    # weights_only restricts loading to tensors and supported simple data types.
    state = torch.load(args.checkpoint, weights_only=True)
    # Do not record operations for this prediction-only check.
    with torch.no_grad():
        # One row, one input: shape (1, 1). The result also has one element.
        test_c = torch.tensor([[25.0]])
        # Predict Fahrenheit using the reloaded weight and bias, not training state.
        predicted = (state["weight"] * test_c + state["bias"]).item()
    print(f"inference: 25 C -> {predicted:.4f} F (true 77)")

    # Optionally save the run settings and measurements for later comparison.
    if args.output:
        # Keep Python float values as returned by item(), with no extra rounding.
        args.output.write_text(json.dumps({
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "initial_weight": init_w,
            "initial_bias": init_b,
            "first_loss": first_loss,
            "first_weight_grad_backward": first_w_grad,
            "first_weight_grad_by_hand": hand_w_grad,
            "first_bias_grad_backward": first_b_grad,
            "first_bias_grad_by_hand": hand_b_grad,
            "learned_weight": final_w,
            "learned_bias": final_b,
            "final_loss": final_loss,
            "history": history,
            "inference_25c_f": predicted,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
