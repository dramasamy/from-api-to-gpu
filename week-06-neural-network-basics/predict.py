#!/usr/bin/env python3
"""Week 6 - load the trained checkpoint and run one inference.

train_tiny.py trains the Celsius-to-Fahrenheit model and saves its parameters to
a .pt file. This script is the other half, with no training code at all: rebuild
the same model shape, load the saved parameters into it, and make a prediction.
Run train_tiny.py first so the checkpoint exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("celsius_to_fahrenheit.pt"))
    parser.add_argument("--celsius", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build the SAME architecture the checkpoint was trained with. A state_dict
    # stores only numbers, not the model shape, so the layer must exist first.
    model = nn.Linear(1, 1)
    # Load the saved weight and bias from the .pt file into this layer.
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()  # inference mode (no visible effect here, but the right habit)

    print(f"loaded {args.checkpoint}")
    print(f"weight={model.weight.item():.4f} bias={model.bias.item():.4f}")

    # One input, wrapped to shape 1 row by 1 column like the training data.
    celsius = torch.tensor([[args.celsius]])
    with torch.no_grad():  # predicting only, so do not track gradients
        fahrenheit = model(celsius).item()
    print(f"inference: {args.celsius:.1f} C -> {fahrenheit:.4f} F")


if __name__ == "__main__":
    main()
