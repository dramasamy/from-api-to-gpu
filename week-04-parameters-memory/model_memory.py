#!/usr/bin/env python3
"""Estimate model weight memory from a parameter count and a precision.

The math is exact for a single uniform precision:

    weight_bytes = parameter_count * bits_per_parameter / 8

This is the tensor payload only. Real runtime memory is larger because of the
KV cache, activations, and framework overhead, and it cannot be derived from the
parameter count. The blog measures that runtime memory separately.
"""

from __future__ import annotations

import argparse

BITS_PER_PARAMETER = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
}

GB = 1_000_000_000
GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--parameters",
        type=float,
        help="Parameter count in billions, e.g. 70 or 3.2",
    )
    group.add_argument(
        "--count",
        type=int,
        help="Exact parameter count, e.g. 3085938688",
    )
    parser.add_argument(
        "--precision",
        choices=sorted(BITS_PER_PARAMETER),
        required=True,
    )
    return parser.parse_args()


def weight_bytes(parameter_count: int, precision: str) -> float:
    return parameter_count * BITS_PER_PARAMETER[precision] / 8


def main() -> None:
    args = parse_args()
    if args.count is not None:
        parameter_count = args.count
    else:
        parameter_count = round(args.parameters * 1_000_000_000)

    raw_bytes = weight_bytes(parameter_count, args.precision)

    print(f"parameter_count    {parameter_count:,}")
    print(f"precision          {args.precision}")
    print(f"bits_per_parameter {BITS_PER_PARAMETER[args.precision]}")
    print(f"weight_bytes       {raw_bytes:,.0f}")
    print(f"weight_gb          {raw_bytes / GB:.2f} GB (decimal, / 1e9)")
    print(f"weight_gib         {raw_bytes / GIB:.2f} GiB (binary, / 1024^3)")


if __name__ == "__main__":
    main()
