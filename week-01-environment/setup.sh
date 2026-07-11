#!/usr/bin/env bash
# Week 1 — create an isolated venv and install PyTorch on the DGX Spark.
#
# Run this ON the Spark (clone the repo there, then `./setup.sh`), or drive it
# from your control machine with:
#   ssh spark 'bash -s' < setup.sh
#
# Never install into system Python — always use a venv.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/w1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"

echo "--- venv ready at $VENV ---"
"$VENV/bin/python" --version
