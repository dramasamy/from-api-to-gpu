#!/usr/bin/env bash
# Create the Week 3 Python environment and install pinned dependencies.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/w3}"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet huggingface-hub==1.23.0
