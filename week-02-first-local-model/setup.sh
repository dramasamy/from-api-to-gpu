#!/usr/bin/env bash
# Pull the model used throughout the week.
set -euo pipefail

MODEL="${MODEL:-phi4}"
ollama pull "$MODEL"
