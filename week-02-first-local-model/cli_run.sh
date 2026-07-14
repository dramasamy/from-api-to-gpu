#!/usr/bin/env bash
# Run one prompt through Ollama's terminal-aware CLI.
set -euo pipefail

MODEL="${MODEL:-phi4}"
PROMPT="${PROMPT:-Reply with exactly these three words: local model ready}"

ollama run --nowordwrap "$MODEL" "$PROMPT"
