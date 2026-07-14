#!/usr/bin/env bash
# Week 2 - call the Ollama generate API and pretty-print the full JSON.
# Run on the Spark (localhost:11434). MODEL and PROMPT are overridable.
set -euo pipefail

MODEL="${MODEL:-phi4}"
PROMPT="${PROMPT:-Explain Kubernetes scheduling in three sentences.}"

curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false
}" | jq
