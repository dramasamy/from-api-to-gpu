#!/usr/bin/env bash
# Week 2 - see how temperature changes output. Runs the same prompt twice at a
# given temperature and prints both replies, so you can watch determinism vs
# variation. Run on the Spark. Override MODEL, PROMPT, TEMP.
set -euo pipefail

MODEL="${MODEL:-phi4}"
PROMPT="${PROMPT:-Give me a one-sentence tagline for a coffee shop.}"
TEMP="${TEMP:-0}"

ask() {
  curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$PROMPT\",
    \"stream\": false,
    \"options\": { \"temperature\": $TEMP }
  }" | jq -r '.response'
}

echo "temperature=$TEMP"
echo "run 1: $(ask)"
echo "run 2: $(ask)"
