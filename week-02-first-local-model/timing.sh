#!/usr/bin/env bash
# Week 2 - measure a cold-load call: load time, tokens, and tokens/sec.
# Stops the model first so load_duration reflects a real cold start.
# Run on the Spark. MODEL and PROMPT are overridable.
set -euo pipefail

MODEL="${MODEL:-phi4}"
PROMPT="${PROMPT:-Explain Kubernetes scheduling in three sentences.}"

ollama stop "$MODEL" 2>/dev/null || true
sleep 1

curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false
}" | python3 -c '
import sys, json
d = json.load(sys.stdin)
ns = 1e9
load = d["load_duration"] / ns
peval = d["prompt_eval_duration"] / ns
evald = d["eval_duration"] / ns
pc = d["prompt_eval_count"]
ec = d["eval_count"]
tps = ec / evald if evald else 0
print("load_duration     %8.2f s   (cold model load)" % load)
print("prompt_eval_count %8d     tokens" % pc)
print("prompt_eval       %8.1f ms  (prefill)" % (peval * 1000))
print("eval_count        %8d     tokens generated" % ec)
print("eval_duration     %8.2f s   (decode)" % evald)
print("tokens_per_second %8.1f" % tps)
'
