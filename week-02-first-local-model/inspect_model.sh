#!/usr/bin/env bash
# Week 2 - inspect a model and its Ollama manifest.
# Run on the Spark (Ollama + models present). MODEL defaults to phi4.
set -euo pipefail

MODEL="${MODEL:-phi4}"
NAME="${MODEL%%:*}"
TAG="${MODEL#*:}"; [ "$TAG" = "$MODEL" ] && TAG="latest"
MODELS_DIR="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"

echo "=== ollama list ==="
ollama list

echo
echo "=== $MODEL model metadata ==="
curl -s http://localhost:11434/api/show -d "{\"model\": \"$MODEL\"}" \
  | jq '{
      architecture: .details.family,
      parameters: .details.parameter_size,
      context_length: (
        .model_info | to_entries
        | map(select(.key | endswith(".context_length"))) | first.value
      ),
      embedding_length: (
        .model_info | to_entries
        | map(select(.key | endswith(".embedding_length"))) | first.value
      ),
      quantization: .details.quantization_level,
      capabilities,
      runtime_parameters: (
        .parameters
        | split("\n")
        | map(select(length > 0) | gsub(" +"; " "))
      )
    }'

echo
echo "=== Ollama manifest ==="
MANIFEST="$MODELS_DIR/manifests/registry.ollama.ai/library/$NAME/$TAG"
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi
jq . "$MANIFEST"
