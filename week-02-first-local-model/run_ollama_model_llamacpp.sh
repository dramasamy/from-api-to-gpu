#!/usr/bin/env bash
# Run llama.cpp against the exact GGUF model blob downloaded by Ollama.
set -euo pipefail

MODEL="${MODEL:-phi4}"
PROMPT="${PROMPT:-Reply with exactly these three words: shared model works}"
MODELS_DIR="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/.local/bin/llama-cli}"

NAME="${MODEL%%:*}"
TAG="${MODEL#*:}"
[ "$TAG" = "$MODEL" ] && TAG="latest"
echo '$ MANIFEST="$MODELS_DIR/manifests/registry.ollama.ai/library/$NAME/$TAG"'
MANIFEST="$MODELS_DIR/manifests/registry.ollama.ai/library/$NAME/$TAG"
cat <<'EOF'
$ MODEL_DIGEST="$(jq -r '[.layers[]
    | select(.mediaType == "application/vnd.ollama.image.model")
  ][0].digest' "$MANIFEST")"
EOF
MODEL_DIGEST="$(
  jq -r '
    [.layers[]
      | select(.mediaType == "application/vnd.ollama.image.model")
    ][0].digest
  ' "$MANIFEST"
)"
echo '$ MODEL_BLOB="$MODELS_DIR/blobs/${MODEL_DIGEST/:/-}"'
MODEL_BLOB="$MODELS_DIR/blobs/${MODEL_DIGEST/:/-}"

if [ ! -x "$LLAMA_CLI" ]; then
  echo "llama-cli not found: $LLAMA_CLI" >&2
  exit 1
fi
if [ ! -f "$MODEL_BLOB" ]; then
  echo "Model blob not found: $MODEL_BLOB" >&2
  exit 1
fi

echo '$ echo "model: $MODEL"'
echo "model: $MODEL"
echo '$ echo "model digest: $MODEL_DIGEST"'
echo "model digest: $MODEL_DIGEST"
echo '$ LLAMA_VERSION="$($LLAMA_CLI --version 2>&1 | head -n 1)"'
LLAMA_VERSION="$($LLAMA_CLI --version 2>&1 | head -n 1)"
echo '$ echo "llama.cpp: ${LLAMA_VERSION#version: }"'
echo "llama.cpp: ${LLAMA_VERSION#version: }"
echo '$ "$LLAMA_CLI" --list-devices'
"$LLAMA_CLI" --list-devices
echo
echo "=== generation from Ollama model blob ==="
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT
cat <<'EOF'
$ "$LLAMA_CLI" \
    --model "$MODEL_BLOB" \
    --prompt "$PROMPT" \
    --conversation --single-turn --simple-io --no-display-prompt \
    --ctx-size 2048 --n-gpu-layers 99 \
    --seed 42 --temperature 0 --n-predict 32 --no-warmup \
    --verbose --log-colors off --no-log-prefix --no-log-timestamps \
    >"$LOG_FILE" 2>&1
EOF
"$LLAMA_CLI" \
  --model "$MODEL_BLOB" \
  --prompt "$PROMPT" \
  --conversation \
  --single-turn \
  --simple-io \
  --no-display-prompt \
  --ctx-size 2048 \
  --n-gpu-layers 99 \
  --seed 42 \
  --temperature 0 \
  --n-predict 32 \
  --no-warmup \
  --verbose \
  --log-colors off \
  --no-log-prefix \
  --no-log-timestamps \
  >"$LOG_FILE" 2>&1

cat <<'EOF'
$ grep 'using device CUDA0' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*using device ([^ ]+) \(([^)]+)\).*/device: \1 (\2)/'
EOF
grep 'using device CUDA0' "$LOG_FILE" | tail -n 1 \
  | sed -E 's/.*using device ([^ ]+) \(([^)]+)\).*/device: \1 (\2)/'
cat <<'EOF'
$ grep 'offloaded .* layers to GPU' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*offloaded ([0-9]+\/[0-9]+) layers.*/GPU layers: \1/'
EOF
grep 'offloaded .* layers to GPU' "$LOG_FILE" | tail -n 1 \
  | sed -E 's/.*offloaded ([0-9]+\/[0-9]+) layers.*/GPU layers: \1/'
cat <<'EOF'
$ grep 'CPU_Mapped model buffer size' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*size = *([^ ]+ MiB).*/CPU mapped model buffer: \1/'
EOF
grep 'CPU_Mapped model buffer size' "$LOG_FILE" | tail -n 1 \
  | sed -E 's/.*size = *([^ ]+ MiB).*/CPU mapped model buffer: \1/'
cat <<'EOF'
$ grep 'CUDA0 model buffer size' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*size = *([^ ]+ MiB).*/CUDA0 model buffer: \1/'
EOF
grep 'CUDA0 model buffer size' "$LOG_FILE" | tail -n 1 \
  | sed -E 's/.*size = *([^ ]+ MiB).*/CUDA0 model buffer: \1/'
cat <<'EOF'
$ grep 'Parsed message: {"role":"assistant"' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*"content":"(.*)"}/assistant: \1/'
EOF
grep 'Parsed message: {"role":"assistant"' "$LOG_FILE" | tail -n 1 \
  | sed -E 's/.*"content":"(.*)"}/assistant: \1/'
cat <<'EOF'
$ grep '^\[ Prompt:' "$LOG_FILE" | tail -n 1 \
  | sed -E -e 's/^\[ Prompt: /speed: prompt /' \
        -e 's/ \| Generation: /, generation /' \
        -e 's/\]$//' -e 's/[[:space:]]*$//'
EOF
grep '^\[ Prompt:' "$LOG_FILE" | tail -n 1 \
  | sed -E -e 's/^\[ Prompt: /speed: prompt /' \
      -e 's/ \| Generation: /, generation /' \
      -e 's/\]$//' -e 's/[[:space:]]*$//'
