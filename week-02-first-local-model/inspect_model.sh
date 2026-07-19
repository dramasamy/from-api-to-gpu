#!/usr/bin/env bash
# Week 2 - inspect a model, its Ollama manifest, and its model blob format.
# Run on the Spark (Ollama + models present). MODEL defaults to phi4.
set -euo pipefail

MODEL="${MODEL:-phi4}"
NAME="${MODEL%%:*}"
TAG="${MODEL#*:}"; [ "$TAG" = "$MODEL" ] && TAG="latest"
MODELS_DIR="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/.local/bin/llama-cli}"

echo "=== ollama list ==="
echo '$ ollama list | sed '\''s/[[:space:]]*$//'\'''
ollama list | sed 's/[[:space:]]*$//'

echo
echo "=== $MODEL model metadata ==="
cat <<EOF
\$ curl -s http://localhost:11434/api/show -d '{"model": "$MODEL"}' | jq '{
    format: .details.format,
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
      | split("\\n")
      | map(select(length > 0) | gsub(" +"; " "))
    )
  }'
EOF
curl -s http://localhost:11434/api/show -d "{\"model\": \"$MODEL\"}" \
  | jq '{
  format: .details.format,
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
echo '$ MANIFEST_ROOT="$MODELS_DIR/manifests/registry.ollama.ai/library"'
MANIFEST_ROOT="$MODELS_DIR/manifests/registry.ollama.ai/library"
echo '$ MANIFEST="$MANIFEST_ROOT/$NAME/$TAG"'
MANIFEST="$MANIFEST_ROOT/$NAME/$TAG"
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi
cat <<'EOF'
$ printf "manifest root: %s\nmanifest model/tag: %s/%s\n" \
  "$MANIFEST_ROOT" "$NAME" "$TAG"
EOF
printf "manifest root: %s\nmanifest model/tag: %s/%s\n" \
  "$MANIFEST_ROOT" "$NAME" "$TAG"
echo '$ jq . "$MANIFEST"'
jq . "$MANIFEST"

echo
echo "=== connect the manifest to the model blob ==="
cat <<'EOF'
$ MODEL_LAYER="$(jq -c '[.layers[]
    | select(.mediaType == "application/vnd.ollama.image.model")
  ][0]' "$MANIFEST")"
EOF
MODEL_LAYER="$(
  jq -c '
    [.layers[]
      | select(.mediaType == "application/vnd.ollama.image.model")
    ][0]
  ' "$MANIFEST"
)"
echo '$ echo "$MODEL_LAYER" | jq .'
echo "$MODEL_LAYER" | jq .
echo '$ MODEL_DIGEST="$(jq -r .digest <<<"$MODEL_LAYER")"'
MODEL_DIGEST="$(jq -r .digest <<<"$MODEL_LAYER")"
echo '$ MODEL_SIZE="$(jq -r .size <<<"$MODEL_LAYER")"'
MODEL_SIZE="$(jq -r .size <<<"$MODEL_LAYER")"
echo '$ BLOB_FILENAME="${MODEL_DIGEST/:/-}"'
BLOB_FILENAME="${MODEL_DIGEST/:/-}"
echo '$ MODEL_BLOB="$MODELS_DIR/blobs/$BLOB_FILENAME"'
MODEL_BLOB="$MODELS_DIR/blobs/$BLOB_FILENAME"
if [ ! -f "$MODEL_BLOB" ]; then
  echo "Model blob not found: $MODEL_BLOB" >&2
  exit 1
fi
cat <<'EOF'
$ printf "manifest digest: %s\nblob filename: %s\n" \
    "$MODEL_DIGEST" "$BLOB_FILENAME"
EOF
printf "manifest digest: %s\nblob filename: %s\n" \
  "$MODEL_DIGEST" "$BLOB_FILENAME"
echo '$ ACTUAL_SIZE="$(stat -c %s "$MODEL_BLOB")"'
ACTUAL_SIZE="$(stat -c %s "$MODEL_BLOB")"
cat <<'EOF'
$ printf "manifest bytes: %s\nfile bytes: %s\n" \
    "$MODEL_SIZE" "$ACTUAL_SIZE"
EOF
printf "manifest bytes: %s\nfile bytes: %s\n" "$MODEL_SIZE" "$ACTUAL_SIZE"
echo '$ test "$MODEL_SIZE" = "$ACTUAL_SIZE" && echo "size check: MATCH"'
test "$MODEL_SIZE" = "$ACTUAL_SIZE" && echo "size check: MATCH"
echo '$ EXPECTED_HASH="${MODEL_DIGEST#sha256:}"'
EXPECTED_HASH="${MODEL_DIGEST#sha256:}"
echo '$ ACTUAL_HASH="$(sha256sum "$MODEL_BLOB" | awk '\''{print $1}'\'')"'
ACTUAL_HASH="$(sha256sum "$MODEL_BLOB" | awk '{print $1}')"
echo '$ test "$EXPECTED_HASH" = "$ACTUAL_HASH" && echo "SHA-256 check: MATCH"'
test "$EXPECTED_HASH" = "$ACTUAL_HASH" && echo "SHA-256 check: MATCH"
echo '$ xxd -l 4 -g 1 "$MODEL_BLOB"'
xxd -l 4 -g 1 "$MODEL_BLOB"
if [ -x "$LLAMA_CLI" ]; then
  echo '$ test -x "$LLAMA_CLI" && echo "$LLAMA_CLI"'
  echo "$LLAMA_CLI"
elif command -v llama-cli >/dev/null; then
  echo '$ command -v llama-cli'
  command -v llama-cli
else
  echo '$ command -v llama-cli || test -x "$LLAMA_CLI"'
  echo "not installed"
fi
