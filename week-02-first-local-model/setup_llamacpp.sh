#!/usr/bin/env bash
# Build a pinned llama.cpp llama-cli with CUDA support on the DGX Spark.
set -euo pipefail

LLAMA_CPP_REF="${LLAMA_CPP_REF:-571d0d540df04f25298d0e159e520d9fc62ed121}"
SOURCE_DIR="${LLAMA_CPP_SOURCE_DIR:-$HOME/src/llama.cpp}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-13.0}"
CUDA_ARCH="${CUDA_ARCH:-121}"
INSTALL_BIN_DIR="${INSTALL_BIN_DIR:-$HOME/.local/bin}"
BUILD_LOG="$(mktemp)"
trap 'rm -f "$BUILD_LOG"' EXIT

if [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
  echo "CUDA compiler not found: $CUDA_ROOT/bin/nvcc" >&2
  exit 1
fi

mkdir -p "$(dirname "$SOURCE_DIR")" "$INSTALL_BIN_DIR"
if [ ! -d "$SOURCE_DIR/.git" ]; then
  cat <<EOF
\$ git clone --quiet --filter=blob:none \\
    https://github.com/ggml-org/llama.cpp.git \\
    "$SOURCE_DIR"
EOF
  git clone --quiet --filter=blob:none \
    https://github.com/ggml-org/llama.cpp.git \
    "$SOURCE_DIR"
fi

echo '$ git -C "$SOURCE_DIR" fetch --quiet --depth 1 origin "$LLAMA_CPP_REF"'
git -C "$SOURCE_DIR" fetch --quiet --depth 1 origin "$LLAMA_CPP_REF"
echo '$ git -C "$SOURCE_DIR" checkout --quiet --detach "$LLAMA_CPP_REF"'
git -C "$SOURCE_DIR" checkout --quiet --detach "$LLAMA_CPP_REF"

cat <<'EOF'
$ cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CCACHE=OFF \
    -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
EOF
if ! cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    >"$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG" >&2
  exit 1
fi
echo '$ cmake --build "$SOURCE_DIR/build" --config Release --target llama-cli \'
echo '    -j "$(nproc)"'
if ! cmake --build "$SOURCE_DIR/build" --config Release \
    --target llama-cli -j "$(nproc)" >>"$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG" >&2
  exit 1
fi

echo '$ install -m 0755 "$SOURCE_DIR/build/bin/llama-cli" \'
echo '    "$INSTALL_BIN_DIR/llama-cli"'
install -m 0755 "$SOURCE_DIR/build/bin/llama-cli" \
  "$INSTALL_BIN_DIR/llama-cli"

echo '$ printf "source commit: "; git -C "$SOURCE_DIR" rev-parse HEAD'
printf "source commit: "
git -C "$SOURCE_DIR" rev-parse HEAD
echo '$ "$INSTALL_BIN_DIR/llama-cli" --version'
"$INSTALL_BIN_DIR/llama-cli" --version 2>&1
echo '$ "$INSTALL_BIN_DIR/llama-cli" --list-devices'
"$INSTALL_BIN_DIR/llama-cli" --list-devices
