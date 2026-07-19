# Reusing an Ollama model in llama.cpp

This is the full investigation behind one question from Week 2:

> Can `llama.cpp` run the same model file that Ollama downloaded?

For the Phi-4 model on this DGX Spark, the answer is yes. I verified more than
the file format. I built `llama.cpp` with CUDA support, passed it the exact model
blob named in Ollama's manifest, offloaded all 41 model layers to the NVIDIA
GB10, and generated a reply.

The result in one place:

| Check | Result |
| ----- | ------ |
| Ollama model digest | `sha256:fd7b...df20` |
| llama.cpp commit | `571d0d54` |
| Device | NVIDIA GB10 through CUDA |
| GPU offload | 41 of 41 model layers |
| Generated reply | `Shared model works.` |

The rest of this file records how I reached that result, including the wrong
first assumption and the checks that corrected it.

## 1. Find Ollama's actual model file

Ollama does not store this model under a friendly filename such as
`phi4-q4_k_m.gguf`. It stores a manifest plus content-addressed blobs. I used
[`inspect_model.sh`](inspect_model.sh) to read the manifest, select the layer
whose media type is `application/vnd.ollama.image.model`, and resolve its blob
path.

Run the inspection from the private parent repository:

```bash
ssh spark 'bash -s' \
  < public/week-02-first-local-model/inspect_model.sh
```

The current inspection selects the model layer and verifies the real file:

```text
$ ssh spark 'bash -s' < public/week-02-first-local-model/inspect_model.sh
=== connect the manifest to the model blob ===
$ MODEL_LAYER="$(jq -c '[.layers[]
    | select(.mediaType == "application/vnd.ollama.image.model")
  ][0]' "$MANIFEST")"
$ echo "$MODEL_LAYER" | jq .
{
  "mediaType": "application/vnd.ollama.image.model",
  "digest": "sha256:fd7b6731c33c57f61767612f56517460ec2d1e2e5a3f0163e0eb3d8d8cb5df20",
  "size": 9053114464
}
$ MODEL_DIGEST="$(jq -r .digest <<<"$MODEL_LAYER")"
$ MODEL_SIZE="$(jq -r .size <<<"$MODEL_LAYER")"
$ BLOB_FILENAME="${MODEL_DIGEST/:/-}"
$ MODEL_BLOB="$MODELS_DIR/blobs/$BLOB_FILENAME"
blob filename: sha256-fd7b6731c33c57f61767612f56517460ec2d1e2e5a3f0163e0eb3d8d8cb5df20
$ ACTUAL_SIZE="$(stat -c %s "$MODEL_BLOB")"
manifest bytes: 9053114464
file bytes: 9053114464
$ test "$MODEL_SIZE" = "$ACTUAL_SIZE" && echo "size check: MATCH"
size check: MATCH
$ EXPECTED_HASH="${MODEL_DIGEST#sha256:}"
$ ACTUAL_HASH="$(sha256sum "$MODEL_BLOB" | awk '{print $1}')"
$ test "$EXPECTED_HASH" = "$ACTUAL_HASH" && echo "SHA-256 check: MATCH"
SHA-256 check: MATCH
$ xxd -l 4 -g 1 "$MODEL_BLOB"
00000000: 47 47 55 46                                      GGUF
```

The byte count and calculated SHA-256 match the selected manifest layer. The
`xxd` command reads four bytes from offset zero. It displays their hex values
(`47 47 55 46`) and their ASCII text (`GGUF`). Those opening bytes are the file
signature, also called the magic header. The missing `.gguf` extension does not
matter because runtimes inspect the file contents. This proves I found the exact
GGUF model file, but it does not yet prove that this `llama.cpp` build can load
Phi-4 and generate text.

## 2. Check the build environment

I first checked the architecture, compiler tools, CUDA compiler, GPU compute
capability, disk space, and any existing installation:

```bash
ssh spark 'set -eu
printf "architecture: "; uname -m
printf "os: "; . /etc/os-release; echo "$PRETTY_NAME"
printf "cmake: "; cmake --version | head -n 1
printf "compiler: "; c++ --version | head -n 1
printf "git: "; git --version
printf "nvcc: "; nvcc --version | tail -n 1
printf "cuda arch: "; \
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader
printf "home free: "; df -h "$HOME" | awk '\''NR == 2 {print $4}'\''
printf "existing source: "; \
  test -d "$HOME/src/llama.cpp/.git" && echo yes || echo no
printf "existing local cli: "; \
  test -x "$HOME/.local/bin/llama-cli" \
    && "$HOME/.local/bin/llama-cli" --version || echo no'
```

The command stopped at `nvcc` because `set -e` treats a missing command as an
error:

```text
architecture: aarch64
os: Ubuntu 24.04.4 LTS
cmake: cmake version 3.28.3
compiler: c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
git: git version 2.43.0
nvcc: bash: line 7: nvcc: command not found
```

My first reading was that the CUDA toolkit might not be installed. That was too
quick. This only proved that `nvcc` was not in the non-login SSH shell's
`PATH`. I searched the normal CUDA installation directory and queried the
installed packages:

```bash
ssh spark 'set -u
printf "%s\n" "=== CUDA paths ==="
find /usr/local -maxdepth 3 -type f -name nvcc -print 2>/dev/null
printf "%s\n" "=== CUDA packages ==="
dpkg-query -W -f='\''${Package}\t${Version}\n'\'' \
  '\''cuda-*'\'' '\''nvidia-cuda-*'\'' 2>/dev/null \
  | head -n 30 || true
printf "%s\n" "=== Driver ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap \
  --format=csv,noheader'
```

The toolkit was installed under `/usr/local/cuda-13.0`:

```text
=== CUDA paths ===
/usr/local/cuda-13.0/bin/nvcc
=== CUDA packages ===
cuda-cccl-13-0  13.0.85-1
cuda-command-line-tools-13-0    13.0.3-1
cuda-compiler-13-0      13.0.3-1
cuda-compute-repo
cuda-compute-repo-lowpri        25.10-2
cuda-crt-13-0   13.0.88-1
cuda-cub-13-0
cuda-cudart-11-2
cuda-cudart-13-0        13.0.96-1
cuda-cudart-dev-13-0    13.0.96-1
cuda-culibos-dev-13-0   13.0.85-1
cuda-cuobjdump-13-0     13.0.85-1
cuda-cupti-13-0 13.0.85-1
cuda-cupti-dev-13-0     13.0.85-1
cuda-documentation-13-0 13.0.85-1
cuda-driver-dev-13-0    13.0.96-1
cuda-gdb-13-0   13.0.85-1
cuda-libraries-13-0     13.0.3-1
cuda-libraries-dev-13-0 13.0.3-1
cuda-nsight-compute-13-0        13.0.3-1
cuda-nsight-systems-13-0        13.0.3-1
cuda-nvcc-13
cuda-nvcc-13-0  13.0.88-1
cuda-nvdisasm-13-0      13.0.85-1
cuda-nvml-dev-13-0      13.0.87-1
cuda-nvprof-13-0
cuda-nvprune-13-0       13.0.85-1
cuda-nvrtc-13-0 13.0.88-1
cuda-nvrtc-dev-13-0     13.0.88-1
=== Driver ===
NVIDIA GB10, 580.159.03, 12.1
```

I then captured the exact build values with a smaller command:

```bash
ssh spark 'printf "architecture: "; uname -m
printf "CUDA compiler: "; \
  /usr/local/cuda-13.0/bin/nvcc --version | tail -n 1
printf "GPU and compute capability: "; \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'
```

```text
architecture: aarch64
CUDA compiler: Build cuda_13.0.r13.0/compiler.36424714_0
GPU and compute capability: NVIDIA GB10, 12.1
```

The Spark is ARM64, and the GB10 reports compute capability 12.1. CMake writes
that architecture as `121`.

## 3. Pin and build llama.cpp with CUDA

I resolved the current `llama.cpp` commit before building it:

```bash
git ls-remote https://github.com/ggml-org/llama.cpp.git refs/heads/master
```

```text
571d0d540df04f25298d0e159e520d9fc62ed121        refs/heads/master
```

The reusable build is in [`setup_llamacpp.sh`](setup_llamacpp.sh). It:

- checks the full path to `nvcc` instead of relying on `PATH`;
- checks out commit `571d0d540df04f25298d0e159e520d9fc62ed121`;
- enables the CUDA backend and targets architecture `121`;
- builds only `llama-cli`;
- installs it to `~/.local/bin`, so root access is not needed;
- keeps normal compiler progress in a temporary file and prints it if the build
  fails.

Run it from the parent repository:

```bash
bash -n public/week-02-first-local-model/setup_llamacpp.sh
ssh spark 'bash -s' \
  < public/week-02-first-local-model/setup_llamacpp.sh
```

The first build printed hundreds of repetitive per-file compiler progress
lines. Those lines were not kept as a lab result because they do not change the
proof. The CMake configuration did confirm these important values:

```text
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- GGML_SYSTEM_ARCH: ARM
-- Found CUDAToolkit: /usr/local/cuda/targets/sbsa-linux/include (found version "13.0.88")
-- The CUDA compiler identification is NVIDIA 13.0.88
-- Using CMAKE_CUDA_ARCHITECTURES=121a CMAKE_CUDA_ARCHITECTURES_NATIVE=121a-real
-- CUDA host compiler is GNU 13.3.0
-- Including CUDA backend
-- ggml commit:  571d0d54
```

The current rerun shows each meaningful build command before its result:

```text
$ ssh spark 'bash -s' \
>   < public/week-02-first-local-model/setup_llamacpp.sh
$ git -C "$SOURCE_DIR" fetch --quiet --depth 1 origin "$LLAMA_CPP_REF"
$ git -C "$SOURCE_DIR" checkout --quiet --detach "$LLAMA_CPP_REF"
$ cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CCACHE=OFF \
    -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
$ cmake --build "$SOURCE_DIR/build" --config Release --target llama-cli \
    -j "$(nproc)"
$ install -m 0755 "$SOURCE_DIR/build/bin/llama-cli" \
    "$INSTALL_BIN_DIR/llama-cli"
$ printf "source commit: "; git -C "$SOURCE_DIR" rev-parse HEAD
source commit: 571d0d540df04f25298d0e159e520d9fc62ed121
$ "$INSTALL_BIN_DIR/llama-cli" --version
version: 1 (571d0d54)
built with GNU 13.3.0 for Linux aarch64
$ "$INSTALL_BIN_DIR/llama-cli" --list-devices
Available devices:
  CUDA0: NVIDIA GB10 (124609 MiB, 121444 MiB free)
```

The free-memory number can change between runs. The important facts are the
source commit, Linux ARM64 build, and detected `CUDA0` device.

## 4. Pass the exact Ollama blob to llama.cpp

The test is in
[`run_ollama_model_llamacpp.sh`](run_ollama_model_llamacpp.sh). It reads the
Ollama manifest itself, so it does not contain a copied digest or hard-coded
blob filename. It then passes the resolved path to `llama-cli --model`.

The run used:

- context size 2,048;
- all available model layers on CUDA with `--n-gpu-layers 99`;
- seed 42 and temperature 0;
- a limit of 32 generated tokens;
- the embedded GGUF chat template through conversation mode.

Run it from the parent repository:

```bash
bash -n public/week-02-first-local-model/run_ollama_model_llamacpp.sh
ssh spark 'bash -s' \
  < public/week-02-first-local-model/run_ollama_model_llamacpp.sh
```

The complete reader-facing output was:

```text
$ ssh spark 'bash -s' \
>   < public/week-02-first-local-model/run_ollama_model_llamacpp.sh
$ MANIFEST="$MODELS_DIR/manifests/registry.ollama.ai/library/$NAME/$TAG"
$ MODEL_DIGEST="$(jq -r '[.layers[]
    | select(.mediaType == "application/vnd.ollama.image.model")
  ][0].digest' "$MANIFEST")"
$ MODEL_BLOB="$MODELS_DIR/blobs/${MODEL_DIGEST/:/-}"
$ echo "model: $MODEL"
model: phi4
$ echo "model digest: $MODEL_DIGEST"
model digest: sha256:fd7b6731c33c57f61767612f56517460ec2d1e2e5a3f0163e0eb3d8d8cb5df20
$ LLAMA_VERSION="$($LLAMA_CLI --version 2>&1 | head -n 1)"
$ echo "llama.cpp: ${LLAMA_VERSION#version: }"
llama.cpp: 1 (571d0d54)
$ "$LLAMA_CLI" --list-devices
Available devices:
  CUDA0: NVIDIA GB10 (124609 MiB, 121388 MiB free)

=== generation from Ollama model blob ===
$ "$LLAMA_CLI" \
    --model "$MODEL_BLOB" \
    --prompt "$PROMPT" \
    --conversation --single-turn --simple-io --no-display-prompt \
    --ctx-size 2048 --n-gpu-layers 99 \
    --seed 42 --temperature 0 --n-predict 32 --no-warmup \
    --verbose --log-colors off --no-log-prefix --no-log-timestamps \
    >"$LOG_FILE" 2>&1
$ grep 'using device CUDA0' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*using device ([^ ]+) \(([^)]+)\).*/device: \1 (\2)/'
device: CUDA0 (NVIDIA GB10)
$ grep 'offloaded .* layers to GPU' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*offloaded ([0-9]+\/[0-9]+) layers.*/GPU layers: \1/'
GPU layers: 41/41
$ grep 'CPU_Mapped model buffer size' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*size = *([^ ]+ MiB).*/CPU mapped model buffer: \1/'
CPU mapped model buffer: 275.62 MiB
$ grep 'CUDA0 model buffer size' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*size = *([^ ]+ MiB).*/CUDA0 model buffer: \1/'
CUDA0 model buffer: 8354.71 MiB
$ grep 'Parsed message: {"role":"assistant"' "$LOG_FILE" | tail -n 1 \
    | sed -E 's/.*"content":"(.*)"}/assistant: \1/'
assistant: Shared model works.
$ grep '^\[ Prompt:' "$LOG_FILE" | tail -n 1 \
  | sed -E -e 's/^\[ Prompt: /speed: prompt /' \
        -e 's/ \| Generation: /, generation /' \
        -e 's/\]$//' -e 's/[[:space:]]*$//'
speed: prompt 242.4 t/s, generation 28.1 t/s
```

The digest matches Ollama's manifest. `llama.cpp` therefore opened the exact
9,053,114,464-byte model layer Ollama had downloaded. It put all 41 model layers
on the GB10 and generated the requested reply.

The 28.1 tokens/sec value came from a three-word reply. It proves generation
worked, but it is not a fair performance comparison with the longer Ollama
benchmark in the Week 2 post.

## 5. What carries over, and what does not

The **model blob** carries over because it is a GGUF file supported by this
`llama.cpp` commit. The GGUF itself contains the model weights, tokenizer
metadata, and an embedded chat template.

The whole **Ollama package** does not carry over. Ollama's manifest also points
to separate template, license, and default-parameter layers. `llama.cpp` does
not read those Ollama layers when given only the model blob. I set context size,
GPU layers, seed, temperature, and output limit explicitly in the test.

This means:

- the same weights can run in both runtimes;
- the model does not need to be downloaded twice;
- runtime defaults and package-level settings may differ;
- the same weights do not promise identical output across runtimes;
- another Ollama model still needs a real test because an older `llama.cpp`
  build may not support a newer model architecture.

## 6. One operational detail that looked like a hang

The first CUDA build emitted hundreds of compiler progress lines and took long
enough to look stuck in the terminal. After it completed, I checked for leftover
build, inference, and GPU processes:

```bash
ssh spark 'printf "%s\n" "=== matching processes ==="
ps -eo pid,ppid,stat,etime,cmd \
  | grep -E "[l]lama-cli|[c]make --build|[s]etup_llamacpp|[r]un_ollama_model" \
  || true
printf "%s\n" "=== GPU compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader 2>/dev/null || true'
```

```text
=== matching processes ===
=== GPU compute processes ===
```

Nothing was left running. The final build script keeps routine compiler output
in a temporary log, which makes a successful rerun much quieter while still
printing the complete log when configuration or compilation fails.

## Final answer

Yes, this Ollama-downloaded Phi-4 model can run in `llama.cpp` without another
model download. That conclusion is limited to the exact digest and pinned
runtime tested here. The reliable check for another model is the same sequence:
resolve the model layer from Ollama's manifest, confirm the GGUF signature, and
run the exact blob through a `llama.cpp` build that supports its architecture.