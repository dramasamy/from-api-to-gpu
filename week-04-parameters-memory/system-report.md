# System report

Captured on 2026-07-15. Runtime memory and blob identity were rerun on
2026-07-19 with the same measured sizes.

```text
$ ollama --version
ollama version is 0.31.2

$ uname -m
aarch64

$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVIDIA GB10, 580.159.03

$ ollama list | awk 'NR==1 || $1=="llama3.2:3b"'
NAME                    ID              SIZE      MODIFIED
llama3.2:3b             a80c4f17acd5    2.0 GB    2 days ago

$ curl -s http://localhost:11434/api/show -d '{"model":"llama3.2:3b"}' \
    | jq '{architecture: .model_info["general.architecture"],
           param_count: .model_info["general.parameter_count"],
           ctx: .model_info["llama.context_length"],
           quant: .details.quantization_level,
           license_head: (.license // "" | split("\n")[0])}'
{
  "architecture": "llama",
  "param_count": 3212749888,
  "ctx": 131072,
  "quant": "Q4_K_M",
  "license_head": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT"
}

$ ollama show --verbose llama3.2:3b 2>/dev/null \
    | awk '/Tensors/{f=1} f' | grep -oE 'Q4_K|Q6_K|F32' | sort | uniq -c
  58 F32
 168 Q4_K
  29 Q6_K
```

## Runtime measurement

Rerun as direct commands on 2026-07-19:

```text
$ ssh spark
$ ollama list | awk 'NR == 1 || $1 == "llama3.2:3b"'
NAME                       ID              SIZE      MODIFIED
llama3.2:3b                a80c4f17acd5    2.0 GB    3 days ago
$ ollama stop llama3.2:3b >/dev/null 2>&1 || true
$ curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' \
    http://localhost:11434/api/generate \
    -d '{"model":"llama3.2:3b","prompt":"hi","stream":false,
    "options":{"num_ctx":4096,"num_predict":1}}'
HTTP 200
$ ollama ps
NAME           ID              SIZE      PROCESSOR    CONTEXT    UNTIL
llama3.2:3b    a80c4f17acd5    2.6 GB    100% GPU     4096       4 minutes from now
$ ollama stop llama3.2:3b >/dev/null 2>&1 || true
$ curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' \
    http://localhost:11434/api/generate \
    -d '{"model":"llama3.2:3b","prompt":"hi","stream":false,
    "options":{"num_ctx":16384,"num_predict":1}}'
HTTP 200
$ ollama ps
NAME           ID              SIZE      PROCESSOR    CONTEXT    UNTIL
llama3.2:3b    a80c4f17acd5    4.1 GB    100% GPU     16384      4 minutes from now
```

The full manifest-to-blob size and SHA-256 comparison is recorded in
[`quantized-file-size.md`](quantized-file-size.md).

The calculator itself is hardware independent and needs only Python. The runtime
sizes were measured on an NVIDIA GB10 with 121 GiB of unified memory.
