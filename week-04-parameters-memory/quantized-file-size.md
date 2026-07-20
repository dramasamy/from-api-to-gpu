# Why a Q4 model uses more than four bits per parameter

This is optional depth for Week 4. The main lesson needs only the clean weight
formula. This note records why that formula is a lower bound for Ollama's
`llama3.2:3b` package.

All Ollama commands ran on the DGX Spark. The calculator ran on the control Mac.

## Start with the official model fields

Ollama's `/api/show` response is the simplest source for the model's parameter
count and quantization label:

```text
$ curl -s http://localhost:11434/api/show \
    -d '{"model":"llama3.2:3b"}' \
    | jq '{param_count: .model_info["general.parameter_count"],
           quant: .details.quantization_level}'
{
  "param_count": 3212749888,
  "quant": "Q4_K_M"
}
```

Treating all 3,212,749,888 parameters as plain INT4 gives the clean estimate:

```text
$ python3 model_memory.py --count 3212749888 --precision int4
parameter_count    3,212,749,888
precision          int4
bits_per_parameter 4
weight_bytes       1,606,374,944
weight_gb          1.61 GB (decimal, / 1e9)
weight_gib         1.50 GiB (binary, / 1024^3)
```

## Follow the manifest to the real blob

Ollama stores two different files here:

- The manifest at
  `/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/`
  `llama3.2/3b` points to each package layer by digest and records its size.
- The model data lives under `/usr/share/ollama/.ollama/models/blobs/`. Ollama
  changes the manifest digest `sha256:...` into the filename `sha256-...`.

First I select the one manifest layer whose media type ends in `image.model`:

```text
$ MANIFEST=/usr/share/ollama/.ollama/models/manifests/\
registry.ollama.ai/library/llama3.2/3b
$ jq -r 'first(.layers[] \
    | select(.mediaType | endswith("image.model"))) \
    | .mediaType, .digest, (.size | tostring)' "$MANIFEST"
application/vnd.ollama.image.model
sha256:dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
2019377376
```

The next commands replace the digest's colon with a hyphen to derive the blob
filename. They then compare both the byte size and SHA-256 digest with the
manifest record:

```bash
$ DIGEST=$(jq -r 'first(.layers[] \
    | select(.mediaType | endswith("image.model"))) | .digest' "$MANIFEST")
$ RECORDED_SIZE=$(jq -r 'first(.layers[] \
    | select(.mediaType | endswith("image.model"))) | .size' "$MANIFEST")
$ BLOB=/usr/share/ollama/.ollama/models/blobs/${DIGEST/:/-}
$ ACTUAL_SIZE=$(stat -c %s "$BLOB")
$ ACTUAL_DIGEST=sha256:$(sha256sum "$BLOB" | awk '{print $1}')
$ printf 'manifest_digest\n%s\nblob_digest\n%s\n' \
    "$DIGEST" "$ACTUAL_DIGEST"
$ printf 'digest_match %s\nmanifest_size %s\nblob_size %s\nsize_match %s\n' \
    "$([[ "$DIGEST" == "$ACTUAL_DIGEST" ]] && echo true || echo false)" \
    "$RECORDED_SIZE" "$ACTUAL_SIZE" \
    "$([[ "$RECORDED_SIZE" == "$ACTUAL_SIZE" ]] && echo true || echo false)"
```

```text
manifest_digest
sha256:dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
blob_digest
sha256:dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
digest_match true
manifest_size 2019377376
blob_size 2019377376
size_match true
```

Matching the size proves the resolved file has the expected byte count.
Matching SHA-256 proves its content is the exact blob named by the manifest.

## Count the tensor formats

`Q4_K_M` is a mixed format from llama.cpp's GGUF ecosystem, not a promise that
every stored value uses four bits.[^gguf] Ollama's verbose tensor list shows
three formats in this package:

```text
$ ollama show --verbose llama3.2:3b 2>/dev/null \
    | awk '/Tensors/{found=1} found' \
    | grep -oE 'Q4_K|Q6_K|F32' | sort | uniq -c
     58 F32
    168 Q4_K
     29 Q6_K
```

Most tensors use `Q4_K`, while some use `Q6_K` or `F32`. Quantized blocks also
store scales and other data needed to interpret their low-bit values. The whole
blob therefore has this effective on-disk cost:

$$
\frac{2{,}019{,}377{,}376 \times 8}{3{,}212{,}749{,}888}
\approx 5.03 \text{ on-disk bits per parameter}
$$

This does not mean each parameter has 5.03 bits of precision. It means the
complete file averages 5.03 bits per parameter after mixed tensor formats and
their storage data are included. The exact file size is the right number for
disk and download planning. Week 15 returns to what the quantization formats do
to values and model quality.

[^gguf]: GGUF documentation in the llama.cpp project:
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md