#!/usr/bin/env python3
"""Read selected tokenizer metadata directly from an Ollama GGUF model layer."""

import argparse
import json
import os
from pathlib import Path

from gguf import GGUFReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL", "phi4"))
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/usr/share/ollama/.ollama/models"),
    )
    return parser.parse_args()


def field(reader: GGUFReader, name: str):
    value = reader.get_field(name)
    if value is None:
        raise KeyError(f"Missing GGUF field: {name}")
    return value.contents()


def main() -> None:
    args = parse_args()
    name, separator, tag = args.model.partition(":")
    if not separator:
        tag = "latest"
    manifest = (
        args.models_dir
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / name
        / tag
    )
    data = json.loads(manifest.read_text())
    digest = next(
        layer["digest"]
        for layer in data["layers"]
        if layer["mediaType"].endswith("image.model")
    )
    blob = args.models_dir / "blobs" / digest.replace(":", "-")
    reader = GGUFReader(blob)
    tokens = field(reader, "tokenizer.ggml.tokens")
    template = field(reader, "tokenizer.chat_template")

    print("tokenizer model:", field(reader, "tokenizer.ggml.model"))
    print("tokenizer pre:", field(reader, "tokenizer.ggml.pre"))
    print("token count:", len(tokens))
    for token_id in (100257, 100264, 100265, 100266):
        print(f"token {token_id}: {tokens[token_id]!r}")
    for marker in ("<|im_start|>", "<|im_end|>", "<|im_sep|>"):
        print(f"template contains {marker}: {marker in template}")


if __name__ == "__main__":
    main()
