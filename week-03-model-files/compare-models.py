#!/usr/bin/env python3
"""Compare a base model repository with its instruction-tuned counterpart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--instruct", default="Qwen/Qwen2.5-3B-Instruct")
    return parser.parse_args()


def load_json(repo_id: str, filename: str, revision: str) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    return json.loads(Path(path).read_text())


def summarize(repo_id: str) -> dict[str, Any]:
    info = HfApi().model_info(repo_id, files_metadata=True)
    config = load_json(repo_id, "config.json", info.sha)
    tokenizer = load_json(repo_id, "tokenizer_config.json", info.sha)
    safe_info = getattr(info, "safetensors", None)
    weights = [
        item
        for item in info.siblings
        if item.rfilename.endswith(".safetensors")
    ]
    card = info.card_data.to_dict() if info.card_data else {}
    return {
        "repository": repo_id,
        "revision": info.sha,
        "base_model": card.get("base_model"),
        "tags": card.get("tags"),
        "architecture": (config.get("architectures") or [None])[0],
        "layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "attention_heads": config.get("num_attention_heads"),
        "kv_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "context": config.get("max_position_embeddings"),
        "dtype": config.get("torch_dtype"),
        "parameters": getattr(safe_info, "total", None),
        "chat_template": tokenizer.get("chat_template") is not None,
        "license": card.get("license"),
        "first_shard_sha256": weights[0].lfs.sha256,
    }


def main() -> None:
    args = parse_args()
    base = summarize(args.base)
    instruct = summarize(args.instruct)
    fields = (
        "repository",
        "revision",
        "base_model",
        "tags",
        "architecture",
        "layers",
        "hidden_size",
        "attention_heads",
        "kv_heads",
        "vocab_size",
        "context",
        "dtype",
        "parameters",
        "chat_template",
        "license",
        "first_shard_sha256",
    )
    for name in fields:
        print(name)
        print(f"  base:     {base[name]}")
        print(f"  instruct: {instruct[name]}")


if __name__ == "__main__":
    main()
