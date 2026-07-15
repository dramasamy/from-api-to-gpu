#!/usr/bin/env python3
"""Inspect a Hugging Face model repository without downloading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(repo_id: str, filename: str, revision: str) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    return json.loads(Path(path).read_text())


def human_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")


def card_value(card_data: Any, name: str) -> Any:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(name)
    return getattr(card_data, name, None)


def first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def main() -> None:
    args = parse_args()
    api = HfApi()
    info = api.model_info(
        args.model,
        revision=args.revision,
        files_metadata=True,
    )
    resolved_revision = info.sha
    files = sorted(info.siblings, key=lambda item: item.rfilename)
    names = {item.rfilename for item in files}

    config = load_json(args.model, "config.json", resolved_revision)
    tokenizer = load_json(
        args.model,
        "tokenizer_config.json",
        resolved_revision,
    )
    license_path = hf_hub_download(
        repo_id=args.model,
        filename="LICENSE",
        revision=resolved_revision,
    )
    license_text = Path(license_path).read_text()
    generation = (
        load_json(args.model, "generation_config.json", resolved_revision)
        if "generation_config.json" in names
        else {}
    )
    index = (
        load_json(
            args.model,
            "model.safetensors.index.json",
            resolved_revision,
        )
        if "model.safetensors.index.json" in names
        else {}
    )

    safe_info = getattr(info, "safetensors", None)
    total_parameters = getattr(safe_info, "total", None)
    parameter_groups = getattr(safe_info, "parameters", None) or {}
    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    chat_template = tokenizer.get("chat_template")
    added_tokens = tokenizer.get("added_tokens_decoder", {})
    configured_ids = {
        config.get("bos_token_id"),
        config.get("eos_token_id"),
        generation.get("bos_token_id"),
        generation.get("pad_token_id"),
    }
    generation_eos = generation.get("eos_token_id")
    if isinstance(generation_eos, list):
        configured_ids.update(generation_eos)
    else:
        configured_ids.add(generation_eos)
    for token_id, token_data in added_tokens.items():
        if token_data.get("content") == "<|im_start|>":
            configured_ids.add(int(token_id))
    configured_ids.discard(None)
    special_token_map = {
        str(token_id): added_tokens.get(str(token_id), {}).get("content")
        for token_id in sorted(configured_ids)
    }

    result = {
        "repository": args.model,
        "revision": resolved_revision,
        "base_model": card_value(info.card_data, "base_model"),
        "tags": card_value(info.card_data, "tags"),
        "license_metadata": card_value(info.card_data, "license"),
        "license_title": next(
            line for line in license_text.splitlines() if line.strip()
        ),
        "noncommercial_only": (
            "FOR NON-COMMERCIAL PURPOSES ONLY" in license_text
        ),
        "pipeline_tag": info.pipeline_tag,
        "architecture": first_value(*(config.get("architectures") or [])),
        "model_type": config.get("model_type"),
        "total_parameters": total_parameters,
        "parameter_groups": parameter_groups,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "bos_token_id": first_value(
            generation.get("bos_token_id"),
            config.get("bos_token_id"),
        ),
        "eos_token_id": first_value(
            generation.get("eos_token_id"),
            config.get("eos_token_id"),
        ),
        "pad_token_id": first_value(
            generation.get("pad_token_id"),
            config.get("pad_token_id"),
        ),
        "bos_token": tokenizer.get("bos_token"),
        "eos_token": tokenizer.get("eos_token"),
        "pad_token": tokenizer.get("pad_token"),
        "tokenizer_class": tokenizer.get("tokenizer_class"),
        "tokenizer_model_max_length": tokenizer.get("model_max_length"),
        "special_token_map": special_token_map,
        "chat_template_present": chat_template is not None,
        "chat_template_characters": (
            len(chat_template) if isinstance(chat_template, str) else None
        ),
        "safetensors_shards": len(shards) or None,
        "safetensors_tensors": len(weight_map) or None,
        "safetensors_total_bytes": index.get("metadata", {}).get("total_size"),
        "generation_defaults": {
            name: generation.get(name)
            for name in (
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
            )
        },
        "files": [
            {"name": item.rfilename, "size_bytes": item.size}
            for item in files
        ],
    }

    print(f"Repository: {result['repository']}")
    print(f"Revision:   {result['revision']}")
    print(f"Base model: {result['base_model']}")
    print(f"Tags:       {result['tags']}")
    print(f"License:    {result['license_metadata']}")
    print(f"Title:      {result['license_title']}")
    print(f"Non-commercial only: {result['noncommercial_only']}")
    print(f"Task:       {result['pipeline_tag']}")
    print("\nRepository files:")
    for item in result["files"]:
        print(f"  {human_bytes(item['size_bytes']):>10}  {item['name']}")

    print("\nModel configuration:")
    fields = (
        "architecture",
        "model_type",
        "total_parameters",
        "parameter_groups",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
        "tie_word_embeddings",
    )
    for name in fields:
        print(f"  {name:27} {result[name]}")

    print("\nGeneration and tokenizer:")
    for name in (
        "tokenizer_class",
        "tokenizer_model_max_length",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "bos_token",
        "eos_token",
        "pad_token",
        "chat_template_present",
        "chat_template_characters",
    ):
        print(f"  {name:27} {result[name]}")
    for token_id, content in result["special_token_map"].items():
        print(f"  special token {token_id:>9} {content}")

    print("\nGeneration defaults:")
    for name, value in result["generation_defaults"].items():
        print(f"  {name:27} {value}")

    print("\nSafetensors:")
    for name in (
        "safetensors_shards",
        "safetensors_tensors",
        "safetensors_total_bytes",
    ):
        print(f"  {name:27} {result[name]}")
    for shard in shards:
        print(f"  shard                       {shard}")

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
