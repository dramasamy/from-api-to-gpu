#!/usr/bin/env python3
"""Measure client TTFT and server timings from Ollama's streaming API."""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phi4")
    parser.add_argument(
        "--prompt",
        default="Explain Kubernetes scheduling in three sentences.",
    )
    parser.add_argument("--contexts", default="2048,4096")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def unload(model: str) -> None:
    subprocess.run(
        ["ollama", "stop", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(1)


def run_once(
    url: str,
    model: str,
    prompt: str,
    num_ctx: int,
    cold: bool,
) -> dict[str, Any]:
    if cold:
        unload(model)
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": 96,
        },
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request) as response:
        for line in response:
            event = json.loads(line)
            if event.get("response") and first_token_at is None:
                first_token_at = time.perf_counter()
            if event.get("done"):
                final = event
    finished = time.perf_counter()

    if first_token_at is None or not final:
        raise RuntimeError("Ollama stream ended without timing data")
    eval_seconds = final["eval_duration"] / 1e9
    return {
        "context": num_ctx,
        "cold": cold,
        "ttft_ms": round((first_token_at - started) * 1000, 1),
        "client_total_ms": round((finished - started) * 1000, 1),
        "load_ms": round(final["load_duration"] / 1e6, 1),
        "prompt_eval_ms": round(final["prompt_eval_duration"] / 1e6, 1),
        "prompt_tokens": final["prompt_eval_count"],
        "output_tokens": final["eval_count"],
        "tokens_per_second": round(final["eval_count"] / eval_seconds, 1),
    }


def main() -> None:
    args = parse_args()
    contexts = [int(value) for value in args.contexts.split(",")]
    rows: list[dict[str, Any]] = []
    print("ctx  run  cold  ttft_ms  total_ms  load_ms  out_tok  tok/s")
    for context in contexts:
        for run_number in range(1, args.runs + 1):
            row = run_once(
                args.url,
                args.model,
                args.prompt,
                context,
                cold=run_number == 1,
            )
            rows.append(row)
            print(
                f"{context:>4}  {run_number:>3}  {str(row['cold']):>5}  "
                f"{row['ttft_ms']:>7.1f}  {row['client_total_ms']:>8.1f}  "
                f"{row['load_ms']:>7.1f}  {row['output_tokens']:>7}  "
                f"{row['tokens_per_second']:>5.1f}"
            )

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "runs_per_context": args.runs,
        "measurements": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
