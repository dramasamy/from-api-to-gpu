#!/usr/bin/env python3
"""Small command-line chat client for Ollama's streaming /api/chat endpoint."""

import argparse
import json
import os
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL", "phi4"))
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--system", default="You are a concise assistant.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--once", help="Send one prompt and exit")
    return parser.parse_args()


def request_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, float | int],
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    parts: list[str] = []
    try:
        with urllib.request.urlopen(request) as response:
            for line in response:
                event = json.loads(line)
                text = event.get("message", {}).get("content", "")
                if text:
                    print(text, end="", flush=True)
                    parts.append(text)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama request failed: {exc}") from exc
    print()
    return "".join(parts)


def main() -> None:
    args = parse_args()
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
    }

    if args.once:
        messages.append({"role": "user", "content": args.once})
        request_chat(args.url, args.model, messages, options)
        return

    print(f"Chatting with {args.model}. Type /exit to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        print("model> ", end="", flush=True)
        answer = request_chat(args.url, args.model, messages, options)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
