#!/usr/bin/env python3
"""Week 2 - show how text becomes tokens for a cl100k model like phi4.

Encodes the prompt and decodes individual token IDs, so you can match the
`context` array from the generate API back to real words.

Run with a Python that has tiktoken installed, e.g. the Week 1 venv:
    ~/venvs/w1/bin/python -m pip install -q tiktoken
    ~/venvs/w1/bin/python decode_tokens.py
"""
import tiktoken

PROMPT = "Explain Kubernetes scheduling in three sentences."
# A few IDs taken from the generate API's context array.
SAMPLE_IDS = [849, 21435, 67474, 38952, 304, 2380, 23719, 13, 882, 78191]


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")

    print("encode:", enc.encode(PROMPT))
    print("decode individual IDs:")
    for t in SAMPLE_IDS:
        print(f"  {t:>6}  {enc.decode([t])!r}")


if __name__ == "__main__":
    main()
