# Troubleshooting

## Ollama CLI prints no captured response over SSH

`ollama run` uses terminal-aware output. A one-shot call returned no visible text
when SSH did not allocate a terminal. Force a pseudo-terminal with `ssh -tt`:

```bash
ssh -tt spark \
  'ollama run --nowordwrap phi4 "Reply with exactly: local model ready"'
```

## DGX Spark memory shows as not supported

`nvidia-smi` does not report a normal discrete VRAM total on this DGX Spark.
That is expected because the GB10 uses unified memory. Use `free -h` to inspect
the shared system pool, and use `ollama ps` to see the model allocation.
