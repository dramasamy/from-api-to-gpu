# Troubleshooting

## Avoid downloading the model weights

This week inspects repository metadata. `model-inspector.py` downloads only small
JSON files through `hf_hub_download`. It uses the Hub API to read weight-file
names and sizes without downloading the Safetensors shards.

## ARM64 dependency note

`huggingface-hub` is pure Python. Its optional `hf-xet` dependency installed an
ARM64 wheel successfully on the DGX Spark. No source build was needed.

## Hub rate limits

Public repositories do not require a token for this lab. If the Hub returns a
rate-limit error, set a read-only `HF_TOKEN` in the shell and rerun the command.
Never commit the token.
