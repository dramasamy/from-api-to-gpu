# Troubleshooting

## Avoid downloading the model weights

This week inspects repository metadata. `model-inspector.py` downloads only small
configuration and license files through `hf_hub_download`. It uses the Hub API
to read weight-file names and sizes without downloading the Safetensors shards.

## Hub rate limits

Public repositories do not require a token for this lab. If the Hub returns a
rate-limit error, set a read-only `HF_TOKEN` in the shell and rerun the command.
Never commit the token.
