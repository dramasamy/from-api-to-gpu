# Troubleshooting

## Avoid downloading the model weights

This week inspects repository metadata. `model-inspector.py` downloads only small
configuration and license files through `hf_hub_download`. It uses the Hub API
to read weight-file names and sizes without downloading the Safetensors shards.

## Hub rate limits

Public repositories do not require a token for this lab. If the Hub returns a
rate-limit error, set a read-only `HF_TOKEN` in the shell and rerun the command.
Never commit the token. Unauthenticated requests also print a warning like
`You are sending unauthenticated requests to the HF Hub`; it is harmless for
public metadata.

## The license metadata says `other`

The Hub's `cardData.license` field for this model is `other`, which only means
the repository does not use a standard SPDX identifier. It is not a permissive
license. The real terms live in the `LICENSE` file: the Qwen Research License
Agreement, which is non-commercial. Always read the `LICENSE` file rather than
trusting the metadata label. (In the browser, the header badge shows the
friendlier `qwen-research`, which is the same non-standard license.)

## Reading the model-card object

`dict(info.card_data)` raises a `KeyError` with the current Hub client. Use the
supported `info.card_data.to_dict()` method instead, which returns `base_model`,
`tags`, and the license metadata correctly.
