# Week 15 — Quantization

**Phase 4: Quantization and model formats**

---

## Learn

- Quantization
- Dequantization
- Calibration
- Post-training quantization
- Quantization-aware training
- Weight-only quantization
- Weight-and-activation quantization
- Per-tensor
- Per-channel
- Group size
- Quantization error
- Accuracy / performance trade-off

Quantization lowers model memory and compute by representing values with fewer bits. Hugging Face documents multiple approaches, including bitsandbytes 8-bit and 4-bit loading, AWQ, and GPTQ.

## Hands-on

Run the same model in available formats/precisions and compare:

| Version | Load memory | TTFT | Tokens/sec | Quality |
| ------- | ----------: | ---: | ---------: | ------- |
| BF16    |             |      |            |         |
| INT8    |             |      |            |         |
| INT4    |             |      |            |         |

## Reference

- Transformers quantization — https://huggingface.co/docs/transformers/main_classes/quantization
