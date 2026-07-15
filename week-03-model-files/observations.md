# Observations

- The instruct repository points back to `Qwen/Qwen2.5-3B` as its base model.
- Both repositories report the same architecture and dimensions: 36 layers,
  hidden size 2,048, 16 attention heads, and 2 key-value heads.
- Their first Safetensors shard hashes differ. They are separate checkpoints,
  even though their architecture is the same.
- Both tokenizer configurations contain a chat template. A chat template alone
  does not prove that a checkpoint was instruction-tuned.
- The instruct checkpoint has 3,085,938,688 BF16 parameters in two shards with a
  combined weight size of 6,171,877,376 bytes.
- The repository uses the Qwen Research License. It permits non-commercial use
  and requires a separate license for commercial use.
- The Hub metadata was inspected without downloading either weight shard.
