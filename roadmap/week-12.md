# Week 12 — Generation and sampling

**Phase 3: Understand transformers and model terminology**

---

## Learn

- Logits
- Probability distribution
- Greedy decoding
- Temperature
- Top-k
- Top-p
- Repetition penalty
- Stop sequence
- Seed
- Determinism
- Structured output
- Hallucination

## Hands-on

Run the same prompt under 20 parameter combinations and save the outputs.

Start from the `generation_config.json` defaults recorded in Week 3. Hold the
prompt and seed fixed, then change temperature, top-k, top-p, and repetition
penalty one at a time before combining them. Explain which tokens each filter
allows into the candidate set and why changing several controls at once makes
the result hard to interpret.

## Deliverable

A generation comparison report.
