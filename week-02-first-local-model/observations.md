# Observations

- Phi-4 generated about 23.4 to 23.5 tokens per second in all six runs.
- Cold TTFT was 6.03 seconds at 2,048 tokens and 6.26 seconds at 4,096 tokens.
- Warm TTFT was between 151.9 and 196.1 milliseconds.
- This short prompt did not show a meaningful speed difference between the two
  context settings. The setting controls available capacity; it does not fill
  that capacity by itself.
- The two warm runs per context are enough to learn the benchmark mechanics,
  but not enough for a production capacity claim.
