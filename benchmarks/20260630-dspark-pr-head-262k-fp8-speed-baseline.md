# DSpark PR-Head 262K/fp8 Speed Baseline - 2026-06-30

This checkpoint is a diagnostic baseline, not the default production profile
for this repo. The default recipe remains the 1M `nvfp4_ds_mla` KV path.

The purpose of this run was to reproduce the older "about 60 tok/s" DSpark
number and prove whether the runtime/model path could still hit it outside an
agent harness.

## Finding

The old speed number belongs to a smaller fp8-KV, single-sequence profile:

- image: `vllm-dspark-runtime:rafael-pr1-3519c3b`
- vLLM PR source: `rafaelcaricio/vllm#1`, commit `3519c3b88`
- base image: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`
- topology: 2x DGX Spark, TP=2, `mp`
- `max_model_len=262144`
- `max_num_seqs=1`
- `max_num_batched_tokens=8192`
- `kv_cache_dtype=fp8`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`

Boot evidence:

```text
GPU KV cache size: 940,580 tokens
Maximum concurrency for 262,144 tokens per request: 3.59x
Application startup complete.
```

Benchmark:

- scenario: `code_completion`
- prompt target: 512 tokens
- generated: 256 tokens
- temperature: 0.0
- ignore EOS: true
- server decode throughput: `63.97 tok/s`
- DSpark acceptance: `67.9%`
- accepted tokens per draft: `3.40`

## Important image gotcha

Do not trust a local tag named `vllm-dspark-runtime:clean` unless it was built
from the current Rafael PR overlay. On the test nodes, that tag had been reused
for a partial overlay and caused false failures:

- a B12X WO projection shape mismatch
- an older `rafael-pr1-clean` image then failed with
  `All drafting layers should belong to the same kv cache group`

Rebuilding the lightweight overlay from `rafaelcaricio/vllm#1` commit
`3519c3b88` fixed those boot failures and reproduced the expected speed.

## Read

This proves the DSpark model/runtime can still hit the historical single-stream
speed target when the exact fp8/262K/seq1 PR-head runtime is used. It does not
replace the 1M NVFP4 profile. Use it to diagnose runtime/image drift, not as the
default long-context deployment.
