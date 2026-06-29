# DSpark 1M + max_num_seqs=6 Concurrency Checkpoint - 2026-06-29

The full-context concurrency profile: keep the 1M ceiling, run 6 concurrent
slots. Validated live on 2x DGX Spark TP=2.

## Config

- Image: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency` (Keys concurrency patch)
- `kv_cache_dtype=nvfp4_ds_mla`, `max_model_len=1048576`, `max_num_seqs=6`, `block_size=256`, `gpu_memory_utilization=0.80`
- DSpark `num_speculative_tokens=5`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`, `VLLM_USE_B12X_WO_PROJECTION=1`

## Boot

```
GPU KV cache size: 1,901,239 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.81x
Application startup complete.
```

API reports `"max_model_len": 1048576`.

## Live 6-way result

6 concurrent streaming requests (code-completion, ~200 tokens each, thinking-off):

| metric | value |
| --- | ---: |
| success | 6/6 |
| aggregate throughput | ~181.7 tok/s |
| per-stream | ~30 tok/s |
| OOM / preemption failures | none |
| single-stream (same profile) | ~67 tok/s (code) |

## Read

`1M + max_num_seqs=6` keeps the full 1M per-request ceiling while serving 6
concurrent sessions, because the ~1.9M-token KV pool is shared on demand
(not reserved per slot). With typical agent turns far below 1M, all 6 slots
stay usable and aggregate throughput is ~2.7x single-stream. The 1.81x boot
number is the *simultaneous-full-1M* limit, not the practical concurrency for
normal-length traffic. See the README "How the KV cache works" section.

Caveat: higher concurrency adds scheduler/prefill/fragmentation pressure under
sustained load. 1M/6 is validated for normal-length agent traffic; 1M/2 is
conservative and 500K/4 is a balanced middle for guaranteed deep-context work.
