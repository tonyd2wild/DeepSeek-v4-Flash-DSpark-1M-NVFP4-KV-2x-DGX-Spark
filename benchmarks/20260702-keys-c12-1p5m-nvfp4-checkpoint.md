# Keys C12 1.5M NVFP4 Checkpoint - 2026-07-02

This checkpoint validates the Keys-style C12 profile while keeping Tony's
known-good Stage C NVFP4 runtime lane.

## Runtime

- image: `vllm-dspark-runtime:dspark-nvfp4-stage-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- topology: 2x DGX Spark, TP=2, `mp`
- `kv_cache_dtype=nvfp4_ds_mla`
- `MAX_MODEL_LEN=1500000`
- `MAX_NUM_SEQS=12`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MTP_NUM_TOKENS=5`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":40,"repetition_penalty":1.05}'`

Do not enable `VLLM_USE_B12X_FP8_GEMM=1` on this Stage C image. In testing it
selected the B12X dense FP8 path but failed during DSpark drafter warmup with a
DeepGEMM layout assertion.

## Boot Evidence

```text
GPU KV cache size: 3,225,280 tokens
Maximum concurrency for 1,500,000 tokens per request: 2.15x
Application startup complete.
```

The OpenAI-compatible `/v1/models` endpoint reported:

```json
"max_model_len": 1500000
```

## Code-Gate Speed Sweep

The code-gate workload uses 256 generated tokens, `ignore_eos=true`,
`temperature=0.0`, and server metric deltas from
`vllm:generation_tokens_total`.

Warmup:

| run | client tok/s | server generation tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 0 | 55.42 | 55.42 | 0.623 |
| 1 | 53.71 | 53.71 | 0.606 |

Sweep:

| concurrency | success | client aggregate tok/s | server generation tok/s | per-request avg tok/s | acceptance | bad outputs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 52.79 | 52.79 | 52.82 | 0.585 | 0 |
| 2 | 2/2 | 79.76 | 79.76 | 41.54 | 0.600 | 0 |
| 4 | 4/4 | 134.70 | 134.70 | 34.20 | 0.602 | 0 |
| 6 | 6/6 | 127.78 | 127.78 | 22.38 | 0.615 | 0 |
| 12 | 12/12 | 230.10 | 230.10 | 21.15 | 0.602 | 0 |

## Read

This proves the older 50-60 tok/s single-stream speed lane still exists when
measured as server generation/decode throughput on the code-gate workload. Long
agent/prose prompts can report lower client tok/s without meaning DSpark is
broken.

The practical production lane is now Tony's Stage C NVFP4 runtime plus Keys'
C12/1.5M serving profile. If acceptance drops back near 30%, first compare the
live request shape and runtime flags against this checkpoint before switching
images or enabling experimental B12X dense FP8 flags.
