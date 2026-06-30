# Asusi/Spark4 1M NVFP4 Agent-Stability Checkpoint - 2026-06-30

This checkpoint records the clean agent-serving profile reproduced on the
Asusi/Spark4 DGX Spark pair after isolating the garble issue outside the
Hermes/OpenClaw harness.

## Endpoint

```text
http://100.90.25.78:8888/v1
model: deepseek-v4-flash-dspark
```

## Runtime

- image: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-keys-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- topology: 2x DGX Spark, TP=2, `mp`
- head fabric IP: `192.168.192.3`
- worker fabric IP: `192.168.192.4`
- `kv_cache_dtype=nvfp4_ds_mla`
- `MAX_MODEL_LEN=1048576`
- `MAX_NUM_SEQS=6`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTILIZATION=0.80`
- `MTP_NUM_TOKENS=5`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`

## Boot Evidence

```text
GPU KV cache size: 1,990,142 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.90x
Application startup complete.
```

The OpenAI-compatible `/v1/models` endpoint reported:

```json
"max_model_len": 1048576
```

## Direct Validation

- deterministic sanity prompt returned exactly `NVFP4 DSPARK OK`
- five longer English prompts completed with no CJK drift and no repeated junk
- prose chat wall-clock speed was roughly `35-41 tok/s`

Code-gate server decode check:

| run | server decode tok/s |
| ---: | ---: |
| 1 | 52.37 |
| 2 | 55.14 |
| 3 | 55.16 |
| mean | 54.22 |

Mean DSpark acceptance: `59.1%`.

## Concurrency Validation

Deterministic concurrent chat prompts:

| concurrency | success | aggregate tok/s | stability |
| ---: | ---: | ---: | --- |
| 2 | 2/2 | 60.95 | no CJK/repeat junk |
| 4 | 4/4 | 83.21 | no CJK/repeat junk |
| 6 | 6/6 | 104.11 | no CJK/repeat junk |

Acceptance dropped under concurrency as expected:

| concurrency | acceptance |
| ---: | ---: |
| 2 | 37.0% |
| 4 | 33.3% |
| 6 | 30.7% |

## Read

This is the current "clean and fresh" agent-serving checkpoint. It keeps 1M
context, NVFP4 KV, MTP5, and `max_num_seqs=6`.

The historical 262K/fp8 profile can still reproduce about `64 tok/s` on a code
gate, but that is a diagnostic speed lane. This checkpoint is the long-context
NVFP4 profile to point agent harnesses at first.

