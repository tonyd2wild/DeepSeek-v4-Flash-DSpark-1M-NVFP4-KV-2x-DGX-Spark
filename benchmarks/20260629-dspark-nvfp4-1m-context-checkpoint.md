# DSpark NVFP4 1M Context Checkpoint - 2026-06-29

## Verdict

We reached the requested checkpoint: the live DSpark service boots with NVFP4 KV cache at 1M max model length and maintains >50 tok/s on the standard single-stream probe.

This is the Stage C padded NVFP4 path, not the unresolved true-layout 416-byte kernel path.

## Live Runtime

- Head/worker: two DGX Spark nodes, one GPU per node, TP=2.
- Image: Stage C NVFP4 runtime, published here as `vllm-dspark-runtime:dspark-nvfp4-stage-c`.
- KV dtype: `nvfp4_ds_mla`
- `MAX_MODEL_LEN=1048576`
- `MAX_NUM_SEQS=1`
- `MAX_NUM_BATCHED_TOKENS=8192`
- WO fallback: off, `VLLM_USE_B12X_WO_PROJECTION=0`
- Compressed MLA: off, `VLLM_DSV4_B12X_COMPRESSED_MLA=0`

API `/v1/models` reported:

- `max_model_len: 1048576`

Boot logs reported:

- Available KV cache memory: `14.48 GiB`
- GPU KV cache size: `2,044,166 tokens`
- Maximum concurrency for `1,048,576` tokens/request: `1.95x`

## Single-Stream Probe

 Probe: standard single-stream p/g dimension sweep against the OpenAI-compatible API.

| Case | Tok/s | TTFC | Acceptance | Accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| p256/g64 | 54.46 | 0.506s | 0.667 | 3.33 |
| p256/g256 | 65.38 | 0.324s | 0.718 | 3.59 |
| p512/g64 | 56.26 | 2.738s | 0.625 | 3.13 |
| p512/g256 | 54.41 | 0.422s | 0.550 | 2.75 |
| p512/g256 warmup1 | 56.73 | 0.417s | 0.585 | 2.92 |

All measured cases stayed above the 50 tok/s target.

## Important Caveat

This does not mean the true-layout NVFP4 kernel problem is solved. The true-layout F/G/J path still failed past roughly 411 real prompt tokens. The working 1M checkpoint uses Stage C padded NVFP4, which keeps DeepSeek V4's known-good 584-byte cache envelope while using the `nvfp4_ds_mla` path.

The practical result is still strong: the live service currently advertises 1M max model length, reports a 2.04M-token KV pool, and keeps the single-stream speed target.
