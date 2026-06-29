# DSpark Keys Concurrency Checkpoint - 2026-06-29

## Result

We reproduced the DSpark in-server concurrency breakthrough on a 2x DGX Spark
TP=2 setup using Keys' released patch.

Runtime configuration:

- Runtime image: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency`
- API: `http://127.0.0.1:8888` on the head node
- Served model: `deepseek-v4-flash-dspark`
- Runtime observed `max_model_len`: `200000`
- `max_num_seqs`: `16`
- KV cache dtype: `nvfp4_ds_mla`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`

Patch source:

- Repo: https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- Commit tested: `7e4d94bbcec95223550517c0fa9244e59f9f6483`
- Applied files:
  - `vllm/v1/spec_decode/dspark_proposer.py`
  - `vllm/models/deepseek_v4/nvidia/dspark.py`
  - `vllm/v1/worker/gpu_model_runner.py`

## Benchmarks

Static simultaneous batch, one TP=2 replica:

| concurrency | best aggregate tok/s | per-stream tok/s | acceptance |
|---:|---:|---:|---:|
| 1 | 57.6 | 57.6 | 0.635 |
| 4 | 140.8 | 35.2 | 0.619 |
| 8 | 252.6 | 31.6 | 0.635 |
| 16 | 315.1 | 19.7 | 0.609 |

Staggered independent arrivals, one TP=2 replica:

| concurrency | success | aggregate tok/s | acceptance |
|---:|---:|---:|---:|
| 4 | 4/4 | 109.2 | 0.544 |
| 8 | 8/8 | 147.3 | 0.534 |
| 16 | 16/16 | 205.0 | 0.567 |

Compared with the released patch README numbers, our static C16 result was slightly higher (`315.1` vs `301.2` tok/s), and our staggered C16 result was also higher (`205.0` vs `190.2` tok/s).

## Correctness

The stock `correctness_test.py` produced byte-identical output under churn twice, but its final verdict failed because the acceptance threshold was below `0.4` on tiny 16-token greeting churn:

- Run 1: byte-identical `True`, acceptance `0.299`
- Run 2: byte-identical `True`, acceptance `0.306`

I then ran a medium-churn condense test that still forced requests to start/finish around the victim request:

- Byte-identical output: `True`
- Acceptance: `0.529`
- Churn window throughput: `99.7 tok/s`

Interpretation: the req-id stable slot patch appears to be preserving correctness under condense. The stock test's failure label is caused by acceptance being low on extremely short churn requests, not by visible output corruption.

## What Changed

This replaces our earlier partial workaround.

The earlier local attempt had:

- Stable req-id KV slots.
- A ragged mixed-batch bypass that returned zero draft tokens.
- A scheduler gate that reduced real continuous batching.

Keys' patch has the missing real implementation:

- Stable request-to-slot mapping for persistent DSpark main KV.
- Ragged `query_start_loc` handling in `prepare_context`, `prefill_main`, and `store_main_kv`.
- No scheduler gate needed.

## Attribution

Keys / drowzeys gets direct credit for the concurrency patch. This repo applies
and validates that patch on the NVFP4 2x DGX Spark profile.
