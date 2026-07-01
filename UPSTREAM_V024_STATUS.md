# vLLM v0.24.0+ and DSpark PR Status

This note separates the released vLLM v0.24.0 DeepSeek-V4 improvements and the
now-merged DSpark PR work from the still-required DGX Spark/SM120 compatibility
work for this NVFP4 deployment.

## Short Answer

Do not blindly replace the current DSpark runtime with stock vLLM v0.24.0 or
current vLLM main and expect this repo to boot. The DSpark PR
`vllm-project/vllm#46995` has merged, and it is directly relevant to the
garble/concurrency class of bugs, but stock official vLLM still does not replace
this repo's GB10/SM120 NVFP4 runtime yet.

The current production default remains the v0.21/Keys/B12X NVFP4 image. It was
re-verified after the upstream test with:

- `max_model_len=1048576`
- `kv_cache_dtype=nvfp4_ds_mla`
- `MAX_NUM_SEQS=6`
- `MTP_NUM_TOKENS=5`
- GPU KV cache size about `2,087,950` tokens
- maximum concurrency for 1,048,576 tokens per request about `1.99x`
- direct `/v1/chat/completions` sanity request returning `OK`

The safe upgrade lane is:

1. Keep the current known-good v0.21/Keys NVFP4 image parked for rollback.
2. Build a new explicit image tag, for example:
   `vllm-dspark-runtime:dspark-v024-nvfp4-stage-c`.
3. Start from vLLM current main or a release that includes PR `#46995`.
4. Re-port the GB10/SM120 survival pieces that stock official vLLM is still
   missing for this checkpoint.
5. Re-port this repo's `nvfp4_ds_mla` and Keys concurrency overlay, or prove the
   official replacement handles the same 1M/NVFP4/2-6 concurrency traffic.
6. Validate at 262K/fp8 first, then 1M/NVFP4, then 2/4/6 concurrency.

## What v0.24.0 Helps

v0.24.0 includes DeepSeek-V4 fixes that are relevant to this deployment:

- DeepSeek-V4 OOM and memory-planning hardening.
- MTP projection-prefix naming fixes.
- supported KV-cache dtype fixes.
- DFlash / FlashInfer / scheduler work that should help the wider DSv4 family.
- DeepSeek-V4 attention, prefix-cache, and KV allocation improvements.

Those are worth staging, but they are not enough by themselves because this repo
depends on DSpark speculative decoding and the `nvfp4_ds_mla` path.

## What Is Still DSpark-Specific

PR `vllm-project/vllm#46995` is the DSpark lane, and it has merged. The important
runtime pieces are the DSpark implementation, DFlash/DSpark shared buffer
cleanup, non-contiguous Gumbel sampling fix, stale `idx_mapping` fix, padded
slot fix, zero dummy-buffer fix, speculative position clamp, and DSpark
regression coverage.

Do not cherry-pick only the two visible garble fixes and call it done. They rely
on the broader DSpark/DFlash buffer contract in the PR.

## Stock Official vLLM Test Result

On 2026-07-01, current vLLM main containing merge commit
`f5a8d73377d0f0a4e00cba172f9fbd0d50471b07` for PR `#46995` was built and tested
as `vllm-dspark-runtime:official-main-dspark-00eb7ce`.

It imported the DSpark modules successfully, but did not boot this DeepSeek V4
Flash DSpark checkpoint on 2x DGX Spark/SM120. Backend attempts failed before
generation:

- auto/DeepGEMM: `Unknown SF transformation`
- DeepGEMM disabled with Marlin fallback: unsupported PTX/toolchain error in
  Marlin FP4 repack
- `flashinfer_trtllm`: backend rejected the current CUDA device
- `flashinfer_cutlass`: selected, then failed in MXFP4 method compatibility
- `triton`: backend rejected the current CUDA device

That means PR `#46995` is relevant to the garble/concurrency fix, but the stock
official-main image is not yet a safe drop-in replacement for this repo's
current NVFP4 GB10 runtime.

For the concrete port checklist, see
[`OFFICIAL_MAIN_PORT_PLAN.md`](OFFICIAL_MAIN_PORT_PLAN.md).

## Current Repo Default

Until the v0.24 DSpark lane is proven, this repo's production default remains:

- `MAX_MODEL_LEN=1048576`
- `MAX_NUM_SEQS=6`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MTP_NUM_TOKENS=5`
- `kv_cache_dtype=nvfp4_ds_mla`
- Keys Patch 1 / Patch 2 / Patch 2b concurrency behavior
- safe server-side generation floor:
  `temperature=0.6`, `top_p=0.95`, `top_k=40`,
  `repetition_penalty=1.05`

Exact deterministic tests should send `temperature: 0` in the request body.

## Validation Gate For a v0.24 Image

Before pointing Hermes/OpenClaw-style agent traffic at a v0.24 image:

```bash
./validate-dspark-config.sh
./start-deepseek-v4-flash-dspark.sh
./smoke-deepseek-v4-flash-dspark.sh
DSPARK_BASE_URL=http://HEAD_NODE_IP:8888/v1 CONCURRENCY=1,2,4,6 \
  python3 scripts/agent_sanity_bench.py
```

Also check:

- `/v1/models` reports the intended `max_model_len`.
- no `mtp_block.main_norm` load failure.
- no BOS / placeholder-token leakage.
- no CJK drift or repeated-character loops in long direct prompts.
- no increase in vLLM preemptions under 2/4/6 concurrency.
- speed is measured against the current known-good image before switching.
