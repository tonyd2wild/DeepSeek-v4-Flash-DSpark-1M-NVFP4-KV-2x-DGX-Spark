# Official vLLM Main Port Plan

This document tracks the work needed before this repo can replace its current
known-good GB10 runtime with stock vLLM current-main plus a small compatibility
overlay.

## Current Finding

The DSpark PR `vllm-project/vllm#46995` has merged into vLLM main and is in the
right lane for the garble/concurrency issue. However, current official vLLM main
does not yet boot this DeepSeek V4 Flash DSpark NVFP4 deployment on 2x DGX
Spark/SM120.

The live production-safe lane remains:

- v0.21-based DSpark runtime
- Keys concurrency behavior
- `kv_cache_dtype=nvfp4_ds_mla`
- B12X MXFP4 MoE backend
- `MAX_MODEL_LEN=1048576`
- `MAX_NUM_SEQS=6`
- `MTP_NUM_TOKENS=5`

## Prototype Patch Built

A Python-only official-main compatibility overlay was built on 2026-07-01:

- base: `vllm-dspark-runtime:official-main-dspark-00eb7ce`
- prototype: `vllm-dspark-runtime:official-main-dspark-b12x-nvfp4-proto-00eb7ce`

The overlay passed runtime smoke checks for:

- `nvfp4` -> `nvfp4_ds_mla` DeepSeek V4 KV-cache dtype resolution.
- `DeepseekV4FlashMLABackend.get_kv_cache_shape(..., "nvfp4_ds_mla")`
  returning the padded 584-byte layout.
- MXFP4 oracle mapping of `flashinfer_b12x` to `B12xExperts`.

This is not promoted to the default launcher yet. The remaining gate is a full
model boot plus direct generation and 2/4/6 concurrency validation.

Files:

- [`patches/official-main-b12x-nvfp4-python.patch`](patches/official-main-b12x-nvfp4-python.patch)
- [`recipe/official-main/Dockerfile.python-patch`](recipe/official-main/Dockerfile.python-patch)

## Official Main Boot Failures

Official-main image tested:

- image tag: `vllm-dspark-runtime:official-main-dspark-00eb7ce`
- vLLM head: `00eb7ce`
- includes DSpark merge commit: `f5a8d73377d0f0a4e00cba172f9fbd0d50471b07`

Backend attempts failed before generation:

| Attempt | Result |
| --- | --- |
| auto / DeepGEMM MXFP4 | fails with `Unknown SF transformation` on SM120 |
| DeepGEMM disabled / Marlin fallback | fails during FP4 Marlin repack with unsupported PTX/toolchain |
| `--moe-backend flashinfer_trtllm` | backend rejects current CUDA device |
| `--moe-backend flashinfer_cutlass` | selects, then fails in MXFP4 method compatibility |
| `--moe-backend triton` | backend rejects current CUDA device |

## Why This Is Not a One-Flag Fix

DeepSeek V4 Flash enters official vLLM through the MXFP4 MoE path:

- `vllm/model_executor/layers/quantization/mxfp4.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`

Official-main already has a FlashInfer B12X expert implementation, but it is
currently wired for NVFP4, not DeepSeek V4's MXFP4 expert weights:

- `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`

That class asserts NVFP4 quantization and supports `kNvfp4Static`, while
DeepSeek V4 Flash selects `kMxfp4Static`.

The current stable runtime's missing MXFP4 bridge is represented by:

- `vllm/model_executor/layers/fused_moe/b12x_moe.py`
- `vllm/models/deepseek_v4/nvidia/sm120.py`
- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`
- `nvfp4_ds_mla` cache dtype wiring in config, KV cache, and DeepSeek attention

## Required Port

The likely compatibility overlay is not just PR `#46995`. It needs:

1. Official DSpark main as the base.
2. An MXFP4 B12X MoE backend integrated into official-main's modular MoE stack.
3. A `flashinfer_b12x` or `b12x_mxfp4` mapping in the MXFP4 oracle.
4. `make_mxfp4_moe_quant_config` support for the B12X W4A16 contract.
5. Weight conversion/retention logic equivalent to the stable runtime's
   `B12xExperts`.
6. `nvfp4_ds_mla` cache dtype support or a proven official replacement with the
   same 1M-context KV pool and stability.
7. Full direct API validation before any Hermes/OpenClaw agent traffic:
   deterministic sanity, long prompts, and 2/4/6 concurrency.

## Current Decision

Do not point agents at stock official-main for this repo yet. It contains the
right DSpark correctness work, but the GB10/SM120 DeepSeek V4 Flash NVFP4
serving path still needs a compatibility port before it can replace the current
runtime.
