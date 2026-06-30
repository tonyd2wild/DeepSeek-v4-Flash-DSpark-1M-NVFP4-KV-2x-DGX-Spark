# DeepSeek V4 Flash DSpark 1M NVFP4 KV on 2x DGX Spark

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
with vLLM TP=2, DSpark speculative decoding, and a 1M-token max model length
using the experimental `nvfp4_ds_mla` KV-cache path.

This repo captures the validated 2026-06-29 checkpoint:

- `max_model_len=1048576`
- `kv_cache_dtype=nvfp4_ds_mla`
- reported KV pool: `2,044,166 tokens`
- reported max concurrency for 1M requests: `1.95x`
- single-stream decode stayed above `50 tok/s`
- DSpark in-server concurrency patch validated at `max_model_len=200000`,
  `max_num_seqs=16`, with static C16 at `315.1 tok/s` aggregate and
  staggered C16 at `205.0 tok/s` aggregate

## Result

### 1M NVFP4 Profile

Validated on 2x DGX Spark, one GPU per node, TP=2, single stream.

| Case | server tok/s | TTFC | acceptance | accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| p256/g64 | 54.46 | 0.506s | 0.667 | 3.33 |
| p256/g256 | 65.38 | 0.324s | 0.718 | 3.59 |
| p512/g64 | 56.26 | 2.738s | 0.625 | 3.13 |
| p512/g256 | 54.41 | 0.422s | 0.550 | 2.75 |
| p512/g256 warmup1 | 56.73 | 0.417s | 0.585 | 2.92 |

Boot logs reported:

```text
GPU KV cache size: 2,044,166 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.95x
```

The API reported:

```json
{"max_model_len":1048576}
```

The checkpoint note is in
[`benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md`](benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md).

### DSpark Concurrency Profile

Validated on the same 2x DGX Spark TP=2 deployment using Keys' DSpark
concurrency patch, `kv_cache_dtype=nvfp4_ds_mla`, `max_model_len=200000`,
`max_num_seqs=16`, `MTP_NUM_TOKENS=5`, and
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.

Patch source:

- [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- Vendored commit: `961d97b5ded1076c37429bf2820753ddac8d9a22`

The live fix documented here keeps `kv_cache_dtype=nvfp4_ds_mla` and refreshes
the repo's already-vendored Keys overlay with the path-adjusted Patch 2b update
from that commit. In Patch 2b, ragged `query_start_loc` detection no longer
depends on `num_rejected_tokens_gpu`. Treat the service as validated only after
the built-in OpenAI-compatible chat smoke request plus agent-client validation
pass on the live service.

Static simultaneous batch, one TP=2 replica:

| concurrency | best aggregate tok/s | per-stream tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 1 | 57.6 | 57.6 | 0.635 |
| 4 | 140.8 | 35.2 | 0.619 |
| 8 | 252.6 | 31.6 | 0.635 |
| 16 | 315.1 | 19.7 | 0.609 |

Staggered independent arrivals, one TP=2 replica:

| concurrency | success | aggregate tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 4 | 4/4 | 109.2 | 0.544 |
| 8 | 8/8 | 147.3 | 0.534 |
| 16 | 16/16 | 205.0 | 0.567 |

Correctness sanity check: deterministic victim output remained byte-identical
under churn. A medium-churn condense test measured `0.529` acceptance and
`99.7 tok/s` across the churn window.

The checkpoint note is in
[`benchmarks/20260629-dspark-keys-concurrency-checkpoint.md`](benchmarks/20260629-dspark-keys-concurrency-checkpoint.md).

### Historical 60 tok/s DSpark Baseline

The older ~60 tok/s number was reproduced, but it is a separate diagnostic
profile, not this repo's default 1M NVFP4 deployment:

- image rebuilt from `rafaelcaricio/vllm#1` commit `3519c3b88`
- `max_model_len=262144`
- `max_num_seqs=1`
- `kv_cache_dtype=fp8`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `temperature=0.0`, `top_p=1.0`
- measured `63.97 tok/s` on the `code_completion` gate with `67.9%`
  DSpark acceptance

Use this to diagnose image/runtime drift. Do not confuse it with the production
1M NVFP4 path. The checkpoint note is in
[`benchmarks/20260630-dspark-pr-head-262k-fp8-speed-baseline.md`](benchmarks/20260630-dspark-pr-head-262k-fp8-speed-baseline.md).

### Full-1M Concurrency Profile (production: 1M / max_num_seqs=6)

The 200K/16 profile above maximizes raw concurrency. For agent fleets that want
the **full 1M context ceiling AND concurrency**, run `max_model_len=1048576`
with `max_num_seqs=6`. Every request can still grow to 1M while up to 6 sessions
run at once, because the shared KV pool — not a per-slot reservation — is the
real limit (see [How the KV cache works](#how-the-kv-cache-works-why-1m--concurrency-is-safe)).

Validated on this deployment (NVFP4, `max_model_len=1048576`, `max_num_seqs=6`,
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`, `VLLM_USE_B12X_WO_PROJECTION=1`):

- Boot: `GPU KV cache size: 1,901,239 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
- 6 concurrent requests: **6/6 success**, **~182 tok/s aggregate** (~30 tok/s per stream), no OOM / no preemption failures
- Single-stream decode on this same profile: ~67 tok/s (code)

This is the right profile when most sessions sit far below 1M (typical agent
turns) but you still want the 1M ceiling available — ~2.7x the single-stream
throughput across the fleet without giving up context.

> Higher concurrency is not free: under sustained pressure you can see added
> scheduler churn, prefill contention, and KV fragmentation. 1M/6 is validated
> for normal-length agent traffic; for guaranteed deep-context work under load,
> 1M/2 is conservative and 500K/4 is a balanced middle.

## How the KV cache works (why 1M + concurrency is safe)

Three independent knobs, often confused:

| knob | what it is | this build |
| --- | --- | --- |
| **KV cache pool** | total shared KV memory in tokens, sized from `gpu_memory_utilization` after weights load | ~1.9–2.04M tokens (NVFP4) |
| `max_model_len` | per-request **ceiling** — how long any one request may grow | 1,048,576 (1M) |
| `max_num_seqs` | **concurrency cap** — max active sequences the scheduler runs at once | 6 |

The pool is **shared and allocated on demand**: PagedAttention hands KV blocks
to each request as it generates tokens and frees them when it finishes.
`max_model_len` and `max_num_seqs` are **ceilings, not reservations** — vLLM does
NOT pre-allocate `max_num_seqs × max_model_len` of KV. So the real constraint is:

```
sum(live tokens across all active requests) <= KV pool (~1.9M)
```

Worked examples at 1M ceiling / 6 slots:

```
6 requests x  50k tokens =  300k   fits easily
6 requests x 200k tokens =  1.2M   fits
6 requests x 317k tokens =  1.9M   ~at the limit
2 requests x 1M   tokens =  2.0M   ~at the limit  (this is the "1.81x" boot number)
6 requests x 1M   tokens =  6.0M   impossible — excess requests queue/preempt
```

The boot log's `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
only means ~1.8 *simultaneous full-1M* requests fit. But agent turns are almost
never near 1M, so 6 normal-length sessions share the pool comfortably while the
1M ceiling stays available for the rare long one. That is exactly why
`1M + max_num_seqs=6` is safe: you are not reserving 6×1M, you are sharing one
~1.9M pool across short requests under a high ceiling.

## Gotcha: gibberish, loops, Chinese drift, or prompt/XML leakage

If the model boots and basic prompts like `hi` work, but real agent traffic
randomly turns into repeated characters, Chinese drift, leaked tool/schema XML,
or Telegram-visible junk, do not assume the weights are bad.

On this deployment there are three checks to make before blaming the weights:

1. **Runtime concurrency safety:** make sure the Keys Patch 2b logic is present
   in `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`. The important
   behavior is that ragged `query_start_loc` handling does not depend on
   `num_rejected_tokens_gpu`, and the no-rejection path creates a zero rejected
   token tensor instead of falling through to unsafe request reshaping. Without
   this, concurrent DSpark requests can mix context.
2. **Runtime image provenance:** make sure the image really contains the current
   DSpark overlay. A reused local tag named `vllm-dspark-runtime:clean` caused
   misleading failures even though a nearby PR-head image worked. Rebuild from
   the intended overlay commit when in doubt.
3. **Decode/fallback safety:** for long OpenAI-compatible agent prompts, avoid
   unstable sampling and hidden fallback transitions. The server default should
   ignore the model card's sampling defaults and use deterministic generation:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "include_reasoning": false,
  "reasoning_effort": "none",
  "chat_template_kwargs": {
    "thinking": false,
    "enable_thinking": false
  }
}
```

The compose launcher now includes `--generation-config vllm` plus
`--override-generation-config '{"temperature":0.0,"top_p":1.0}'` and
`thinking=false` so default requests do not inherit unstable model-card sampling.
Agent harnesses may still add a small repetition penalty during validation if
they see loops.

Also clear agent fallback lists during validation. A model that looks fixed in
direct vLLM tests can still appear poisoned if the orchestration layer silently
falls back, reboots a session, or replays a stale prompt/tool transcript into
the visible message stream. Keep OpenClaw/Hermes changes separate from model
runtime validation unless you are deliberately testing that harness.

Validation gates to run after a live fix:

```text
direct vLLM prompts: clean
direct concurrent vLLM prompts: clean
agent harness prompts: clean, DeepSeek, no fallback
MTP5 accepted-token positions 0..4 active
```

This keeps NVFP4 KV and MTP5. Do not switch to fp8 or drop to a smaller fallback
model just to hide the symptom unless you intentionally accept the context and
quality tradeoff.

## Important Caveat

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
584-byte sparse-MLA cache envelope while routing the runtime through
`nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel fix. The
true-layout experiments were useful for diagnosis but failed past roughly 411
real prompt tokens, so they are intentionally not presented here as the
reproducible recipe.

## Credits

See [`CREDITS.md`](CREDITS.md) for the full attribution and license notes.

This recipe stands on prior public work:

- Keys / drowzeys' DSpark in-server concurrency patch:
  [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash).
  This patch fixes the request-stable DSpark main-KV slot mapping and the
  ragged `query_start_loc` path needed for real independent-arrival
  continuous batching. The concurrency results in this repo depend directly on
  that work.
- Rafael Caricio's DSpark vLLM integration:
  [rafaelcaricio/vllm#1](https://github.com/rafaelcaricio/vllm/pull/1)
- Rafael Caricio's DSpark deployment/runbook PR:
  [rafaelcaricio/spark_vllm_docker#1](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
- Fraser Price's DeepSeek V4 Flash DSpark model/runtime work:
  [fraserprice/DeepSeek-V4-Flash-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)
  and [fraserprice/dspark-vllm](https://github.com/fraserprice/dspark-vllm)
- MiaAI-Lab's two-node DGX Spark packaging and worker-first launch runbook:
  [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- **drowzeys ("Keys")** — origin of wiring the `nvfp4_ds_mla` KV-cache dtype into a DGX Spark launch recipe
  ([Keys---Full-GLM-5.2-Quantrio…](https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)).
  This build's 1M NVFP4 KV path descends from that `nvfp4_ds_mla` work.
- Upstream vLLM, FlashInfer, NVIDIA Blackwell/CUDA/NCCL tooling, and
  DeepSeek V4 Flash.
- DeepSeek-AI's DeepSpec work as the public DSpark/speculative decoding
  foundation.

Our contribution here is the 1M NVFP4-KV checkpoint recipe, the Stage A/B/C
runtime patches, sanitized two-node launch config, applying and validating
Keys' concurrency patch on the NVFP4 profile, and measured benchmark artifacts
from the validated runs.

## License Notes

Repo scripts and docs are published under this repo's `LICENSE`. The vLLM
overlay/runtime files are vLLM-derived and retain their Apache-2.0 lineage and
SPDX headers where present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL,
and model weights are separate upstream artifacts with their own licenses and
usage terms.

## Files

| path | purpose |
| --- | --- |
| `recipe/overlay/` | base DSpark vLLM overlay files |
| `recipe/Dockerfile.dspark-runtime-overlay` | builds the base DSpark runtime overlay |
| `recipe/nvfp4/Dockerfile.stage-a` | adds `nvfp4_ds_mla` dtype plumbing |
| `recipe/nvfp4/Dockerfile.stage-b` | enables DeepSeek V4 `nvfp4_ds_mla` probe path |
| `recipe/nvfp4/Dockerfile.stage-c` | switches DeepSeek V4 NVFP4 to the validated 584-byte padded envelope |
| `docker-compose.dspark.yml` | two-node vLLM/DSpark service |
| `.env.dspark.example` | sanitized cluster configuration template |
| `build-dspark-vllm-runtime.sh` | builds the Stage C image locally and on the worker |
| `prepare-dspark-model-cache.sh` | downloads/verifies the model cache |
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test; honors `WORKER_DIR` for worker-local paths |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `patches/keys-concurrency.patch` | path-adjusted Keys Patch 2b update for this repo's already-vendored DSpark concurrency overlay |
| `benchmarks/keys-concurrency/` | benchmark scripts from Keys' patch repo |
| `benchmarks/` | measured 1M and concurrency checkpoint evidence |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit these values for your cluster:

- `WORKER_HOST`
- `WORKER_DIR` if the worker checkout/deployment path differs from the head
- `MASTER_ADDR`
- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_GID_INDEX`
- `HF_CACHE`

Build the base overlay and Stage C NVFP4 image:

```bash
./build-dspark-vllm-runtime.sh
```

Prepare the model cache:

```bash
./prepare-dspark-model-cache.sh
```

Start the service:

```bash
./start-deepseek-v4-flash-dspark.sh
```

The API serves at:

```text
http://127.0.0.1:8888/v1
```

By default the service binds to `127.0.0.1`. Set `VLLM_HOST=0.0.0.0` only
when you intentionally want to expose the API beyond the head node.

## Runtime Profile

### 1M Keys-Concurrency Profile

Core vLLM flags:

- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 1048576`
- `--max-num-seqs 6`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.80`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":${MTP_NUM_TOKENS:-5}}'`

Key runtime env:

- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`

### 200k Concurrency Profile

For DSpark concurrency, use the included overlay files with Keys'
concurrency patch and set:

- `MAX_MODEL_LEN=200000`
- `MAX_NUM_SEQS=16`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`

Run the static and staggered checks:

```bash
python3 benchmarks/keys-concurrency/bench_concurrent.py http://127.0.0.1:8888 1,4,8,16
python3 benchmarks/keys-concurrency/staggered_bench.py http://127.0.0.1:8888 16 0.4
python3 benchmarks/keys-concurrency/correctness_test.py http://127.0.0.1:8888
```

### 1M Single-Stream Legacy Profile

For conservative single-stream testing, set `MAX_NUM_SEQS=1` and
`VLLM_USE_B12X_WO_PROJECTION=0`. Keep `MTP_NUM_TOKENS=5` unless you are
deliberately running an experiment; upstream Mia and Keys both validate the
DSpark path at MTP5.

## Verify

After launch:

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Confirm the returned model entry reports:

```json
"max_model_len": 1048576
```

Then check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

Expected 1M/6 checkpoint values are around:

```text
GPU KV cache size: 1.6M-1.9M tokens
Maximum concurrency for 1,048,576 tokens per request: 1.5x-1.8x
```

## Notes

- The old speed checkpoint is single stream, not aggregate throughput.
- The high-concurrency benchmark is aggregate throughput and was validated at
  `max_model_len=200000`, not full 1M context.
- Full 1M context and high concurrency compete for the same KV pool. The
  1M/6 profile is intended for normal agent traffic where most sessions sit far
  below the 1M ceiling; it is not six simultaneous full-1M requests.
- To combine DSpark concurrency with longer context, pick a lower context
  target first, then raise concurrency slowly while watching boot logs, KV
  allocation, acceptance, and request errors.
- 1M was validated as booted/advertised `max_model_len` with KV headroom and
  short-prompt speed probes. This repo does not claim a full 1M-token retrieval
  or correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The current validated agent-serving profile is `MAX_NUM_SEQS=6`,
  `MTP_NUM_TOKENS=5`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`,
  `VLLM_USE_B12X_WO_PROJECTION=1`, and `VLLM_DSV4_B12X_COMPRESSED_MLA=0`.
- Worker-first startup avoids a race during multi-node `mp` initialization.
- Requires matching images on both nodes, correct NCCL/RoCE settings, and a
  two-node Blackwell-class/DGX Spark setup.
- The API binds to `127.0.0.1` by default; exposing it is a deliberate security
  choice.
- The next max-sequence ladder to try is approximately 1.25M, 1.5M, then
  1.75M, with the same boot/log/speed gates. Raw KV math alone is not enough
  because DeepSeek V4 sparse MLA also allocates max-length-dependent workspaces.
