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

## Result

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

## Important Caveat

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
584-byte sparse-MLA cache envelope while routing the runtime through
`nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel fix. The
true-layout experiments were useful for diagnosis but failed past roughly 411
real prompt tokens, so they are intentionally not presented here as the
reproducible recipe.

## Credits

This recipe stands on prior public work:

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
runtime patches, sanitized two-node launch config, and measured benchmark
artifact from the validated run.

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
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `benchmarks/` | measured 1M checkpoint evidence |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit these values for your cluster:

- `WORKER_HOST`
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

Core vLLM flags:

- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 1048576`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.80`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`

Key runtime env:

- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=0`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`

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

Expected checkpoint values:

```text
GPU KV cache size: 2,044,166 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.95x
```

## Notes

- The benchmark is single stream, not aggregate throughput.
- 1M was validated as booted/advertised `max_model_len` with KV headroom and
  short-prompt speed probes. This repo does not claim a full 1M-token retrieval
  or correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The validated profile is `MAX_NUM_SEQS=1`, greedy/low-concurrency oriented,
  `VLLM_USE_B12X_WO_PROJECTION=0`, and `VLLM_DSV4_B12X_COMPRESSED_MLA=0`.
- Worker-first startup avoids a race during multi-node `mp` initialization.
- Requires matching images on both nodes, correct NCCL/RoCE settings, and a
  two-node Blackwell-class/DGX Spark setup.
- The API binds to `127.0.0.1` by default; exposing it is a deliberate security
  choice.
- The next max-sequence ladder to try is approximately 1.25M, 1.5M, then
  1.75M, with the same boot/log/speed gates. Raw KV math alone is not enough
  because DeepSeek V4 sparse MLA also allocates max-length-dependent workspaces.
