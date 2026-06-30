# Agent Garble Fix - Fresh 1M NVFP4 DSpark Profile

This note is for anyone who already cloned or deployed this repo and then saw
agent traffic degrade into repeated characters, Chinese drift, leaked tool/XML
prompt text, or unstable loops.

The fix is not to drop DeepSeek V4 Flash DSpark, switch to a smaller fallback,
or move production to fp8. The current stable path keeps:

- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1048576`
- `max_num_seqs=6`
- `MTP_NUM_TOKENS=5`
- Keys Patch 2b concurrency behavior
- deterministic server defaults

## What Was Happening

The bad symptom usually appeared only under real agent traffic. Basic direct
prompts like `hi` could look fine, while Hermes/OpenClaw-style long prompts
with tools, schemas, and concurrent sessions could drift or loop.

The failures we isolated came from a mix of deployment drift and unsafe defaults:

1. A reused runtime image tag can hide an older or partial DSpark overlay.
2. Two-node launches can bind the worker to the head node's fabric IP unless
   `VLLM_HOST_IP` is explicit per node.
3. Agent clients can inherit unstable model-card sampling unless the server
   overrides generation defaults.
4. Harness testing can be contaminated by stale sessions or silent fallbacks.
5. Some worker nodes need their own checkout path and Hugging Face cache path.

## What Changed

The public recipe now carries the stable agent-serving defaults directly:

- `--default-chat-template-kwargs '{"thinking":false}'`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
- explicit `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP`
- explicit `WORKER_SCRIPT_DIR` and `WORKER_HF_CACHE`

## Update Path For Existing Deployments

From the head node checkout:

```bash
git pull
cp .env.dspark.example .env.dspark.new
```

Copy your node-specific values from the old `.env.dspark` into the new file.
At minimum verify these values:

```bash
WORKER_HOST=...
WORKER_SCRIPT_DIR=...
MASTER_ADDR=...
NCCL_IB_HCA=...
NCCL_SOCKET_IFNAME=...
NCCL_IB_GID_INDEX=...
HF_CACHE=...
WORKER_HF_CACHE=...
VLLM_HOST=0.0.0.0
VLLM_HOST_IP=...
WORKER_VLLM_HOST_IP=...
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.80
MTP_NUM_TOKENS=5
```

Then replace the old env file:

```bash
mv .env.dspark .env.dspark.before-garble-fix
mv .env.dspark.new .env.dspark
```

Rebuild both node images so stale local tags cannot keep the old overlay alive:

```bash
./build-dspark-vllm-runtime.sh
```

Restart worker-first:

```bash
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

## Verify Before Pointing Agents At It

Direct API first:

```bash
curl -fsS http://HEAD_NODE_IP:8888/v1/models
```

Confirm the model reports:

```json
"max_model_len": 1048576
```

Then run a deterministic chat check:

```bash
curl -fsS http://HEAD_NODE_IP:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-dspark",
    "messages": [{"role": "user", "content": "Reply exactly: NVFP4 DSPARK OK"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

Expected text:

```text
NVFP4 DSPARK OK
```

Check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency|Application startup complete|generation_config|override_generation_config"
```

Expected shape:

```text
GPU KV cache size: about 1.9M to 2.0M tokens
Maximum concurrency for 1,048,576 tokens per request: about 1.8x to 1.9x
Application startup complete.
```

## Agent Harness Rules

Only after direct vLLM prompts are clean, point Hermes/OpenClaw/other agents to:

```text
http://HEAD_NODE_IP:8888/v1
model: deepseek-v4-flash-dspark
context_length: 1048576
temperature: 0
top_p: 1.0
thinking: false
```

During validation:

- disable hidden fallbacks to Qwen/27B/other models
- clear or restart stale sessions if a session already garbled
- test one direct prompt, then 2/4/6 concurrent prompts, then agent traffic
- keep sampling deterministic while debugging

If direct vLLM is clean but agent traffic still garbles, the remaining problem
is probably harness/session/fallback state, not the DSpark weights.

