#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-runtime-capture-$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_FILE="${ENV_FILE:-.env.dspark}"
KNOWN="${KNOWN:-$HOME/.ssh/known_hosts}"
CONTAINER_NAME="${CONTAINER_NAME:-$(basename "$PWD")-vllm-dspark-1}"

mkdir -p "$OUT"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$PWD}}"

hostname > "$OUT/head-hostname.txt"
date -u > "$OUT/head-date.txt"
docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}' > "$OUT/head-ps.txt"
docker inspect "$CONTAINER_NAME" > "$OUT/head-dspark-inspect.json" 2>/dev/null || true
docker logs --tail 300 "$CONTAINER_NAME" > "$OUT/head-dspark-log-tail.txt" 2>&1 || true

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile="$KNOWN" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && hostname; date -u; docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'" \
  > "$OUT/worker-ps.txt"

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile="$KNOWN" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && docker inspect '$CONTAINER_NAME' 2>/dev/null || true" \
  > "$OUT/worker-dspark-inspect.json"

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile="$KNOWN" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && docker logs --tail 300 '$CONTAINER_NAME' 2>&1 || true" \
  > "$OUT/worker-dspark-log-tail.txt"

echo "$OUT"

