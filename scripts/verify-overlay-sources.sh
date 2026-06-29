#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKERFILE="${1:-$REPO_DIR/recipe/Dockerfile.dspark-runtime-overlay}"
CONTEXT_DIR="${2:-$REPO_DIR/recipe/overlay}"

missing=0

while IFS= read -r source; do
  [ -z "$source" ] && continue
  if [ ! -e "$CONTEXT_DIR/$source" ]; then
    echo "Missing overlay source referenced by $(basename "$DOCKERFILE"): $source" >&2
    missing=1
  fi
done < <(awk '$1 == "COPY" { print $2 }' "$DOCKERFILE")

if [ "$missing" -ne 0 ]; then
  exit 1
fi

echo "Overlay source check passed for $DOCKERFILE"
