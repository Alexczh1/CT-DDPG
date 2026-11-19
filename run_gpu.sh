#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

seed="${1:-0}"
environment="${3:-halfcheetah}"
output="${2:-runs/${environment}-seed${seed}}"

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "output already exists: $output" >&2
  exit 2
fi

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable")'

exec python train.py \
  --env "$environment" \
  --device cuda \
  --seed "$seed" \
  --output "$output" \
  --eval-envs 8 \
  --eval-every 2
