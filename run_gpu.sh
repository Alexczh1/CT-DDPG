#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

seed="${1:-0}"
output="${2:-runs/halfcheetah-seed${seed}}"

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "output already exists: $output" >&2
  exit 2
fi

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable")'

exec python train_halfcheetah.py \
  --device cuda \
  --seed "$seed" \
  --output "$output" \
  --num-envs 8 \
  --eval-envs 8 \
  --episodes 200 \
  --eval-every 2 \
  --eval-episodes 1 \
  --dt 0.05 \
  --horizon 50 \
  --batch-size 256 \
  --min-sequence-length 2 \
  --max-sequence-length 10 \
  --warmup-episodes 5 \
  --hidden-dim 400 \
  --lr-decay-steps 80000 \
  --lr-decay-gamma 0.8 \
  --force-noise-std 0.0
