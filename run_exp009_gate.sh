#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CHECKPOINT_ROOT [OUTPUT_ROOT]" >&2
  echo "expected CHECKPOINT_ROOT/{proxy-seed-0,full-seed-0}/checkpoint.pt" >&2
  exit 2
fi

checkpoint_root=$1
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root=${2:-runs/exp009-rca-gate-${timestamp}}
python_bin=${REMOTE_PYTHON:-python}
input_bin=${FINEWEB_TRAIN_PATTERN:-data/fineweb10B/fineweb_train_*.bin}
proxy_checkpoint=${checkpoint_root}/proxy-seed-0/checkpoint.pt
full_checkpoint=${checkpoint_root}/full-seed-0/checkpoint.pt

for checkpoint in "$proxy_checkpoint" "$full_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

if ! compgen -G "$input_bin" >/dev/null; then
  echo "no training shards match: $input_bin" >&2
  exit 1
fi

expected_branch=exp009/rca-offline-gate
actual_branch=$(git branch --show-current)
if [[ "$actual_branch" != "$expected_branch" ]]; then
  echo "expected branch ${expected_branch}, got ${actual_branch}" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked worktree changes would invalidate exp009" >&2
  exit 1
fi

echo "experiment=exp009"
echo "branch=${actual_branch}"
echo "commit=$(git rev-parse HEAD)"
echo "proxy_checkpoint=${proxy_checkpoint}"
echo "full_checkpoint=${full_checkpoint}"
echo "input_bin=${input_bin}"
echo "output_root=${output_root}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

mkdir -p "$output_root"
"$python_bin" analyze_rca.py --self-test
"$python_bin" analyze_rca.py \
  --checkpoint "proxy=${proxy_checkpoint}" \
  --checkpoint "full=${full_checkpoint}" \
  --input-bin "$input_bin" \
  --output "${output_root}/summary.json" \
  2>&1 | tee "${output_root}/stdout.log"

echo "exp009 artifacts: ${output_root}"
