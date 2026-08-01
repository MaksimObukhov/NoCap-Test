#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CHECKPOINT_ROOT [OUTPUT_ROOT]" >&2
  exit 2
fi

checkpoint_root=$1
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root=${2:-runs/exp011-a-${timestamp}}
python_bin=${REMOTE_PYTHON:-python}
input_bin=${FINEWEB_TRAIN_PATTERN:-data/fineweb10B/fineweb_train_*.bin}
proxy_checkpoint=${checkpoint_root}/proxy-seed-0/checkpoint.pt
full_checkpoint=${checkpoint_root}/full-seed-0/checkpoint.pt

expected_branch=exp011/attn-v-spectral-cap
actual_branch=$(git branch --show-current)
[[ "$actual_branch" == "$expected_branch" ]] || {
  echo "expected branch ${expected_branch}, got ${actual_branch}" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "tracked worktree changes invalidate exp011-A" >&2
  exit 1
}
for checkpoint in "$proxy_checkpoint" "$full_checkpoint"; do
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
done
compgen -G "$input_bin" >/dev/null || { echo "missing FineWeb shards" >&2; exit 1; }

echo "experiment=exp011-A"
echo "treatment=causal attribution of attention-V spectral cap"
echo "mode=offline"
echo "branch=${actual_branch}"
echo "sha=$(git rev-parse HEAD)"
echo "launcher=run_exp011_a.sh"
echo "updates=8 per checkpoint"
echo "output_root=${output_root}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

mkdir -p "$output_root"
"$python_bin" analyze_exp011_a.py --self-test
"$python_bin" analyze_exp011_a.py \
  --checkpoint "proxy=${proxy_checkpoint}" \
  --checkpoint "full=${full_checkpoint}" \
  --input-bin "$input_bin" \
  --output "${output_root}/summary.json" \
  2>&1 | tee "${output_root}/stdout.log"

echo "exp011-A artifacts: ${output_root}"
echo "STOP: review A before running B"
