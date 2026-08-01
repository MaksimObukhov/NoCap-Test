#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 EXP011_A_SUMMARY [RUN_ROOT]" >&2
  exit 2
fi

a_summary=$1
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${2:-runs/exp011-b-${timestamp}}
expected_branch=exp011/attn-v-spectral-cap
actual_branch=$(git branch --show-current)

[[ "$actual_branch" == "$expected_branch" ]] || {
  echo "expected branch ${expected_branch}, got ${actual_branch}" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "tracked worktree changes invalidate exp011-B" >&2
  exit 1
}
[[ ! -e "$run_root" ]] || { echo "run root exists: $run_root" >&2; exit 1; }
[[ -f "$a_summary" ]] || { echo "missing A summary: $a_summary" >&2; exit 1; }
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["decision"] == "pass", "exp011-A did not pass"' "$a_summary"
compgen -G 'data/fineweb10B/fineweb_train_*.bin' >/dev/null || {
  echo "missing FineWeb shards" >&2
  exit 1
}

echo "experiment=exp011-B"
echo "treatment=attention-V selective spectral AdamW"
echo "mode=benchmark"
echo "branch=${actual_branch}"
echo "sha=$(git rev-parse HEAD)"
echo "launcher=run_exp011_b.sh"
echo "updates=50 control-before + 50 treatment + 50 control-after"
echo "seed=0"
echo "auto_stop=disabled"
echo "run_root=${run_root}"
echo "a_summary=${a_summary}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python -m unittest test_exp011_spectral_cap.py -v
mkdir -p "$run_root"

WANDB_ENABLED=0 \
TORCH_LOGS=recompiles \
TORCHINDUCTOR_CACHE_DIR="${run_root}/control-before/torchinductor-cache" \
bash run.sh smoke 0 "${run_root}/control-before" \
  --run_mode custom \
  --run_name exp011-b-control-before \
  --num_iterations 50 \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every 0 \
  --save_every 0 \
  --skip_final_checkpoint \
  2>&1 | tee "${run_root}/control-before-stdout.log"
mv "${run_root}/control-before-stdout.log" "${run_root}/control-before/stdout.log"

WANDB_ENABLED=0 \
TORCH_LOGS=recompiles \
TORCHINDUCTOR_CACHE_DIR="${run_root}/treatment/torchinductor-cache" \
bash run.sh smoke 0 "${run_root}/treatment" \
  --run_mode custom \
  --run_name exp011-b-treatment \
  --num_iterations 50 \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every 0 \
  --save_every 0 \
  --skip_final_checkpoint \
  --spectral_attn_v \
  2>&1 | tee "${run_root}/treatment-stdout.log"
mv "${run_root}/treatment-stdout.log" "${run_root}/treatment/stdout.log"

WANDB_ENABLED=0 \
TORCH_LOGS=recompiles \
TORCHINDUCTOR_CACHE_DIR="${run_root}/control-after/torchinductor-cache" \
bash run.sh smoke 0 "${run_root}/control-after" \
  --run_mode custom \
  --run_name exp011-b-control-after \
  --num_iterations 50 \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every 0 \
  --save_every 0 \
  --skip_final_checkpoint \
  2>&1 | tee "${run_root}/control-after-stdout.log"
mv "${run_root}/control-after-stdout.log" "${run_root}/control-after/stdout.log"

python summarize_exp011_b.py "$run_root"
sync
echo "exp011-B artifacts: ${run_root}"
echo "STOP: review B before running C"
