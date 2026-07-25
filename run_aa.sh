#!/usr/bin/env bash
# A/A gate for exp002.
#
# Runs the instrumented binary with --noise_scale_every 0, i.e. the measurement
# code is compiled in but never taken. Two things must hold before the 30-minute
# measurement run is worth starting:
#
#   1. the instrumented binary still reproduces the recorded baseline, so the
#      B_simple numbers describe the baseline trajectory and not a drifted one;
#   2. this rented card reproduces the baseline at all.
#
# Cheap: 50 updates, roughly 5 minutes on a healthy 4090.
set -euo pipefail

SEED="${1:-0}"
UPDATES="${2:-50}"
OUTPUT_DIR="${3:-}"

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/exp002-aa-${RUN_TIMESTAMP}/aa-seed-${SEED}"
fi
mkdir -p "$OUTPUT_DIR"

echo "A/A gate: $UPDATES updates, seed $SEED, output $OUTPUT_DIR"
echo "reference: nocap-runs-backup/baseline-20260718T074451Z/proxy-seed-${SEED}/metrics.jsonl"

# Proxy schedule, so get_lr() sees exactly the same 1788/96/384 it saw in the
# baseline proxy run. --max_updates only truncates it, it does not rescale it.
torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "exp002-aa-seed-${SEED}" \
  --run_mode proxy \
  --seed "$SEED" \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --num_iterations 1788 \
  --warmup_iters 96 \
  --warmdown_iters 384 \
  --learning_rate 0.0018 \
  --weight_decay 0.1 \
  --val_loss_every 0 \
  --val_batch_size 16 \
  --save_every 0 \
  --noise_scale_every 0 \
  --max_updates "$UPDATES" \
  "${@:4}" \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"

echo
echo "=== A/A comparison ==="
python3 compare_aa.py \
  "nocap-runs-backup/baseline-20260718T074451Z/proxy-seed-${SEED}/metrics.jsonl" \
  "$OUTPUT_DIR/metrics.jsonl"
