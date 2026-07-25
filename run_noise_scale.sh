#!/usr/bin/env bash
# exp002: measure the gradient noise scale along the baseline trajectory.
#
# Training is unchanged - this is a measurement, not an intervention. The run
# uses the proxy schedule (1788/96/384) so the trajectory matches the recorded
# baseline proxy, and stops early because the interesting dynamics are all in
# the first ~20% of the budget.
#
# Roughly 30 minutes for 400 updates on a healthy 4090.
#
# Auto-stops the Vast instance when it exits, including on a crash. `stop` ends
# GPU billing but keeps container storage, so the artifacts survive - restart
# the instance to fetch them. Set VAST_AUTO_STOP=0 to keep the box alive.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=remote/vast_autostop.sh
source "$SCRIPT_DIR/remote/vast_autostop.sh"

UPDATES="${1:-400}"
MEASURE_EVERY="${2:-8}"
SEED="${3:-0}"
OUTPUT_DIR="${4:-}"

# Fail now, not in 30 minutes, if auto-stop could not work.
vast_autostop_preflight

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/exp002-noise-scale-${RUN_TIMESTAMP}/seed-${SEED}"
fi
mkdir -p "$OUTPUT_DIR"

WANDB_ARGS=()
if [[ "${WANDB_ENABLED:-1}" == "1" ]]; then
  WANDB_ARGS+=(
    --log_wandb
    --wandb_project "${WANDB_PROJECT:-nocap-baseline}"
    --wandb_group "${WANDB_GROUP:-exp002-noise-scale}"
  )
fi

echo "exp002 noise scale: $UPDATES updates, measuring every $MEASURE_EVERY, seed $SEED"
echo "output: $OUTPUT_DIR"

# Armed only now, so a failed preflight or a bad argument does not stop the box
# before it has done any work.
vast_autostop_arm

torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "exp002-noise-scale-seed-${SEED}" \
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
  --val_loss_every 64 \
  --val_batch_size 16 \
  --save_every 0 \
  --noise_scale_every "$MEASURE_EVERY" \
  --max_updates "$UPDATES" \
  "${WANDB_ARGS[@]}" \
  "${@:5}" \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"

echo
echo "=== B_simple curve ==="
python3 analyze_noise_scale.py "$OUTPUT_DIR/metrics.jsonl" \
  2>&1 | tee "$OUTPUT_DIR/noise_scale_summary.txt"

echo
echo "Artifacts survive the stop. Restart the instance, then from the Mac:"
echo "  scp -i \"\$VAST_SSH_KEY\" -P \"\$VAST_SSH_PORT\" -r \\"
echo "    \"\$VAST_SSH_USER@\$VAST_SSH_HOST:\$REMOTE_REPO/$OUTPUT_DIR\" \\"
echo "    ./nocap-runs-backup/"
