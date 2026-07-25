#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-}"
EXPERIMENT_ID="exp012"
EXPERIMENT_SLUG="3d-batch-combo"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 {smoke|proxy|full} [seed] [output_dir] [extra train_gpt2.py args...]" >&2
  exit 2
fi

if [[ $# -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

case "$MODE" in
  full)
    NUM_ITERATIONS=4768
    WARMUP_ITERS=256
    WARMDOWN_ITERS=1024
    SAVE_EVERY=512
    ;;
  proxy)
    NUM_ITERATIONS=1788
    WARMUP_ITERS=96
    WARMDOWN_ITERS=384
    SAVE_EVERY=256
    ;;
  smoke)
    NUM_ITERATIONS=596
    WARMUP_ITERS=32
    WARMDOWN_ITERS=128
    SAVE_EVERY=128
    ;;
  *)
    echo "unknown mode: $MODE (expected smoke, proxy, or full)" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/${EXPERIMENT_ID}-${EXPERIMENT_SLUG}-${RUN_TIMESTAMP}/${MODE}-seed-${SEED}"
fi

RUN_NAME="${EXPERIMENT_ID}-${EXPERIMENT_SLUG}-${MODE}-seed-${SEED}"
WANDB_ARGS=()
if [[ "${WANDB_ENABLED:-1}" == "1" ]]; then
  WANDB_ARGS+=(
    --log_wandb
    --wandb_project "${WANDB_PROJECT:-nocap-baseline}"
  )
  WANDB_ARGS+=(--wandb_group "${WANDB_GROUP:-${EXPERIMENT_ID}-${EXPERIMENT_SLUG}}")
fi

mkdir -p "$OUTPUT_DIR"
echo "mode=$MODE seed=$SEED output_dir=$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --run_mode "$MODE" \
  --seed "$SEED" \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --batch_ramp_start_accumulation_steps 8 \
  --batch_ramp_fraction 0.5 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations "$NUM_ITERATIONS" \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters "$WARMUP_ITERS" \
  --warmdown_iters "$WARMDOWN_ITERS" \
  --save_every "$SAVE_EVERY" \
  "${WANDB_ARGS[@]}" \
  "$@"
