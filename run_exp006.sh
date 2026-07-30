#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-}"

case "$MODE" in
  proxy)
    BASE_MODE=proxy
    TRANSITION_STEP=384
    EXTRA_ARGS=()
    ;;
  smoke)
    BASE_MODE=smoke
    TRANSITION_STEP=4
    EXTRA_ARGS=(
      --run_mode custom
      --num_iterations 8
      --warmup_iters 1
      --warmdown_iters 2
      --val_loss_every 4
      --save_every 0
    )
    ;;
  *)
    echo "usage: $0 {smoke|proxy} [seed] [output_dir] [extra train args...]" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/exp006-${RUN_TIMESTAMP}/${MODE}-seed-${SEED}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
export WANDB_GROUP="${WANDB_GROUP:-exp006}"
export TORCH_LOGS="${TORCH_LOGS:-recompiles,graph_breaks}"

exec ./run.sh "$BASE_MODE" "$SEED" "$OUTPUT_DIR" \
  --run_name "exp006-${MODE}-seed-${SEED}" \
  --batch_size 32 \
  --sequence_length 512 \
  --train_shape_transition_step "$TRANSITION_STEP" \
  --train_batch_size_after 16 \
  --train_sequence_length_after 1024 \
  --validation_sequence_length 1024 \
  --compile_shape_policy static \
  "${EXTRA_ARGS[@]}" \
  "${@:4}"
