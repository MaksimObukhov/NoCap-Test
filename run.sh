#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-}"
EXPERIMENT_ID="exp022"
EXPERIMENT_SLUG="swiglu-small-batch-combo"

if [[ "$MODE" != "proxy" && "$MODE" != "full" ]]; then
  echo "usage: $0 {proxy|full} [seed] [output_dir] [extra train_gpt2.py args...]" >&2
  exit 2
fi
if [[ "$SEED" != "0" ]]; then
  echo "exp022 currently authorises seed 0 only" >&2
  exit 2
fi

if [[ $# -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

case "$MODE" in
  proxy)
    NUM_ITERATIONS=3576
    WARMUP_ITERS=192
    WARMDOWN_ITERS=768
    VAL_LOSS_EVERY=256
    SAVE_EVERY=512
    MILESTONE_START_ITERS=0
    MILESTONE_EVERY=0
    ;;
  full)
    # Exact 2,700,083,200-token budget at a 262,144-token final batch.
    NUM_ITERATIONS=10300
    WARMUP_ITERS=553
    WARMDOWN_ITERS=2212
    VAL_LOSS_EVERY=256
    SAVE_EVERY=512
    MILESTONE_START_ITERS=8088
    MILESTONE_EVERY=256
    ;;
esac

DATA_DIR="${NOCAP_DATA_DIR:-}"
if [[ -z "$DATA_DIR" ]]; then
  if [[ -d "/workspace/nocap-control/data/fineweb10B" ]]; then
    DATA_DIR="/workspace/nocap-control/data/fineweb10B"
  elif [[ -d "$PWD/data/fineweb10B" ]]; then
    DATA_DIR="$PWD/data/fineweb10B"
  else
    echo "dataset not found; set NOCAP_DATA_DIR to the absolute fineweb10B directory" >&2
    exit 2
  fi
fi
if [[ "$DATA_DIR" != /* ]]; then
  echo "NOCAP_DATA_DIR must be an absolute path: $DATA_DIR" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_ROOT="${NOCAP_OUTPUT_ROOT:-/workspace/nocap-results}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}-${RUN_TIMESTAMP}/${MODE}-seed-${SEED}"
fi

RUN_NAME="${EXPERIMENT_ID}-${EXPERIMENT_SLUG}-${MODE}-seed-${SEED}"
WANDB_ARGS=()
if [[ "${WANDB_ENABLED:-1}" == "1" ]]; then
  WANDB_ARGS+=(
    --log_wandb
    --wandb_project "${WANDB_PROJECT:-nocap-baseline}"
    --wandb_group "${WANDB_GROUP:-${EXPERIMENT_ID}}"
    --wandb_checkpoint_artifact
  )
fi

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/stdout.log") 2>&1
echo "experiment=$EXPERIMENT_ID mode=$MODE seed=$SEED"
echo "data_dir=$DATA_DIR"
echo "output_dir=$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "$DATA_DIR/fineweb_train_*.bin" \
  --input_val_bin "$DATA_DIR/fineweb_val_*.bin" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --run_mode "$MODE" \
  --seed "$SEED" \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 16 \
  --batch_ramp_start_accumulation_steps 1 \
  --batch_ramp_fraction 0.5 \
  --sequence_length 1024 \
  --val_loss_every "$VAL_LOSS_EVERY" \
  --val_batch_size 16 \
  --num_iterations "$NUM_ITERATIONS" \
  --weight_decay 0.1 \
  --grad_clip 10 \
  --learning_rate 0.0018 \
  --lr_reference_batch_tokens 524288 \
  --warmup_iters "$WARMUP_ITERS" \
  --warmdown_iters "$WARMDOWN_ITERS" \
  --save_every "$SAVE_EVERY" \
  --milestone_start_iter "$MILESTONE_START_ITERS" \
  --milestone_every "$MILESTONE_EVERY" \
  --abort_on_nonfinite \
  "${WANDB_ARGS[@]}" \
  "$@"
