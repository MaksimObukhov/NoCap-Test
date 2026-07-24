#!/usr/bin/env bash
set -euo pipefail

NUM_ITERATIONS="${1:-20}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-runs/profile-$(date -u +%Y%m%dT%H%M%SZ)}"

PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-1}"

for value in \
  "$NUM_ITERATIONS" \
  "$SEED" \
  "$PROFILE_WARMUP_STEPS" \
  "$PROFILE_ACTIVE_STEPS"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "expected non-negative integer, got: $value" >&2
    exit 2
  fi
done

if (( NUM_ITERATIONS < 2 )); then
  echo "num_iterations must be at least 2" >&2
  exit 2
fi
if (( PROFILE_ACTIVE_STEPS < 1 )); then
  echo "PROFILE_ACTIVE_STEPS must be at least 1" >&2
  exit 2
fi

# By default, use all preceding optimizer steps as runtime warmup and profile
# the final active step(s). Override this through the environment if needed.
PROFILE_WAIT_STEPS="${PROFILE_WAIT_STEPS:-$((NUM_ITERATIONS - PROFILE_WARMUP_STEPS - PROFILE_ACTIVE_STEPS))}"

if ! [[ "$PROFILE_WAIT_STEPS" =~ ^[0-9]+$ ]]; then
  echo "PROFILE_WAIT_STEPS must be a non-negative integer" >&2
  exit 2
fi

PROFILE_SCHEDULE_STEPS=$((PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS))
if (( PROFILE_SCHEDULE_STEPS > NUM_ITERATIONS )); then
  echo "profile wait + warmup + active exceeds num_iterations" >&2
  exit 2
fi

RUN_NAME="profile-${NUM_ITERATIONS}steps-seed-${SEED}"

echo "run_name=$RUN_NAME"
echo "output_dir=$OUTPUT_DIR"
echo "profile schedule: wait=$PROFILE_WAIT_STEPS warmup=$PROFILE_WARMUP_STEPS active=$PROFILE_ACTIVE_STEPS"

WANDB_ENABLED=0 bash run.sh smoke "$SEED" "$OUTPUT_DIR" \
  --run_mode custom \
  --run_name "$RUN_NAME" \
  --num_iterations "$NUM_ITERATIONS" \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every 0 \
  --save_every 0 \
  --profile \
  --profile_wait_steps "$PROFILE_WAIT_STEPS" \
  --profile_warmup_steps "$PROFILE_WARMUP_STEPS" \
  --profile_active_steps "$PROFILE_ACTIVE_STEPS"
