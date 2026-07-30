#!/usr/bin/env bash
set -euo pipefail

GATE_DIR="${1:-runs/exp005-shape-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
GATE_STEPS="${EXP005_GATE_STEPS:-20}"
TRANSITION_STEP="${EXP005_GATE_TRANSITION_STEP:-10}"

if (( GATE_STEPS < 6 )); then
  echo "EXP005_GATE_STEPS must be at least 6" >&2
  exit 2
fi
if (( TRANSITION_STEP < 2 || TRANSITION_STEP > GATE_STEPS - 2 )); then
  echo "gate transition must leave at least two updates in each stage" >&2
  exit 2
fi

mkdir -p "$GATE_DIR"
export WANDB_ENABLED=0
export TORCH_LOGS="${TORCH_LOGS:-recompiles,graph_breaks}"

bash run.sh smoke 0 "$GATE_DIR/fixed-t1024" \
  --run_mode custom \
  --run_name exp005-gate-fixed-t1024 \
  --batch_size 16 \
  --sequence_length 1024 \
  --validation_sequence_length 1024 \
  --compile_shape_policy static \
  --num_iterations "$GATE_STEPS" \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every "$GATE_STEPS" \
  --save_every 0 \
  2>&1 | tee "$GATE_DIR/fixed-t1024.log"

bash run.sh smoke 0 "$GATE_DIR/scheduled" \
  --run_mode custom \
  --run_name exp005-gate-scheduled \
  --batch_size 32 \
  --sequence_length 512 \
  --train_shape_transition_step "$TRANSITION_STEP" \
  --train_batch_size_after 16 \
  --train_sequence_length_after 1024 \
  --validation_sequence_length 1024 \
  --compile_shape_policy static \
  --num_iterations "$GATE_STEPS" \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every "$GATE_STEPS" \
  --save_every 0 \
  2>&1 | tee "$GATE_DIR/scheduled.log"

python summarize_exp005_gate.py "$GATE_DIR" | tee "$GATE_DIR/gate-summary.log"
