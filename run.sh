#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-}"
EXPERIMENT_ID="exp024"
EXPERIMENT_SLUG="absolute-token-staircase"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$PWD/.venv/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ "$MODE" != "full" ]]; then
  echo "usage: $0 [full 0 [output_dir] [--resume /absolute/checkpoint.pt]]" >&2
  exit 2
fi
if [[ "$SEED" != "0" ]]; then
  echo "exp024 authorises full seed 0 only" >&2
  exit 2
fi

if [[ $# -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi
RESUME_REQUESTED=0
if [[ $# -ne 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--resume" || "$2" != /* ]]; then
    echo "only an absolute --resume PATH may override the frozen exp024 launcher" >&2
    exit 2
  fi
  RESUME_REQUESTED=1
fi

NUM_ITERATIONS=10300
WARMUP_ITERS=553
WARMDOWN_ITERS=2212
VAL_LOSS_EVERY=256
SAVE_EVERY=512
MILESTONE_START_ITERS=8088
MILESTONE_EVERY=256

RUN_NAME="${EXPERIMENT_ID}-${EXPERIMENT_SLUG}-${MODE}-seed-${SEED}"
if [[ "${WANDB_ENABLED:-1}" != "1" ]]; then
  echo "exp024 requires W&B logging; WANDB_ENABLED must be 1" >&2
  exit 2
fi
WANDB_ARGS=(
  --log_wandb
  --wandb_project "${WANDB_PROJECT:-nocap-baseline}"
  --wandb_group "${WANDB_GROUP:-${EXPERIMENT_ID}}"
  --wandb_checkpoint_artifact
)

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
  OUTPUT_ROOT="${NOCAP_OUTPUT_ROOT:-$PWD/runs}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}-${RUN_TIMESTAMP}/${MODE}-seed-${SEED}"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  echo "exp024 output_dir must be absolute: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$RESUME_REQUESTED" == "0" && -e "$OUTPUT_DIR/metrics.jsonl" ]]; then
  echo "refusing to mix a fresh exp024 run with existing metrics: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/stdout.log") 2>&1
echo "experiment=$EXPERIMENT_ID mode=$MODE seed=$SEED"
echo "data_dir=$DATA_DIR"
echo "output_dir=$OUTPUT_DIR"

MINIMUM_FREE_GIB=25
if [[ "$RESUME_REQUESTED" == "1" ]]; then
  MINIMUM_FREE_GIB=5
fi
"$PYTHON_BIN" verify_exp024_preflight.py \
  --data-root "$DATA_DIR" \
  --output "$OUTPUT_DIR/dataset_manifest.json" \
  --minimum-free-gib "$MINIMUM_FREE_GIB"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "$DATA_DIR/fineweb_train_*.bin" \
  --input_val_bin "$DATA_DIR/fineweb_val_*.bin" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_manifest "$OUTPUT_DIR/dataset_manifest.json" \
  --run_name "$RUN_NAME" \
  --run_mode "$MODE" \
  --seed "$SEED" \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --exp024_staircase \
  --token_clock_batch_tokens 262144 \
  --sequence_length 1024 \
  --val_loss_every "$VAL_LOSS_EVERY" \
  --val_batch_size 16 \
  --num_iterations "$NUM_ITERATIONS" \
  --weight_decay 0.1 \
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
