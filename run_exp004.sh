#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SEED="${2:-0}"
OUTPUT_DIR="${3:-}"
CURRICULUM_DIR="${EXP004_CURRICULUM_DIR:-data/fineweb10B/exp004-B-proxy}"
MANIFEST="$CURRICULUM_DIR/manifest.json"

if [[ "$MODE" != "smoke" && "$MODE" != "proxy" ]]; then
  echo "usage: $0 {smoke|proxy} [seed] [output_dir] [extra train_gpt2.py args...]" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing curriculum manifest: $MANIFEST" >&2
  echo "build and audit the exp004-B stream before training" >&2
  exit 1
fi

python data/build_exp004_curriculum.py --verify-only "$MANIFEST"

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/exp004-B-${RUN_TIMESTAMP}/${MODE}-seed-${SEED}"
fi

export TRAIN_INPUT_BIN="$CURRICULUM_DIR/exp004_train_*.bin"
export TRAIN_INPUT_MANIFEST="$MANIFEST"
export WANDB_PROJECT="${WANDB_PROJECT:-nocap-exp004}"
export WANDB_GROUP="${WANDB_GROUP:-exp004-B}"

exec ./run.sh "$MODE" "$SEED" "$OUTPUT_DIR" \
  --run_name "exp004-B-${MODE}-seed-${SEED}" \
  "${@:4}"
