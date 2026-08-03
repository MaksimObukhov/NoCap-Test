#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:?usage: run_exp016_ab.sh RESULTS_ROOT CHECKPOINT_ROOT DATA_ROOT}"
CHECKPOINT_ROOT="${2:?usage: run_exp016_ab.sh RESULTS_ROOT CHECKPOINT_ROOT DATA_ROOT}"
DATA_ROOT="${3:?usage: run_exp016_ab.sh RESULTS_ROOT CHECKPOINT_ROOT DATA_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

if [[ "$RESULTS_ROOT" != /* || "$CHECKPOINT_ROOT" != /* || "$DATA_ROOT" != /* ]]; then
  echo "all paths must be absolute" >&2
  exit 2
fi

PROXY_DIR="$CHECKPOINT_ROOT/nocap-runs-backup/baseline-20260718T074451Z/proxy-seed-0"
FULL_DIR="$CHECKPOINT_ROOT/nocap-runs-backup/baseline-20260718T074451Z/full-seed-0"
PROXY_CHECKPOINT="$PROXY_DIR/checkpoint.pt"
FULL_CHECKPOINT="$FULL_DIR/checkpoint.pt"
TRAIN_GLOB="$DATA_ROOT/fineweb_train_*.bin"
VAL_GLOB="$DATA_ROOT/fineweb_val_*.bin"
A_DIR="$RESULTS_ROOT/exp016/causal-a"
B_DIR="$RESULTS_ROOT/exp016/systems-b"
PREFLIGHT_DIR="$RESULTS_ROOT/exp016/preflight"

mkdir -p "$PREFLIGHT_DIR"
"$PYTHON_BIN" verify_exp016_inputs.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --data-root "$DATA_ROOT" \
  --output "$PREFLIGHT_DIR/input_manifest.json" \
  2>&1 | tee "$PREFLIGHT_DIR/input_manifest.log"
"$PYTHON_BIN" analyze_exp016_a.py --self-test \
  2>&1 | tee "$PREFLIGHT_DIR/causal_self_test.log"

mkdir -p "$A_DIR"
"$PYTHON_BIN" analyze_exp016_a.py \
  --checkpoint "proxy=$PROXY_CHECKPOINT" \
  --checkpoint "full=$FULL_CHECKPOINT" \
  --input-bin "$TRAIN_GLOB" \
  --output "$A_DIR/summary.json" \
  2>&1 | tee "$A_DIR/stdout.log"

A_DECISION="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$A_DIR/summary.json")"
if [[ "$A_DECISION" != "pass" ]]; then
  echo "exp016-A decision=$A_DECISION; exp016-B is intentionally skipped"
  exit 0
fi

mkdir -p "$B_DIR"
export TORCHINDUCTOR_CACHE_DIR="$RESULTS_ROOT/compile-cache/exp016-b"
"$PYTHON_BIN" run_exp016_b.py \
  --output-dir "$B_DIR" \
  --input-bin "$TRAIN_GLOB" \
  --input-val-bin "$VAL_GLOB" \
  2>&1 | tee "$B_DIR/stdout.log"

B_DECISION="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$B_DIR/summary.json")"
echo "exp016 complete: A=$A_DECISION B=$B_DECISION"
