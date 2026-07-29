#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${REMOTE_PYTHON:-python}"
MODE="${1:-smoke}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-64}"
CLASSIFIER_DTYPE="${CLASSIFIER_DTYPE:-float32}"
FULL_OUTPUT="${FINEWEB_EDU_OUTPUT:-$REPO_ROOT/data/fineweb10B/fineweb_edu_scores.jsonl}"

if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  cat <<'EOF'
Usage: classify_fineweb_edu.sh [smoke|full]

smoke  Classify 256 documents into a timestamped runs/ directory.
full   Classify the exact 2.5B-token baseline subset; safely resumes.

Environment:
  REMOTE_PYTHON         Python from the CUDA environment (default: python)
  CLASSIFIER_BATCH_SIZE CUDA inference batch size (default: 64)
  CLASSIFIER_DTYPE      float32, float16, or bfloat16 (default: float32)
  FINEWEB_EDU_OUTPUT    Full-run JSONL path
EOF
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "ERROR: expected mode smoke or full, got: $MODE" >&2
  exit 2
fi

cd "$REPO_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required on the remote host." >&2
  exit 1
fi

uv pip install \
  --python "$PYTHON_BIN" \
  -r data/requirements-fineweb-edu.txt

"$PYTHON_BIN" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

COMMON_ARGS=(
  --input-bin "data/fineweb10B/fineweb_train_*.bin"
  --device cuda
  --dtype "$CLASSIFIER_DTYPE"
  --classifier-batch-size "$CLASSIFIER_BATCH_SIZE"
)

if [[ "$MODE" == "smoke" ]]; then
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="$REPO_ROOT/runs/fineweb-edu-smoke-$RUN_TIMESTAMP"
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" data/classify_fineweb_edu.py \
    "${COMMON_ARGS[@]}" \
    --max-documents 256 \
    --output "$OUTPUT_DIR/scores.jsonl" \
    2>&1 | tee "$OUTPUT_DIR/stdout.log"
  echo "Smoke output: $OUTPUT_DIR"
else
  mkdir -p "$(dirname -- "$FULL_OUTPUT")"
  "$PYTHON_BIN" data/classify_fineweb_edu.py \
    "${COMMON_ARGS[@]}" \
    --resume \
    --output "$FULL_OUTPUT" \
    2>&1 | tee -a "${FULL_OUTPUT}.stdout.log"
  echo "Full output: $FULL_OUTPUT"
  echo "Summary: ${FULL_OUTPUT}.summary.json"
fi
