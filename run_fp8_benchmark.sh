#!/usr/bin/env bash
set -euo pipefail

WINNER="${1:?usage: run_fp8_benchmark.sh {exp021|exp022} WINNER_WORKTREE OUTPUT_DIR}"
WINNER_WORKTREE="${2:?usage: run_fp8_benchmark.sh {exp021|exp022} WINNER_WORKTREE OUTPUT_DIR}"
OUTPUT_DIR="${3:?usage: run_fp8_benchmark.sh {exp021|exp022} WINNER_WORKTREE OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

if [[ "$WINNER" != "exp021" && "$WINNER" != "exp022" ]]; then
  echo "winner must be exp021 or exp022" >&2
  exit 2
fi
if [[ "$WINNER_WORKTREE" != /* || "$OUTPUT_DIR" != /* ]]; then
  echo "winner worktree and output directory must be absolute" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" fp8_integration_benchmark.py \
  --winner "$WINNER" \
  --winner-worktree "$WINNER_WORKTREE" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"
