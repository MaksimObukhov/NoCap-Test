#!/usr/bin/env bash
set -Eeuo pipefail

SUITE_ID="${1:-baseline-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITE_DIR="runs/$SUITE_ID"
export WANDB_GROUP="$SUITE_ID"

mkdir -p "$SUITE_DIR"

if [[ "${VAST_AUTO_STOP:-1}" == "1" ]]; then
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "CONTAINER_ID is missing; refusing to start without verified auto-stop." >&2
    echo "Set VAST_AUTO_STOP=0 only if you will stop the instance manually." >&2
    exit 2
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI is missing; install it before starting the suite." >&2
    exit 2
  fi
fi

stop_vast_instance() {
  local exit_code=$?
  trap - EXIT
  sync

  if [[ "${VAST_AUTO_STOP:-1}" == "1" && -n "${CONTAINER_ID:-}" ]]; then
    if command -v vastai >/dev/null 2>&1; then
      echo "Stopping Vast instance $CONTAINER_ID to end GPU billing."
      vastai stop instance "$CONTAINER_ID" || true
    else
      echo "WARNING: vastai CLI is unavailable; stop the instance manually." >&2
    fi
  fi
  exit "$exit_code"
}
trap stop_vast_instance EXIT

run_one() {
  local mode="$1"
  local seed="$2"
  local run_dir="$SUITE_DIR/${mode}-seed-${seed}"
  local summary_path="$run_dir/summary.json"
  local checkpoint_path="$run_dir/checkpoint.pt"
  local resume_args=()

  if [[ -f "$summary_path" ]]; then
    echo "Already complete: $mode seed $seed"
    return
  fi
  if [[ -f "$checkpoint_path" ]]; then
    echo "Resuming: $mode seed $seed"
    resume_args+=(--resume "$checkpoint_path")
  fi

  mkdir -p "$run_dir"
  bash run.sh "$mode" "$seed" "$run_dir" "${resume_args[@]}" \
    2>&1 | tee -a "$run_dir/stdout.log"
}

echo "suite_id=$SUITE_ID"
echo "suite_dir=$SUITE_DIR"

run_one proxy 0
run_one proxy 1
run_one proxy 2
run_one full 0

python summarize_suite.py "$SUITE_DIR" | tee "$SUITE_DIR/statistics.log"
echo "Suite complete: $SUITE_DIR"
