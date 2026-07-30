#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${1:-exp006-proxy-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="runs/$RUN_ID/proxy-seed-0"
SUMMARY_PATH="$RUN_DIR/summary.json"
CHECKPOINT_PATH="$RUN_DIR/checkpoint.pt"
RESUME_ARGS=()

export WANDB_GROUP="$RUN_ID"

if [[ "${VAST_AUTO_STOP:-1}" == "1" ]]; then
  if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]]; then
    echo "CONTAINER_ID and CONTAINER_API_KEY are required for auto-stop." >&2
    exit 2
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI is required for auto-stop." >&2
    exit 2
  fi
fi

stop_vast_instance() {
  local exit_code=$?
  trap - EXIT
  sync

  if [[ "${VAST_AUTO_STOP:-1}" == "1" ]]; then
    stopped=0
    for attempt in 1 2 3; do
      if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
        echo "Vast instance $CONTAINER_ID stop request succeeded."
        stopped=1
        break
      fi
      echo "WARNING: Vast auto-stop failed (attempt $attempt)." >&2
      sleep 30
    done
    if [[ "$stopped" != "1" ]]; then
      echo "ERROR: auto-stop failed; stop the instance manually." >&2
    fi
  fi
  exit "$exit_code"
}
trap stop_vast_instance EXIT

if [[ -f "$SUMMARY_PATH" ]]; then
  echo "Already complete: $SUMMARY_PATH"
  exit 0
fi
if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo "Resuming exp006 seed 0 from $CHECKPOINT_PATH"
  RESUME_ARGS+=(--resume "$CHECKPOINT_PATH")
fi

mkdir -p "$RUN_DIR"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"

./run_exp006.sh proxy 0 "$RUN_DIR" "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/stdout.log"
