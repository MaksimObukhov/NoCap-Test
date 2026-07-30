#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${1:-exp004-B-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="runs/$RUN_ID/proxy-seed-0"
SUMMARY_PATH="$RUN_DIR/summary.json"
CHECKPOINT_PATH="$RUN_DIR/checkpoint.pt"
export WANDB_GROUP="$RUN_ID"

if [[ "${VAST_AUTO_STOP:-1}" == "1" ]]; then
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "CONTAINER_ID is missing; refusing to start without verified auto-stop." >&2
    exit 2
  fi
  if [[ -z "${CONTAINER_API_KEY:-}" ]]; then
    echo "CONTAINER_API_KEY is missing; auto-stop would fail unauthenticated." >&2
    exit 2
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI is missing; refusing to start without auto-stop." >&2
    exit 2
  fi
fi

stop_vast_instance() {
  local exit_code=$?
  trap - EXIT
  sync

  if [[ "${VAST_AUTO_STOP:-1}" == "1" ]]; then
    local stopped=0
    for attempt in 1 2 3; do
      if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
        echo "Vast instance stop request succeeded."
        stopped=1
        break
      fi
      echo "WARNING: Vast stop failed (attempt $attempt); retrying in 30s." >&2
      sleep 30
    done
    if [[ "$stopped" != "1" ]]; then
      echo "ERROR: auto-stop FAILED; stop the instance manually." >&2
    fi
  fi
  exit "$exit_code"
}
trap stop_vast_instance EXIT

if [[ -f "$SUMMARY_PATH" ]]; then
  echo "Proxy run is already complete: $SUMMARY_PATH"
  exit 0
fi

resume_args=()
if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo "Resuming exp004-B proxy seed 0."
  resume_args+=(--resume "$CHECKPOINT_PATH")
fi

mkdir -p "$RUN_DIR"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
bash run_exp004.sh proxy 0 "$RUN_DIR" "${resume_args[@]}" \
  2>&1 | tee -a "$RUN_DIR/stdout.log"
