#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${1:-exp007-proxy-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="runs/$RUN_ID/proxy-seed-0"

if [[ -z "${EXPECTED_SHA:-}" ]]; then
  echo "ERROR: EXPECTED_SHA is required." >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: wrong Git SHA; refusing to start." >&2
  git rev-parse HEAD >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected." >&2
  git status --short >&2
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is missing." >&2
  exit 2
fi
if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]]; then
  echo "ERROR: CONTAINER_ID and CONTAINER_API_KEY are required for auto-stop." >&2
  exit 2
fi
if ! command -v vastai >/dev/null 2>&1; then
  echo "ERROR: vastai CLI is missing; refusing to start without auto-stop." >&2
  exit 2
fi

stop_instance() {
  local exit_code=$?
  trap - EXIT
  sync

  echo "Stopping Vast instance $CONTAINER_ID to end GPU billing."
  local stopped=0
  for attempt in 1 2 3; do
    if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
      echo "Vast instance $CONTAINER_ID stop request succeeded."
      stopped=1
      break
    fi
    echo "WARNING: Vast stop failed (attempt $attempt); retrying in 30s." >&2
    sleep 30
  done
  if [[ "$stopped" != "1" ]]; then
    echo "ERROR: auto-stop failed; stop the instance manually." >&2
  fi
  exit "$exit_code"
}
trap stop_instance EXIT

mkdir -p "$RUN_DIR"
resume_args=()
if [[ -s "$RUN_DIR/checkpoint.pt" && ! -s "$RUN_DIR/summary.json" ]]; then
  resume_args+=(--resume "$RUN_DIR/checkpoint.pt")
fi

export WANDB_ENABLED=1
export WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
export WANDB_GROUP="${WANDB_GROUP:-exp007}"

echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "git_sha=$EXPECTED_SHA"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_group=$WANDB_GROUP"

bash run.sh proxy 0 "$RUN_DIR" "${resume_args[@]}" \
  2>&1 | tee -a "$RUN_DIR/stdout.log"

for artifact in checkpoint.pt metrics.jsonl summary.json stdout.log config.json; do
  test -s "$RUN_DIR/$artifact"
done

echo "Proxy seed 0 complete; artifacts saved in $RUN_DIR and W&B finished."
