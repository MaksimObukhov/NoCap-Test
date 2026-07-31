#!/usr/bin/env bash
set -Eeuo pipefail

required_vars=(
  VAST_SSH_HOST
  VAST_SSH_PORT
  VAST_SSH_KEY
  CONTAINER_ID
  CONTAINER_API_KEY
  EXPECTED_SHA
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: $name is required." >&2
    exit 2
  fi
done

for command_name in ssh scp vastai; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: $command_name is required on the Mac." >&2
    exit 2
  fi
done

if [[ ! -r "$VAST_SSH_KEY" ]]; then
  echo "ERROR: SSH private key is not readable: $VAST_SSH_KEY" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-exp007-proxy-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/NoCap-Test}"
REMOTE_RUN_DIR="runs/$RUN_ID/proxy-seed-0"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$PWD/nocap-runs-backup}"
LOCAL_RUN_DIR="$LOCAL_BACKUP_ROOT/$RUN_ID/proxy-seed-0"
WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
WANDB_GROUP="${WANDB_GROUP:-exp007}"
SSH_TARGET="${VAST_SSH_USER:-root}@$VAST_SSH_HOST"
SSH_ARGS=(-i "$VAST_SSH_KEY" -p "$VAST_SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
SCP_ARGS=(-i "$VAST_SSH_KEY" -P "$VAST_SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

stop_instance() {
  local exit_code=$?
  trap - EXIT INT TERM
  echo "Stopping Vast instance $CONTAINER_ID to end GPU billing."
  local stopped=0
  for attempt in 1 2 3; do
    if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
      echo "Vast stop request succeeded. Destroy it manually after checking the backup."
      stopped=1
      break
    fi
    echo "WARNING: stop attempt $attempt failed." >&2
    sleep 15
  done
  if [[ "$stopped" != "1" ]]; then
    echo "ERROR: auto-stop failed; stop instance $CONTAINER_ID manually now." >&2
    exit 1
  fi
  exit "$exit_code"
}
trap stop_instance EXIT INT TERM

echo "run_id=$RUN_ID"
echo "remote_run_dir=$REMOTE_REPO/$REMOTE_RUN_DIR"
echo "local_run_dir=$LOCAL_RUN_DIR"

run_status=0
ssh "${SSH_ARGS[@]}" "$SSH_TARGET" bash -s -- \
  "$REMOTE_REPO" "$REMOTE_RUN_DIR" "$EXPECTED_SHA" "$WANDB_PROJECT" "$WANDB_GROUP" <<'REMOTE' || run_status=$?
set -Eeuo pipefail
REMOTE_REPO="$1"
REMOTE_RUN_DIR="$2"
EXPECTED_SHA="$3"
WANDB_PROJECT="$4"
WANDB_GROUP="$5"

cd "$REMOTE_REPO"
source /venv/main/bin/activate

actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: wrong SHA: expected $EXPECTED_SHA, got $actual_sha" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected." >&2
  git status --short >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "exp007/mlp-3d" ]]; then
  echo "ERROR: expected branch exp007/mlp-3d." >&2
  exit 2
fi

python - <<'PY'
import wandb
if not wandb.login(verify=True):
    raise SystemExit("W&B authentication failed")
print("W&B authentication verified")
PY

mkdir -p "$REMOTE_RUN_DIR"
WANDB_MODE=online \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_GROUP="$WANDB_GROUP" \
bash run.sh proxy 0 "$REMOTE_RUN_DIR" 2>&1 | tee -a "$REMOTE_RUN_DIR/stdout.log"

for artifact in checkpoint.pt metrics.jsonl summary.json stdout.log config.json; do
  test -s "$REMOTE_RUN_DIR/$artifact"
done
sync
REMOTE

copy_status=0
mkdir -p "$(dirname "$LOCAL_RUN_DIR")"
scp "${SCP_ARGS[@]}" -r \
  "$SSH_TARGET:$REMOTE_REPO/$REMOTE_RUN_DIR" \
  "$(dirname "$LOCAL_RUN_DIR")/" || copy_status=$?

if [[ "$copy_status" == "0" ]]; then
  for artifact in checkpoint.pt metrics.jsonl summary.json stdout.log config.json; do
    test -s "$LOCAL_RUN_DIR/$artifact"
  done
  echo "Local backup verified: $LOCAL_RUN_DIR"
else
  echo "ERROR: SCP failed; partial remote artifacts may remain on the stopped instance." >&2
fi

if [[ "$run_status" != "0" ]]; then
  echo "ERROR: remote proxy exited with status $run_status." >&2
  exit "$run_status"
fi
if [[ "$copy_status" != "0" ]]; then
  exit "$copy_status"
fi

echo "Proxy and local backup completed successfully."
