#!/usr/bin/env bash
# Sourceable Vast.ai auto-stop, extracted from run_suite.sh.
#
# Two calls:
#   vast_autostop_preflight   before the long-running work - refuses to start
#                             unless auto-stop is actually going to work, so a
#                             missing API key is discovered in the first second
#                             rather than after the run finishes unattended
#   vast_autostop_arm         installs the EXIT trap, so a crash ends billing too
#
# VAST_AUTO_STOP=0 disables both. `stop` ends GPU billing but keeps container
# storage, so run artifacts survive and can be fetched after a restart.

vast_autostop_preflight() {
  if [[ "${VAST_AUTO_STOP:-1}" != "1" ]]; then
    echo "auto-stop disabled (VAST_AUTO_STOP=0); stop the instance yourself."
    return 0
  fi
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "CONTAINER_ID is missing; refusing to start without verified auto-stop." >&2
    echo "Set VAST_AUTO_STOP=0 only if you will stop the instance manually." >&2
    exit 2
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI is missing; install it before starting an unattended run." >&2
    exit 2
  fi
  if [[ -z "${CONTAINER_API_KEY:-}" ]]; then
    echo "CONTAINER_API_KEY is missing; auto-stop would fail unauthenticated." >&2
    echo "Set VAST_AUTO_STOP=0 only if you will stop the instance manually." >&2
    exit 2
  fi
  echo "auto-stop armed for instance $CONTAINER_ID (GPU billing ends when this exits)."
}

vast_autostop_stop() {
  local exit_code=$?
  trap - EXIT
  sync

  if [[ "${VAST_AUTO_STOP:-1}" == "1" && -n "${CONTAINER_ID:-}" ]]; then
    if command -v vastai >/dev/null 2>&1; then
      echo "Stopping Vast instance $CONTAINER_ID to end GPU billing."
      local stopped=0
      for attempt in 1 2 3; do
        if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
          echo "Vast instance $CONTAINER_ID stop request succeeded."
          stopped=1
          break
        fi
        echo "WARNING: vastai stop failed (attempt $attempt); retrying in 30s." >&2
        sleep 30
      done
      if [[ "$stopped" != "1" ]]; then
        echo "ERROR: auto-stop FAILED; stop the instance manually to end billing." >&2
      fi
    else
      echo "WARNING: vastai CLI is unavailable; stop the instance manually." >&2
    fi
  fi
  exit "$exit_code"
}

vast_autostop_arm() {
  trap vast_autostop_stop EXIT
}
