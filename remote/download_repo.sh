#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${NOCAP_REPO_URL:-https://github.com/MaksimObukhov/NoCap-Test.git}"
BRANCH="${NOCAP_BRANCH:-max/implementation}"
TARGET_DIR="${1:-NoCap-Test}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: download_repo.sh [target_dir]

Environment:
  NOCAP_REPO_URL  Git repository to clone or update.
  NOCAP_BRANCH    Branch to check out. Default: max/implementation.

The script refuses to update a dirty checkout.
EOF
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  exit 1
fi

if [[ -e "$TARGET_DIR" && ! -d "$TARGET_DIR/.git" ]]; then
  echo "ERROR: target exists but is not a Git checkout: $TARGET_DIR" >&2
  exit 1
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  if [[ -n "$(git -C "$TARGET_DIR" status --porcelain)" ]]; then
    echo "ERROR: checkout is dirty; refusing to pull into $TARGET_DIR" >&2
    git -C "$TARGET_DIR" status --short >&2
    exit 1
  fi

  ACTUAL_ORIGIN="$(git -C "$TARGET_DIR" remote get-url origin)"
  normalize_repo_url() {
    local url="$1"
    url="${url#git@github.com:}"
    url="${url#https://github.com/}"
    url="${url%.git}"
    printf '%s\n' "$url"
  }

  if [[ "$(normalize_repo_url "$ACTUAL_ORIGIN")" != "$(normalize_repo_url "$REPO_URL")" ]]; then
    echo "ERROR: origin does not match NOCAP_REPO_URL." >&2
    echo "actual:   $ACTUAL_ORIGIN" >&2
    echo "expected: $REPO_URL" >&2
    exit 1
  fi

  git -C "$TARGET_DIR" fetch origin "$BRANCH"
  if git -C "$TARGET_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$TARGET_DIR" switch "$BRANCH"
  else
    git -C "$TARGET_DIR" switch --create "$BRANCH" --track "origin/$BRANCH"
  fi
  git -C "$TARGET_DIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$TARGET_DIR"
fi

echo "Repository ready:"
git -C "$TARGET_DIR" status --short --branch
git -C "$TARGET_DIR" log -1 --oneline
echo
echo "Next:"
echo "  cd \"$TARGET_DIR\""
echo "  ./remote/prepare_instance.sh"
