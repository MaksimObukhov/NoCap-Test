#!/usr/bin/env python3
"""Fetch every pinned branch once and create immutable detached worktrees."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def verify_sha(value, label):
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a full lowercase Git SHA, got {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--state-output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    repo_root = Path(args.repo_root).resolve()
    worktree_root = Path(args.worktree_root).resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)

    control = manifest["control"]
    verify_sha(control["sha"], "control.sha")
    actual_control_sha = run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if actual_control_sha != control["sha"]:
        raise RuntimeError(
            f"control SHA mismatch: expected {control['sha']}, got {actual_control_sha}"
        )
    if run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root):
        raise RuntimeError("control checkout has tracked changes")

    refs = [control] + manifest["experiments"]
    for reference in refs:
        verify_sha(reference["sha"], f"{reference['id']}.sha")
        branch = reference["branch"]
        subprocess.check_call(
            [
                "git",
                "fetch",
                "origin",
                f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
            ],
            cwd=repo_root,
        )
        remote_sha = run(["git", "rev-parse", f"refs/remotes/origin/{branch}"], cwd=repo_root)
        if remote_sha != reference["sha"]:
            raise RuntimeError(
                f"remote SHA mismatch for {branch}: expected {reference['sha']}, got {remote_sha}"
            )

    prepared = {
        "suite_id": manifest["suite_id"],
        "manifest": str(manifest_path),
        "control": {
            "path": str(repo_root),
            "branch": control["branch"],
            "sha": actual_control_sha,
        },
        "experiments": {},
    }
    for reference in manifest["experiments"]:
        target = worktree_root / reference["id"]
        if target.exists():
            try:
                actual = run(["git", "rev-parse", "HEAD"], cwd=target)
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(f"existing target is not a Git worktree: {target}") from error
            if actual != reference["sha"]:
                raise RuntimeError(
                    f"existing worktree SHA mismatch at {target}: expected "
                    f"{reference['sha']}, got {actual}"
                )
        else:
            subprocess.check_call(
                ["git", "worktree", "add", "--detach", str(target), reference["sha"]],
                cwd=repo_root,
            )
            actual = run(["git", "rev-parse", "HEAD"], cwd=target)
        if run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=target):
            raise RuntimeError(f"worktree has tracked changes: {target}")
        prepared["experiments"][reference["id"]] = {
            "path": str(target),
            "branch": reference["branch"],
            "sha": actual,
        }

    write_json(Path(args.state_output).resolve(), prepared)
    print(json.dumps(prepared, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
