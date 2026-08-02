#!/usr/bin/env python3
"""Create or validate detached experiment worktrees without switching branches."""

import argparse
import json
import subprocess
from pathlib import Path


def git(repo, *arguments, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    root = Path(args.worktree_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    if args.fetch:
        for entry in manifest["experiments"].values():
            branch = entry["branch"]
            subprocess.check_call(
                [
                    "git",
                    "-C",
                    str(repo),
                    "fetch",
                    "origin",
                    f"{branch}:refs/remotes/origin/{branch}",
                ]
            )

    root.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for name, entry in manifest["experiments"].items():
        commit = entry["commit"]
        if git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode:
            raise SystemExit(f"missing pinned commit for {name}: {commit}")
        path = root / name
        if path.exists():
            if not (path / ".git").exists():
                raise SystemExit(f"refusing non-worktree path: {path}")
            actual = git(path, "rev-parse", "HEAD").stdout.strip()
            if actual != commit:
                raise SystemExit(
                    f"existing {name} worktree is {actual}, expected {commit}: {path}"
                )
        else:
            subprocess.check_call(
                ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), commit]
            )
        dirty = git(path, "status", "--porcelain", "--untracked-files=no").stdout
        if dirty:
            raise SystemExit(f"tracked changes in {name} worktree: {path}\n{dirty}")
        resolved[name] = {
            "path": str(path),
            "branch": entry["branch"],
            "commit": commit,
        }

    output = root / "worktrees.json"
    output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
