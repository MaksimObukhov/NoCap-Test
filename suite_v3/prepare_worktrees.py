"""Materialize one detached worktree per experiment branch.

Detached on purpose: nothing in the suite ever switches a branch while a run
is in flight, and a detached HEAD makes an accidental commit or checkout
during the night impossible to confuse with a real branch state.
"""

import argparse
import json
import os
import subprocess
import sys


def git(args, cwd=None, check=True):
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def resolve(repo, branch):
    for candidate in (branch, f"origin/{branch}"):
        sha = git(["rev-parse", "--verify", "--quiet", candidate], cwd=repo, check=False)
        if sha:
            return sha, candidate
    raise RuntimeError(f"cannot resolve branch {branch!r}")


def existing_worktrees(repo):
    listing = git(["worktree", "list", "--porcelain"], cwd=repo)
    trees = {}
    path = None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            path = line.split(" ", 1)[1]
        elif line.startswith("HEAD ") and path:
            trees[os.path.realpath(path)] = line.split(" ", 1)[1]
    return trees


def prepare(repo, manifest_path, worktree_root, base_worktree):
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    os.makedirs(worktree_root, exist_ok=True)
    prepared = {}
    trees = existing_worktrees(repo)

    targets = [("control", manifest["control"]["branch"], base_worktree)]
    for experiment in manifest["experiments"]:
        if not experiment.get("enabled", True):
            continue
        if experiment["branch"] == manifest["control"]["branch"]:
            # Runs directly out of the shared base; nothing to create.
            prepared[experiment["id"]] = {
                "path": base_worktree,
                "shared_with": "control",
            }
            continue
        targets.append(
            (
                experiment["id"],
                experiment["branch"],
                os.path.join(worktree_root, experiment["id"]),
            )
        )

    for label, branch, path in targets:
        sha, source = resolve(repo, branch)
        real = os.path.realpath(path)
        if real in trees:
            current = trees[real]
            if current != sha:
                raise RuntimeError(
                    f"{label}: existing worktree at {path} is at {current[:12]}, "
                    f"manifest branch {branch} is at {sha[:12]}. Remove it first."
                )
        else:
            git(["worktree", "add", "--detach", path, sha], cwd=repo)
        status = git(["status", "--porcelain"], cwd=path)
        if status.strip():
            raise RuntimeError(f"{label}: worktree {path} is not clean:\n{status}")
        prepared[label] = {"path": path, "branch": branch, "sha": sha, "from": source}
        print(f"{label:<10} {branch:<34} {sha[:12]}  {path}")

    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--base-worktree", required=True)
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()

    prepared = prepare(
        arguments.repo,
        arguments.manifest,
        arguments.worktree_root,
        arguments.base_worktree,
    )
    if arguments.output:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
        with open(arguments.output, "w") as handle:
            json.dump(prepared, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
