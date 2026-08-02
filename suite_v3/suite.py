"""Orchestrator for the v3 proxy suite.

Design goals, in priority order:

1. Start it, disconnect, and trust it. A failure inside one experiment is
   recorded and the suite moves to the next one. Only preflight, disk or GPU
   failures stop everything.
2. Never spend GPU-hours on a stage whose predecessor already said no.
3. Leave enough on local disk to reconstruct every decision without W&B.
4. Contain no full run, anywhere.

State lives in two places: an append-only ledger (`suite_events.jsonl`) that
is never rewritten, and `suite_status.json`, a derived snapshot that is
replaced atomically. The v2 suite kept only the snapshot, so its first
infrastructure failure vanished from the record on resume.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

from suite_v3 import gates

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


class InfrastructureFailure(RuntimeError):
    """Stops the whole suite. Reserved for host-level problems only."""


# ---------------------------------------------------------------------------
# Small IO helpers


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sha256_file(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Ledger


class Ledger:
    def __init__(self, root):
        self.root = root
        self.events_path = os.path.join(root, "suite_events.jsonl")
        self.status_path = os.path.join(root, "suite_status.json")
        os.makedirs(root, exist_ok=True)

    def append(self, kind, **fields):
        record = {"time": time.time(), "kind": kind, **fields}
        with open(self.events_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary = " ".join(f"{k}={v}" for k, v in fields.items() if k != "detail")
        print(f"[suite] {kind} {summary}", flush=True)
        return record

    def snapshot(self, payload):
        write_json_atomic(self.status_path, payload)


# ---------------------------------------------------------------------------
# Process execution


def run_logged(command, cwd, stdout_path, env=None):
    """Run a child process, streaming combined output to a file.

    Returns the exit code rather than raising, because a non-zero exit from a
    training stage is frequently a result (a tripwire fired) and not an error.
    """
    os.makedirs(os.path.dirname(os.path.abspath(stdout_path)), exist_ok=True)
    merged = dict(os.environ)
    if env:
        merged.update(env)
    with open(stdout_path, "a") as handle:
        handle.write(f"$ {' '.join(shlex.quote(c) for c in command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=merged,
        )
        return process.wait()


def git_sha(worktree):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_is_clean(worktree):
    """Full porcelain, including untracked files.

    The v2 check passed --untracked-files=no, so an untracked module or
    config sitting in a worktree could influence execution without being
    recorded anywhere.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return (not lines), lines


# ---------------------------------------------------------------------------
# Preflight


def preflight(manifest, worktrees, data_glob, results_root, minimum_free_gib):
    report = {"status": "pass", "checks": {}, "time": time.time()}

    for label, path in sorted(worktrees.items()):
        if not os.path.isdir(path):
            raise InfrastructureFailure(f"worktree missing for {label}: {path}")
        clean, dirt = git_is_clean(path)
        report["checks"][f"worktree::{label}"] = {
            "path": path,
            "sha": git_sha(path),
            "clean": clean,
            "dirt": dirt[:20],
        }
        if not clean:
            raise InfrastructureFailure(
                f"worktree {label} at {path} is not clean: {dirt[:5]}"
            )

    import glob as globmodule

    shards = sorted(globmodule.glob(data_glob))
    if not shards:
        raise InfrastructureFailure(f"no data shards matched {data_glob}")
    report["checks"]["data"] = {
        "glob": data_glob,
        "shards": len(shards),
        "total_bytes": sum(os.path.getsize(s) for s in shards),
        "first": shards[0],
        "last": shards[-1],
    }

    statvfs = os.statvfs(results_root if os.path.isdir(results_root) else "/")
    free_gib = statvfs.f_bavail * statvfs.f_frsize / (1024**3)
    report["checks"]["disk"] = {
        "free_gib": free_gib,
        "minimum_free_gib": minimum_free_gib,
    }
    if free_gib < minimum_free_gib:
        raise InfrastructureFailure(
            f"only {free_gib:.1f} GiB free, need {minimum_free_gib} GiB"
        )

    try:
        import torch

        report["checks"]["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
        if not torch.cuda.is_available():
            raise InfrastructureFailure("no CUDA device visible")
    except ImportError as error:
        raise InfrastructureFailure(f"torch unavailable: {error}") from error

    return report


# ---------------------------------------------------------------------------
# Stage identity and reuse


def stage_identity(manifest_hash, experiment, stage_name, command, extra=None):
    return {
        "manifest_hash": manifest_hash,
        "experiment": experiment["id"],
        "sha": experiment["sha"],
        "stage": stage_name,
        "command": command,
        "extra": extra or {},
    }


def stage_is_reusable(stage_dir, identity):
    """Reuse a completed stage only when it is provably the same stage.

    The v2 suite accepted any directory containing five expected filenames,
    which is how a stage measured at one control SHA was reused against a
    manifest that had moved to another.
    """
    identity_path = os.path.join(stage_dir, "stage.json")
    gate_path = os.path.join(stage_dir, "gate.json")
    if not (os.path.exists(identity_path) and os.path.exists(gate_path)):
        return False, "stage not complete"
    recorded = read_json(identity_path)
    if stable_hash(recorded.get("identity", {})) != stable_hash(identity):
        return False, "stage identity differs from current manifest"
    return True, "identity matches"


# ---------------------------------------------------------------------------
# Command builders


def wandb_args(manifest, experiment_id, stage_name, log_wandb):
    if not log_wandb:
        return []
    return [
        "--log_wandb",
        "--wandb_project",
        manifest["wandb_project"],
        # group is the experiment ID so a single experiment's stages sit
        # together; the suite ID lives in the run name.
        "--wandb_group",
        experiment_id,
        "--run_name",
        f"{manifest['suite_id']}-{experiment_id}-{stage_name}",
    ]


def parse_arg_pairs(tokens):
    """Split a flat argument list into (flag, value) pairs.

    Raises on a stray positional. That is not pedantry: an earlier version of
    the stack merge appended a value without its flag, producing
    `--grad_clip 10.0 3.0`, which argparse would have rejected only after the
    process had already been launched on a paid instance.
    """
    pairs = []
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--"):
            raise ValueError(f"stray positional argument {flag!r} in {tokens}")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            pairs.append((flag, tokens[index + 1]))
            index += 2
        else:
            pairs.append((flag, None))
            index += 1
    return pairs


def merge_args(*groups):
    """Merge argument lists so each flag appears exactly once, later winning.

    argparse would take the last value anyway, but a recorded command that
    lists --warmdown_iters twice with different values is not a record anyone
    should have to reason about when auditing what actually ran.
    """
    merged = {}
    for group in groups:
        for flag, value in parse_arg_pairs(group or []):
            merged[flag] = value
    flattened = []
    for flag, value in merged.items():
        flattened.append(flag)
        if value is not None:
            flattened.append(value)
    return flattened


def training_command(
    manifest, experiment_id, stage_name, mode, extra_args, output_dir, options
):
    tail = ["--input_bin", options.train_glob]
    if mode == "proxy":
        tail += ["--input_val_bin", options.val_glob]
    tail += ["--output_dir", output_dir]
    tail += wandb_args(manifest, experiment_id, stage_name, options.log_wandb)
    return [options.python, "train_gpt2.py"] + merge_args(
        manifest["common_args"][mode], extra_args, tail
    )


# ---------------------------------------------------------------------------
# Stage implementations


def run_accounting(manifest, experiment, worktree, stage_dir, options, ledger):
    spec = experiment.get("accounting", {})
    report_path = os.path.join(stage_dir, "accounting.json")
    proxy_extra = experiment.get("proxy", {}).get("extra_args", [])

    command = [options.python, os.path.join("suite_v3", "accounting.py")]
    if "--mlp_ratio" in proxy_extra:
        command += ["--mlp_ratio", proxy_extra[proxy_extra.index("--mlp_ratio") + 1]]
    elif experiment.get("proxy", {}).get("use_exp012_stack_ramp"):
        command += ["--mlp_ratio", "3.0"]
    if "--no_wd_tied_embedding" in proxy_extra:
        command += ["--no_wd_tied_embedding"]
    if spec.get("reference_equivalence"):
        command += ["--reference_equivalence", spec["reference_equivalence"]]
    command += ["--output", report_path]

    code = run_logged(command, worktree, os.path.join(stage_dir, "stdout.log"))
    if code != 0 or not os.path.exists(report_path):
        return gates._decision(
            "accounting",
            gates.INVALID,
            {"reason": f"accounting process exited {code}"},
        ), command

    report = read_json(report_path)
    decision = gates.accounting_gate(report, spec.get("expectations", {}))
    return decision, command


def run_benchmark(
    manifest, experiment, worktree, base_worktree, stage_dir, options, ledger
):
    """Bracketed control / treatment / control on one host."""
    spec = experiment["benchmark"]
    control_args = ["--mlp_ratio", str(spec["control_mlp_ratio"])]
    treatment_args = list(spec["treatment_args"])

    plan = [
        ("control-before", base_worktree, control_args),
        ("treatment", worktree, treatment_args),
        ("control-after", base_worktree, control_args),
    ]
    measurements = {}
    commands = []
    for label, tree, extra in plan:
        sub_dir = os.path.join(stage_dir, label)
        command = training_command(
            manifest,
            experiment["id"],
            f"benchmark-{label}",
            "benchmark",
            extra,
            sub_dir,
            options,
        )
        commands.append(command)
        code = run_logged(command, tree, os.path.join(sub_dir, "stdout.log"))
        metrics_path = os.path.join(sub_dir, "metrics.jsonl")
        if code != 0 or not os.path.exists(metrics_path):
            return (
                gates._decision(
                    "benchmark",
                    gates.INVALID,
                    {"reason": f"{label} exited {code}"},
                ),
                commands,
            )
        measurements[label] = gates.steady_step_stats(read_jsonl(metrics_path))
        ledger.append(
            "benchmark_measurement",
            experiment=experiment["id"],
            label=label,
            median_step_time_ms=measurements[label]["median_step_time_ms"],
        )

    decision = gates.benchmark_gate(
        measurements["treatment"],
        measurements["control-before"],
        measurements["control-after"],
        warning_regression_pct=spec.get("warning_regression_pct", 1.0),
        kill_regression_pct=spec.get("kill_regression_pct", 2.0),
        drift_tolerance_pct=spec.get("drift_tolerance_pct", 1.0),
    )
    return decision, commands


def run_proxy(manifest, experiment, worktree, stage_dir, options, ledger):
    spec = experiment["proxy"]
    # The exp012 stack supplies the ramp and its default 3.0 ratio; the
    # experiment's own flags come second and win, so exp019 keeps 2.0.
    if spec.get("use_exp012_stack_ramp"):
        extra = merge_args(
            manifest["common_args"]["exp012_stack"], spec.get("extra_args", [])
        )
    else:
        extra = merge_args(spec.get("extra_args", []))

    envelope = manifest["envelopes"].get(spec.get("envelope_reference"))
    if envelope:
        envelope_path = os.path.join(stage_dir, "val_envelope.json")
        write_json_atomic(envelope_path, envelope)
        extra += ["--val_envelope", f"@{envelope_path}"]

    command = training_command(
        manifest, experiment["id"], "proxy-seed-0", "proxy", extra, stage_dir, options
    )
    code = run_logged(command, worktree, os.path.join(stage_dir, "stdout.log"))

    summary_path = os.path.join(stage_dir, "summary.json")
    if not os.path.exists(summary_path):
        return (
            gates._decision(
                "proxy",
                gates.INVALID,
                {"reason": f"proxy exited {code} with no summary.json"},
            ),
            [command],
        )

    summary = read_json(summary_path)
    summary["process_exit_code"] = code
    reference_key = spec.get("reference")
    reference = manifest["references"].get(reference_key, {})
    decision = gates.proxy_gate(
        summary,
        spec["pass_loss"],
        spec["kill_loss"],
        reference_key,
        reference.get("final_val_loss"),
    )

    # Post-hoc, per-phase gradient reporting. Never a gate.
    records = read_jsonl(os.path.join(stage_dir, "metrics.jsonl"))
    decision["evidence"]["spike_clusters"] = {
        phase: gates.spike_clusters(records, phase)
        for phase in ("warmup", "steady", "warmdown")
    }
    return decision, [command]


def run_checkpoint_validation(manifest, experiment, stage_dir, options, ledger):
    """Strict canonicity check for exp016's replay checkpoints.

    Done here rather than trusting the experiment branch's own validator,
    which checks a few args and a commit prefix but not finality, step,
    tokens, or content hash.
    """
    spec = experiment["checkpoint_validation"]
    findings = {}
    failures = []
    for key in spec["require"]:
        declared = manifest["canonical_checkpoints"][key]
        path = declared["path"]
        if not os.path.isabs(path):
            path = os.path.join(options.checkpoint_root, path)
        entry = {"path": path}
        if not os.path.exists(path):
            entry["missing"] = True
            failures.extend(
                f"{key}: {problem}"
                for problem in gates.checkpoint_field_failures(declared, entry)
            )
            findings[key] = entry
            continue
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = sha256_file(path)
        if entry["sha256"] == declared["sha256"]:
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_args = checkpoint.get("args", {})
            entry["next_step"] = checkpoint.get("next_step")
            entry["tokens_seen"] = checkpoint.get("tokens_seen")
            entry["run_mode"] = checkpoint_args.get("run_mode")
            entry["seed"] = checkpoint_args.get("seed")
            entry["state_keys"] = sorted(checkpoint.keys())
            del checkpoint

        problems = gates.checkpoint_field_failures(declared, entry)
        failures.extend(f"{key}: {problem}" for problem in problems)
        findings[key] = entry

    decision = gates.PASS if not failures else gates.INVALID
    result = gates._decision(
        "checkpoint_validation",
        decision,
        {"findings": findings, "failures": failures},
    )
    write_json_atomic(os.path.join(stage_dir, "checkpoints.json"), findings)
    return result, []


def run_causal_replay(manifest, experiment, worktree, stage_dir, options, ledger):
    spec = experiment["causal_replay"]
    command = [options.python, spec["script"]]
    for key in experiment["checkpoint_validation"]["require"]:
        declared = manifest["canonical_checkpoints"][key]
        path = declared["path"]
        if not os.path.isabs(path):
            path = os.path.join(options.checkpoint_root, path)
        command += ["--checkpoint", f"{declared['expect_run_mode']}={path}"]
    command += ["--output-dir", stage_dir]
    code = run_logged(command, worktree, os.path.join(stage_dir, "stdout.log"))

    verdict_path = os.path.join(stage_dir, "gate_decision.json")
    if code != 0 or not os.path.exists(verdict_path):
        return (
            gates._decision(
                "causal_replay",
                gates.KILL if code != 0 else gates.INVALID,
                {"reason": f"stage A exited {code}"},
            ),
            [command],
        )
    verdict = read_json(verdict_path)
    decision = verdict.get("decision", gates.INVALID)
    if decision not in {gates.PASS, gates.KILL, gates.INVALID}:
        decision = gates.INVALID
    return (
        gates._decision("causal_replay", decision, verdict),
        [command],
    )


# ---------------------------------------------------------------------------
# Experiment driver


STAGE_RUNNERS = {
    "accounting",
    "benchmark",
    "proxy",
    "checkpoint_validation",
    "causal_replay",
    "fp8_microbench",
}


def run_experiment(manifest, manifest_hash, experiment, options, ledger, worktrees):
    experiment_root = os.path.join(options.results_root, experiment["id"])
    os.makedirs(experiment_root, exist_ok=True)
    worktree = worktrees[experiment["id"]]
    base_worktree = worktrees["control"]

    outcome = {"experiment": experiment["id"], "stages": {}, "decision": None}

    for stage_name in experiment["stages"]:
        if stage_name not in STAGE_RUNNERS:
            raise InfrastructureFailure(f"unknown stage {stage_name!r}")
        stage_dir = os.path.join(experiment_root, stage_name)
        os.makedirs(stage_dir, exist_ok=True)

        identity = stage_identity(
            manifest_hash, experiment, stage_name, [], {"stage": stage_name}
        )
        reusable, why = stage_is_reusable(stage_dir, identity)
        if reusable and not options.force:
            decision = read_json(os.path.join(stage_dir, "gate.json"))
            ledger.append(
                "stage_reused",
                experiment=experiment["id"],
                stage=stage_name,
                decision=decision["decision"],
            )
            outcome["stages"][stage_name] = decision
            if not gates.suite_should_continue(decision["decision"]):
                outcome["decision"] = decision["decision"]
                return outcome
            continue

        ledger.append("stage_start", experiment=experiment["id"], stage=stage_name)
        started = time.time()
        try:
            if stage_name == "accounting":
                decision, commands = run_accounting(
                    manifest, experiment, worktree, stage_dir, options, ledger
                )
            elif stage_name == "benchmark":
                decision, commands = run_benchmark(
                    manifest,
                    experiment,
                    worktree,
                    base_worktree,
                    stage_dir,
                    options,
                    ledger,
                )
            elif stage_name == "proxy":
                decision, commands = run_proxy(
                    manifest, experiment, worktree, stage_dir, options, ledger
                )
            elif stage_name == "checkpoint_validation":
                decision, commands = run_checkpoint_validation(
                    manifest, experiment, stage_dir, options, ledger
                )
            elif stage_name == "causal_replay":
                decision, commands = run_causal_replay(
                    manifest, experiment, worktree, stage_dir, options, ledger
                )
            elif stage_name == "fp8_microbench":
                decision, commands = run_fp8_microbench(
                    manifest, experiment, worktree, stage_dir, options, ledger
                )
        except InfrastructureFailure:
            raise
        except Exception as error:  # noqa: BLE001
            # One experiment blowing up must not take the night with it.
            decision = gates._decision(
                stage_name,
                gates.INVALID,
                {"reason": f"{type(error).__name__}: {error}"},
            )
            commands = []

        decision["evidence"]["elapsed_seconds"] = time.time() - started
        identity["command"] = commands
        write_json_atomic(
            os.path.join(stage_dir, "stage.json"),
            {
                "identity": stage_identity(
                    manifest_hash, experiment, stage_name, [], {"stage": stage_name}
                ),
                "commands": commands,
                "experiment": experiment,
                "started": started,
                "finished": time.time(),
            },
        )
        write_json_atomic(os.path.join(stage_dir, "gate.json"), decision)
        outcome["stages"][stage_name] = decision
        ledger.append(
            "stage_finish",
            experiment=experiment["id"],
            stage=stage_name,
            decision=decision["decision"],
            seconds=round(decision["evidence"]["elapsed_seconds"], 1),
        )

        if options.log_wandb:
            upload_stage_artifacts(manifest, experiment, stage_name, stage_dir, ledger)

        if not gates.suite_should_continue(decision["decision"]):
            outcome["decision"] = decision["decision"]
            ledger.append(
                "experiment_stopped",
                experiment=experiment["id"],
                stage=stage_name,
                decision=decision["decision"],
            )
            return outcome

    terminal = experiment["stages"][-1]
    outcome["decision"] = outcome["stages"][terminal]["decision"]
    return outcome


def run_fp8_microbench(manifest, experiment, worktree, stage_dir, options, ledger):
    spec = experiment["fp8_microbench"]
    report_path = os.path.join(stage_dir, "fp8_microbench.json")
    command = [
        options.python,
        spec["script"],
        "--tokens_per_microbatch",
        str(spec["tokens_per_microbatch"]),
        "--n_embd",
        str(spec["n_embd"]),
        "--vocab_size",
        str(spec["vocab_size"]),
        "--padded_vocab_size",
        str(spec["padded_vocab_size"]),
        "--microbatches_per_step",
        str(spec["microbatches_per_step"]),
        "--reference_step_time_ms",
        str(spec["reference_step_time_ms"]),
        "--minimum_projected_full_step_gain_pct",
        str(spec["minimum_projected_full_step_gain_pct"]),
        "--output",
        report_path,
    ]
    code = run_logged(command, worktree, os.path.join(stage_dir, "stdout.log"))
    if not os.path.exists(report_path):
        return (
            gates._decision(
                "fp8_microbench",
                gates.INVALID,
                {"reason": f"microbenchmark exited {code} with no report"},
            ),
            [command],
        )
    report = read_json(report_path)
    return (
        gates._decision(
            "fp8_microbench", report.get("decision", gates.INVALID), report
        ),
        [command],
    )


def upload_stage_artifacts(manifest, experiment, stage_name, stage_dir, ledger):
    from suite_v3 import upload

    try:
        receipt = upload.upload_stage(
            project=manifest["wandb_project"],
            group=experiment["id"],
            run_name=f"{manifest['suite_id']}-{experiment['id']}-{stage_name}",
            stage_dir=stage_dir,
        )
        ledger.append(
            "upload",
            experiment=experiment["id"],
            stage=stage_name,
            status=receipt.get("status"),
        )
    except Exception as error:  # noqa: BLE001
        # An upload failure must never invalidate finished training. The
        # local artifacts are the source of truth; the receipt records the
        # gap so the upload can be retried later without recomputing.
        ledger.append(
            "upload_failed",
            experiment=experiment["id"],
            stage=stage_name,
            detail=f"{type(error).__name__}: {error}",
        )


# ---------------------------------------------------------------------------
# Entry point


def load_manifest(path):
    manifest = read_json(path)
    for experiment in manifest["experiments"]:
        for stage in experiment["stages"]:
            if stage not in STAGE_RUNNERS:
                raise InfrastructureFailure(
                    f"{experiment['id']} declares unknown stage {stage!r}"
                )
    assert_no_full_run(manifest)
    return manifest


def assert_no_full_run(manifest):
    """Refuse to start if anything here would launch a full run.

    Deliberately precise rather than a substring scan. The manifest legitimately
    mentions "full" when declaring exp000's full-seed-0 checkpoint as a
    read-only replay input for exp016; reading a full run's checkpoint is not
    launching one. What must never appear is `--run_mode full` in an argument
    list, or a stage named for one.
    """
    offenders = []

    def scan(args, where):
        for index, token in enumerate(args or []):
            if (
                token == "--run_mode"
                and index + 1 < len(args)
                and args[index + 1] == "full"
            ):
                offenders.append(f"{where}: --run_mode full")

    for mode, args in (manifest.get("common_args") or {}).items():
        if mode == "full":
            offenders.append("common_args.full exists")
        scan(args, f"common_args.{mode}")

    for experiment in manifest["experiments"]:
        for stage in experiment["stages"]:
            if "full" in stage.replace("-", "_").split("_"):
                offenders.append(f"{experiment['id']}.stages.{stage}")
        for key in ("proxy", "benchmark", "accounting", "causal_replay"):
            spec = experiment.get(key) or {}
            if not isinstance(spec, dict):
                continue
            for field in ("extra_args", "treatment_args", "control_args"):
                scan(spec.get(field), f"{experiment['id']}.{key}.{field}")

    if offenders:
        raise InfrastructureFailure(
            "manifest would launch a full run; this suite must not contain "
            f"one: {offenders}"
        )


def materialize(manifest, worktrees):
    """Freeze the manifest against the SHAs actually checked out."""
    materialized = json.loads(json.dumps(manifest))
    materialized["control"]["sha"] = git_sha(worktrees["control"])
    for experiment in materialized["experiments"]:
        experiment["sha"] = git_sha(worktrees[experiment["id"]])
    materialized["materialized_at"] = time.time()
    return materialized


def main():
    parser = argparse.ArgumentParser(description="NoCap proxy suite v3")
    parser.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--base-worktree", required=True)
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--checkpoint-root", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    parser.add_argument("--only", default="", help="comma-separated experiment IDs")
    parser.add_argument("--force", action="store_true", help="ignore stage reuse")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument(
        "--run-paid",
        action="store_true",
        help="actually execute; without it the suite plans and exits",
    )
    options = parser.parse_args()

    manifest = load_manifest(options.manifest)
    experiments = sorted(
        (e for e in manifest["experiments"] if e.get("enabled", True)),
        key=lambda e: e["order"],
    )
    if options.only:
        wanted = {x.strip() for x in options.only.split(",") if x.strip()}
        experiments = [e for e in experiments if e["id"] in wanted]

    worktrees = {"control": options.base_worktree}
    for experiment in experiments:
        if experiment["branch"] == manifest["control"]["branch"]:
            worktrees[experiment["id"]] = options.base_worktree
        else:
            worktrees[experiment["id"]] = os.path.join(
                options.worktree_root, experiment["id"]
            )

    if not options.run_paid:
        print("DRY RUN -- no compute will be started.\n")
        print(f"suite_id            : {manifest['suite_id']}")
        print(f"results root        : {options.results_root}")
        print(f"experiments enabled : {len(experiments)}")
        total = 0
        for experiment in experiments:
            total += experiment.get("expected_minutes", 0)
            print(
                f"  {experiment['order']}. {experiment['id']:<8} "
                f"stages={','.join(experiment['stages']):<45} "
                f"~{experiment.get('expected_minutes', 0)} min"
            )
            print(f"       worktree: {worktrees[experiment['id']]}")
            print(f"       {experiment['treatment']}")
        print(f"\nworst-case GPU time : {total} min ({total / 60:.1f} h)")
        print("\nNo full stage is present in this manifest.")
        print("Re-run with --run-paid to execute.")
        return 0

    ledger = Ledger(options.results_root)
    lock_path = os.path.join(options.results_root, "suite.lock")
    if os.path.exists(lock_path):
        existing = read_json(lock_path)
        if _pid_alive(existing.get("pid")):
            raise InfrastructureFailure(
                f"another suite is running as pid {existing.get('pid')}"
            )
        ledger.append("stale_lock_cleared", pid=existing.get("pid"))
    write_json_atomic(lock_path, {"pid": os.getpid(), "started": time.time()})

    try:
        report = preflight(
            manifest,
            worktrees,
            options.train_glob,
            options.results_root,
            options.minimum_free_gib,
        )
        attempt = len(
            [
                name
                for name in os.listdir(options.results_root)
                if name.startswith("preflight-attempt-")
            ]
        ) + 1
        # Append-only: every attempt keeps its own immutable record instead
        # of overwriting one preflight.json, which is how the v2 bundle lost
        # the state it actually started from.
        write_json_atomic(
            os.path.join(options.results_root, f"preflight-attempt-{attempt}.json"),
            report,
        )
        ledger.append("preflight", attempt=attempt, status="pass")

        materialized = materialize(manifest, worktrees)
        manifest_hash = stable_hash(materialized)
        write_json_atomic(
            os.path.join(options.results_root, "manifest.runtime.json"),
            {"manifest": materialized, "manifest_hash": manifest_hash},
        )
        ledger.append("manifest", manifest_hash=manifest_hash[:16])

        outcomes = []
        for experiment in experiments:
            materialized_experiment = next(
                e for e in materialized["experiments"] if e["id"] == experiment["id"]
            )
            ledger.append("experiment_start", experiment=experiment["id"])
            outcome = run_experiment(
                materialized,
                manifest_hash,
                materialized_experiment,
                options,
                ledger,
                worktrees,
            )
            outcomes.append(outcome)
            ledger.append(
                "experiment_finish",
                experiment=experiment["id"],
                decision=outcome["decision"],
            )
            ledger.snapshot(
                {
                    "suite_id": manifest["suite_id"],
                    "manifest_hash": manifest_hash,
                    "status": "running",
                    "results": outcomes,
                }
            )

        ledger.snapshot(
            {
                "suite_id": manifest["suite_id"],
                "manifest_hash": manifest_hash,
                "status": "complete",
                "results": outcomes,
            }
        )
        ledger.append("suite_complete", experiments=len(outcomes))
        _print_final_table(outcomes)
        return 0
    except InfrastructureFailure as error:
        ledger.append("infrastructure_failure", detail=str(error))
        ledger.snapshot(
            {
                "suite_id": manifest["suite_id"],
                "status": "infrastructure_failure",
                "detail": str(error),
            }
        )
        print(f"INFRASTRUCTURE FAILURE: {error}", file=sys.stderr)
        return 2
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _print_final_table(outcomes):
    print("\n" + "=" * 78)
    print("PROXY SUITE V3 -- FINAL")
    print("=" * 78)
    print(f"{'experiment':<10} {'decision':<14} stages")
    for outcome in outcomes:
        stages = ", ".join(
            f"{name}:{value['decision']}" for name, value in outcome["stages"].items()
        )
        print(f"{outcome['experiment']:<10} {str(outcome['decision']):<14} {stages}")
    print("=" * 78)
    for outcome in outcomes:
        proxy = outcome["stages"].get("proxy")
        if proxy and "final_val_loss" in proxy["evidence"]:
            evidence = proxy["evidence"]
            print(
                f"{outcome['experiment']}: final val "
                f"{evidence.get('final_val_loss')} vs "
                f"{evidence.get('reference_label')} "
                f"{evidence.get('reference_loss')} "
                f"(delta {evidence.get('delta_vs_reference')})"
            )


if __name__ == "__main__":
    sys.exit(main())
