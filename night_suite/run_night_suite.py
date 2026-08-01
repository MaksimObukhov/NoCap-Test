#!/usr/bin/env python3
"""Preflight and run the pinned NoCap overnight benchmark/diagnostic/proxy queue."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path


class InfrastructureFailure(RuntimeError):
    pass


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def run_capture(command, cwd=None, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def run_logged(command, cwd, env, stdout_path):
    stdout_path = Path(stdout_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise InfrastructureFailure(
            f"command failed with exit {return_code}: {' '.join(command)}"
        )


def full_sha(value, label):
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise InfrastructureFailure(f"{label} is not a full lowercase Git SHA: {value!r}")


def process_snapshot():
    output = run_capture(["ps", "-eo", "pid=,args="])
    needles = (
        "torchrun",
        "train_gpt2.py",
        "health_diagnostic.py",
        "analyze_exp016_a.py",
        "run_exp016_b.py",
    )
    return [line.strip() for line in output.splitlines() if any(item in line for item in needles)]


def data_inventory(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise InfrastructureFailure(f"no data files match {pattern}")
    inventory = []
    for filename in files:
        path = Path(filename).resolve()
        with path.open("rb") as handle:
            header = handle.read(1024)
        if len(header) != 1024:
            raise InfrastructureFailure(f"short data header: {path}")
        magic, version, tokens = struct.unpack_from("<iii", header)
        if magic != 20240520 or version != 1 or tokens <= 0:
            raise InfrastructureFailure(f"invalid data header: {path}")
        inventory.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "tokens": tokens,
                "header_sha256": hashlib.sha256(header).hexdigest(),
            }
        )
    return inventory


def verify_checkout(path, expected_sha, label):
    full_sha(expected_sha, f"{label}.sha")
    actual = run_capture(["git", "rev-parse", "HEAD"], cwd=path)
    if actual != expected_sha:
        raise InfrastructureFailure(
            f"{label} SHA mismatch: expected {expected_sha}, got {actual}"
        )
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path
    )
    if dirty:
        raise InfrastructureFailure(f"{label} has tracked changes:\n{dirty}")
    return actual


def wandb_probe(project, group, output_dir):
    script = """
import json, os, pathlib, wandb
root = pathlib.Path(os.environ['PROBE_DIR'])
root.mkdir(parents=True, exist_ok=True)
payload = root / 'probe.txt'
payload.write_text('nocap-night-suite-wandb-probe\\n')
run = wandb.init(project=os.environ['PROBE_PROJECT'], group=os.environ['PROBE_GROUP'], name='preflight-wandb-probe', job_type='preflight')
run.log({'preflight/value': 1})
artifact = wandb.Artifact(os.environ['PROBE_GROUP'] + '-preflight', type='nocap-preflight')
artifact.add_file(str(payload))
logged = run.log_artifact(artifact)
logged.wait()
result = {'run_url': run.url, 'artifact_id': logged.id, 'artifact_version': logged.version}
run.finish()
print(json.dumps(result))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PROBE_DIR": str(output_dir),
            "PROBE_PROJECT": project,
            "PROBE_GROUP": group,
            "WANDB_MODE": "online",
        }
    )
    output = run_capture([sys.executable, "-c", script], env=environment)
    return json.loads(output.splitlines()[-1])


def preflight(manifest, prepared, data_root, results_root, checkpoint_paths, probe_wandb):
    if manifest["auto_stop"] != "disabled":
        raise InfrastructureFailure(
            "auto_stop must remain disabled until the live Vast guide is reviewed"
        )
    if not os.environ.get("TMUX"):
        raise InfrastructureFailure(
            "TMUX is empty; attach to the instance-provided tmux session instead of creating one"
        )
    tmux_state = run_capture(["tmux", "ls"])
    processes = process_snapshot()
    if processes:
        raise InfrastructureFailure("training-related processes already exist: " + repr(processes))
    guide = Path("/etc/vast-agents-guide.md")
    if not guide.is_file():
        raise InfrastructureFailure("/etc/vast-agents-guide.md is missing")

    control = prepared["control"]
    verify_checkout(control["path"], manifest["control"]["sha"], "control")
    experiment_by_id = {item["id"]: item for item in manifest["experiments"]}
    checkout_state = {"control": control}
    for experiment_id, state in prepared["experiments"].items():
        expected = experiment_by_id[experiment_id]
        verify_checkout(state["path"], expected["sha"], experiment_id)
        checkout_state[experiment_id] = state

    train_pattern = str(Path(data_root).resolve() / "fineweb_train_*.bin")
    val_pattern = str(Path(data_root).resolve() / "fineweb_val_*.bin")
    train_inventory = data_inventory(train_pattern)
    val_inventory = data_inventory(val_pattern)
    free_bytes = shutil.disk_usage(results_root).free
    if free_bytes < 50 * 1024**3:
        raise InfrastructureFailure(
            f"less than 50 GiB free at {results_root}: {free_bytes / 1024**3:.1f} GiB"
        )

    exp016 = experiment_by_id.get("exp016")
    checkpoint_state = {}
    if exp016 and exp016.get("enabled", True):
        for label in ("proxy", "full"):
            path = Path(checkpoint_paths[label]).resolve()
            if not path.is_file():
                raise InfrastructureFailure(f"missing exp016 {label} checkpoint: {path}")
            checkpoint_state[label] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }

    gpu = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,pstate,clocks.sm,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ]
    )
    if len(gpu.splitlines()) != 1 or "4090" not in gpu:
        raise InfrastructureFailure(f"expected one RTX 4090, got: {gpu}")
    python_cuda = json.loads(
        run_capture(
            [
                sys.executable,
                "-c",
                "import json,torch; assert torch.cuda.is_available(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))",
            ]
        ).splitlines()[-1]
    )

    preflight_dir = Path(results_root) / manifest["suite_id"] / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    wandb_state = None
    if probe_wandb:
        wandb_state = wandb_probe(
            manifest["wandb_project"], manifest["suite_id"], preflight_dir / "wandb-probe"
        )
    payload = {
        "status": "pass",
        "suite_id": manifest["suite_id"],
        "tmux": tmux_state,
        "processes": processes,
        "vast_guide_sha256": hashlib.sha256(guide.read_bytes()).hexdigest(),
        "checkouts": checkout_state,
        "data": {"train": train_inventory, "validation": val_inventory},
        "checkpoints": checkpoint_state,
        "disk_free_gib": free_bytes / 1024**3,
        "gpu": gpu,
        "python_cuda": python_cuda,
        "wandb_probe": wandb_state,
        "auto_stop": manifest["auto_stop"],
        "timestamp": time.time(),
    }
    write_json(preflight_dir / "preflight.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return {"train": train_pattern, "val": val_pattern, "preflight": payload}


def stage_manifest(manifest, experiment, stage, mode, updates, worktree):
    return {
        "suite_id": manifest["suite_id"],
        "experiment": experiment["id"],
        "treatment": experiment["treatment"],
        "stage": stage,
        "mode": mode,
        "branch": experiment["branch"],
        "sha": experiment["sha"],
        "launcher": "night_suite/run_night_suite.py",
        "updates_or_token_equivalent": updates,
        "seed": 0,
        "wandb_project": manifest["wandb_project"],
        "wandb_group": manifest["suite_id"],
        "auto_stop": manifest["auto_stop"],
        "worktree": str(worktree),
    }


def stage_ready(stage_dir):
    required = (
        "manifest.json",
        "summary.json",
        "gate_decision.json",
        "stdout.log",
        "wandb_upload.json",
    )
    return all((stage_dir / name).is_file() for name in required)


def start_stage(stage_dir, payload):
    if stage_ready(stage_dir):
        return False
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise InfrastructureFailure(
            f"partial stage exists and requires inspection before restart: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "manifest.json", payload)
    return True


def parse_training_metrics(path):
    records = []
    with Path(path).open() as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "train":
                records.append(payload)
    if not records:
        raise InfrastructureFailure(f"no training metrics in {path}")
    return records


def steady_token_rate(records):
    selected = records[10:-2]
    if len(selected) < 20:
        raise InfrastructureFailure("not enough updates 11-48 for a stable benchmark")
    previous_tokens = 0
    rates = []
    for record in records:
        tokens = record["tokens_seen"]
        delta_tokens = record.get("effective_batch_tokens", tokens - previous_tokens)
        previous_tokens = tokens
        if record in selected:
            rates.append(delta_tokens / (record["step_time_ms"] / 1000.0))
    return statistics.median(rates), len(rates)


def upload_stage(control_path, manifest, stage_dir, stage_name):
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "online"
    command = [
        sys.executable,
        str(Path(control_path) / "night_suite" / "upload_stage.py"),
        "--stage-dir",
        str(stage_dir),
        "--artifact-name",
        f"{manifest['suite_id']}-{stage_name}",
        "--project",
        manifest["wandb_project"],
        "--group",
        manifest["suite_id"],
        "--run-name",
        f"{stage_name}-artifacts",
    ]
    run_logged(
        command,
        control_path,
        environment,
        stage_dir / "wandb-upload.log",
    )
    receipt = stage_dir / "wandb_upload.json"
    if read_json(receipt).get("status") != "uploaded":
        raise InfrastructureFailure(f"W&B upload receipt is invalid: {receipt}")


def training_stage(
    manifest,
    control_path,
    experiment,
    worktree,
    stage_dir,
    stage_name,
    data,
    cache_root,
    mode,
    iterations,
    extra_args,
):
    payload = stage_manifest(
        manifest, experiment, stage_name, mode, iterations, worktree
    )
    if not start_stage(stage_dir, payload):
        return read_json(stage_dir / "gate_decision.json")
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_ENABLED": "1",
            "WANDB_MODE": "online",
            "WANDB_PROJECT": manifest["wandb_project"],
            "WANDB_GROUP": manifest["suite_id"],
            "WANDB_DIR": str(stage_dir / "wandb"),
            "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / stage_name),
            "TORCH_LOGS": "recompiles",
        }
    )
    command = [
        "bash",
        "run.sh",
        "smoke" if mode == "benchmark" else mode,
        "0",
        str(stage_dir),
    ]
    if mode == "benchmark":
        command.extend(
            [
                "--run_mode",
                "custom",
                "--run_name",
                stage_name,
                "--num_iterations",
                str(iterations),
                "--warmup_iters",
                "0",
                "--warmdown_iters",
                "0",
                "--val_loss_every",
                "0",
                "--save_every",
                "0",
                "--skip_final_checkpoint",
                "--profile",
                "--profile_wait_steps",
                "46",
                "--profile_warmup_steps",
                "1",
                "--profile_active_steps",
                "1",
            ]
        )
    else:
        command.extend(["--run_name", stage_name, "--skip_final_checkpoint"])
    command.extend(
        [
            "--input_bin",
            data["train"],
            "--input_val_bin",
            data["val"],
        ]
    )
    command.extend(extra_args)
    run_logged(command, worktree, environment, stage_dir / "stdout.log")
    summary = read_json(stage_dir / "summary.json")
    if summary.get("status") != "complete":
        raise InfrastructureFailure(f"incomplete training summary: {stage_dir}")
    if summary.get("git_commit") != experiment["sha"] or summary.get("git_dirty"):
        raise InfrastructureFailure(f"training provenance mismatch: {stage_dir}")
    return summary


def benchmark_gate(stage_dir, reference_dir, minimum_speedup_pct):
    treatment_rate, treatment_count = steady_token_rate(
        parse_training_metrics(stage_dir / "metrics.jsonl")
    )
    reference_rate, reference_count = steady_token_rate(
        parse_training_metrics(reference_dir / "metrics.jsonl")
    )
    speedup_pct = 100.0 * (treatment_rate / reference_rate - 1.0)
    decision = "pass" if speedup_pct >= minimum_speedup_pct else "kill"
    evidence = {
        "decision": decision,
        "median_treatment_tokens_per_second": treatment_rate,
        "median_reference_tokens_per_second": reference_rate,
        "speedup_pct": speedup_pct,
        "minimum_speedup_pct": minimum_speedup_pct,
        "treatment_updates": treatment_count,
        "reference_updates": reference_count,
        "reference_stage": str(reference_dir),
    }
    write_json(stage_dir / "benchmark.json", evidence)
    gate = {"stage": "benchmark", "decision": decision, "evidence": evidence}
    write_json(stage_dir / "gate_decision.json", gate)
    return gate


def health_stage(manifest, control_path, experiment, worktree, stage_dir, data, cache_root):
    config = experiment["health"]
    payload = stage_manifest(
        manifest,
        experiment,
        "health",
        "diagnostic",
        config["updates"],
        worktree,
    )
    if not start_stage(stage_dir, payload):
        return read_json(stage_dir / "gate_decision.json")
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "online",
            "WANDB_DIR": str(stage_dir / "wandb"),
            "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / f"{experiment['id']}-health"),
            "TORCH_LOGS": "recompiles",
        }
    )
    command = [
        sys.executable,
        "night_suite/health_diagnostic.py",
        "--output-dir",
        str(stage_dir),
        "--input-bin",
        data["train"],
        "--run-name",
        f"{experiment['id']}-health",
        "--wandb-project",
        manifest["wandb_project"],
        "--wandb-group",
        manifest["suite_id"],
        "--log-wandb",
        "--num-updates",
        str(config["updates"]),
        "--grad-accumulation-steps",
        str(config.get("grad_accumulation_steps", 32)),
        "--lr-scale",
        str(config.get("lr_scale", 1.0)),
    ]
    command.extend(config.get("extra_args", []))
    run_logged(command, worktree, environment, stage_dir / "stdout.log")
    summary = read_json(stage_dir / "summary.json")
    if summary.get("git_commit") != experiment["sha"] or summary.get("git_dirty"):
        raise InfrastructureFailure(f"health provenance mismatch: {stage_dir}")
    gate = read_json(stage_dir / "gate_decision.json")
    upload_stage(control_path, manifest, stage_dir, f"{experiment['id']}-health")
    return gate


def proxy_gate(stage_dir, pass_loss, kill_loss):
    summary = read_json(stage_dir / "summary.json")
    loss = summary.get("final_val_loss")
    if loss is None:
        raise InfrastructureFailure(f"proxy summary has no final loss: {stage_dir}")
    if loss <= pass_loss:
        decision = "pass"
    elif loss >= kill_loss:
        decision = "kill"
    else:
        decision = "inconclusive"
    gate = {
        "stage": "proxy-seed-0",
        "decision": decision,
        "evidence": {
            "final_val_loss": loss,
            "pass_loss": pass_loss,
            "kill_loss": kill_loss,
            "tokens_seen": summary.get("tokens_seen"),
        },
    }
    write_json(stage_dir / "gate_decision.json", gate)
    return gate


def proxy_allowed(benchmark, health):
    return benchmark["decision"] != "kill" and health["decision"] != "kill"


def run_standard_experiment(
    manifest,
    control_path,
    experiment,
    worktree,
    suite_root,
    reference_dir,
    data,
    cache_root,
):
    experiment_root = suite_root / experiment["id"]
    benchmark_dir = experiment_root / "benchmark"
    benchmark_summary = training_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        benchmark_dir,
        f"{experiment['id']}-benchmark",
        data,
        cache_root,
        "benchmark",
        50,
        experiment.get("benchmark_extra_args", []),
    )
    if "decision" not in benchmark_summary:
        benchmark = benchmark_gate(
            benchmark_dir,
            reference_dir,
            experiment["minimum_speedup_pct"],
        )
        upload_stage(control_path, manifest, benchmark_dir, f"{experiment['id']}-benchmark")
    else:
        benchmark = read_json(benchmark_dir / "gate_decision.json")

    health = health_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        experiment_root / "health",
        data,
        cache_root,
    )
    if not proxy_allowed(benchmark, health):
        return {
            "experiment": experiment["id"],
            "decision": "kill",
            "benchmark": benchmark,
            "health": health,
            "proxy": "skipped",
        }

    proxy_dir = experiment_root / "proxy-seed-0"
    proxy_summary = training_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        proxy_dir,
        f"{experiment['id']}-proxy-seed-0",
        data,
        cache_root,
        "proxy",
        1788,
        experiment.get("proxy_extra_args", []),
    )
    if "decision" not in proxy_summary:
        proxy = proxy_gate(
            proxy_dir,
            experiment["proxy"]["pass_loss"],
            experiment["proxy"]["kill_loss"],
        )
        upload_stage(control_path, manifest, proxy_dir, f"{experiment['id']}-proxy-seed-0")
    else:
        proxy = read_json(proxy_dir / "gate_decision.json")
    return {
        "experiment": experiment["id"],
        "decision": proxy["decision"],
        "benchmark": benchmark,
        "health": health,
        "proxy": proxy,
    }


def external_stage(
    manifest,
    control_path,
    experiment,
    worktree,
    stage_dir,
    stage_name,
    updates,
    command,
    cache_root,
):
    payload = stage_manifest(
        manifest, experiment, stage_name, "offline", updates, worktree
    )
    if not start_stage(stage_dir, payload):
        return read_json(stage_dir / "gate_decision.json")
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "online",
            "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / stage_name),
        }
    )
    run_logged(command, worktree, environment, stage_dir / "stdout.log")
    summary = read_json(stage_dir / "summary.json")
    if (
        summary.get("source", {}).get("commit") != experiment["sha"]
        or summary.get("source", {}).get("tracked_dirty")
    ):
        raise InfrastructureFailure(f"external-stage provenance mismatch: {stage_dir}")
    decision = summary.get("decision")
    if decision not in {"pass", "kill", "inconclusive"}:
        raise InfrastructureFailure(f"invalid scientific decision: {summary}")
    gate = {"stage": stage_name, "decision": decision, "evidence": summary}
    write_json(stage_dir / "gate_decision.json", gate)
    upload_stage(control_path, manifest, stage_dir, stage_name)
    return gate


def run_exp016(
    manifest,
    control_path,
    experiment,
    worktree,
    suite_root,
    data,
    cache_root,
    checkpoints,
):
    root = suite_root / experiment["id"]
    a_dir = root / "a-causal-replay"
    a = external_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        a_dir,
        "exp016-a-causal-replay",
        16,
        [
            sys.executable,
            "analyze_exp016_a.py",
            "--checkpoint",
            f"proxy={Path(checkpoints['proxy']).resolve()}",
            "--checkpoint",
            f"full={Path(checkpoints['full']).resolve()}",
            "--input-bin",
            data["train"],
            "--output",
            str(a_dir / "summary.json"),
        ],
        cache_root,
    )
    if a["decision"] != "pass":
        return {"experiment": "exp016", "decision": a["decision"], "a": a, "b": "skipped", "proxy": "skipped"}

    b_dir = root / "b-systems"
    b = external_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        b_dir,
        "exp016-b-systems",
        150,
        [
            sys.executable,
            "run_exp016_b.py",
            "--output-dir",
            str(b_dir),
            "--input-bin",
            data["train"],
            "--input-val-bin",
            data["val"],
        ],
        cache_root,
    )
    if b["decision"] != "pass":
        return {"experiment": "exp016", "decision": b["decision"], "a": a, "b": b, "proxy": "skipped"}

    health = health_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        root / "health",
        data,
        cache_root,
    )
    if health["decision"] != "pass":
        return {"experiment": "exp016", "decision": "kill", "a": a, "b": b, "health": health, "proxy": "skipped"}
    proxy_dir = root / "proxy-seed-0"
    proxy_summary = training_stage(
        manifest,
        control_path,
        experiment,
        worktree,
        proxy_dir,
        "exp016-proxy-seed-0",
        data,
        cache_root,
        "proxy",
        1788,
        ["--descent_budgeted_attn_v"],
    )
    if "decision" not in proxy_summary:
        proxy = proxy_gate(
            proxy_dir,
            experiment["proxy"]["pass_loss"],
            experiment["proxy"]["kill_loss"],
        )
        upload_stage(control_path, manifest, proxy_dir, "exp016-proxy-seed-0")
    else:
        proxy = read_json(proxy_dir / "gate_decision.json")
    return {"experiment": "exp016", "decision": proxy["decision"], "a": a, "b": b, "health": health, "proxy": proxy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepared-state", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--proxy-checkpoint", required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--probe-wandb", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("choose exactly one of --preflight-only or --run")

    manifest = read_json(args.manifest)
    prepared = read_json(args.prepared_state)
    if prepared["suite_id"] != manifest["suite_id"]:
        raise InfrastructureFailure("prepared-state suite_id does not match manifest")
    results_root = Path(args.results_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {"proxy": args.proxy_checkpoint, "full": args.full_checkpoint}
    data = preflight(
        manifest,
        prepared,
        args.data_root,
        results_root,
        checkpoints,
        args.probe_wandb,
    )
    if args.preflight_only:
        return

    suite_root = results_root / manifest["suite_id"]
    status_path = suite_root / "suite_status.json"
    control_path = prepared["control"]["path"]
    experiments = {item["id"]: item for item in manifest["experiments"]}
    results = []
    try:
        baseline = manifest["control"]
        baseline_dir = suite_root / "references" / "baseline-benchmark"
        baseline_summary = training_stage(
            manifest,
            control_path,
            baseline,
            control_path,
            baseline_dir,
            "baseline-benchmark",
            data,
            cache_root,
            "benchmark",
            50,
            [],
        )
        if "decision" not in baseline_summary:
            gate = {"stage": "baseline-benchmark", "decision": "reference", "evidence": baseline_summary}
            write_json(baseline_dir / "gate_decision.json", gate)
            upload_stage(control_path, manifest, baseline_dir, "baseline-benchmark")

        exp012 = experiments["exp012"]
        exp012_path = prepared["experiments"]["exp012"]["path"]
        uniform_dir = suite_root / "references" / "uniform-3d-benchmark"
        uniform_summary = training_stage(
            manifest,
            control_path,
            exp012,
            exp012_path,
            uniform_dir,
            "uniform-3d-benchmark",
            data,
            cache_root,
            "benchmark",
            50,
            ["--batch_ramp_start_accumulation_steps", "32", "--batch_ramp_fraction", "0.5"],
        )
        if "decision" not in uniform_summary:
            gate = benchmark_gate(uniform_dir, baseline_dir, -100.0)
            gate["decision"] = "reference"
            write_json(uniform_dir / "gate_decision.json", gate)
            upload_stage(control_path, manifest, uniform_dir, "uniform-3d-benchmark")

        for experiment_id in ("exp012", "exp013", "exp014", "exp015"):
            experiment = experiments[experiment_id]
            if not experiment.get("enabled", True):
                continue
            reference = baseline_dir if experiment["benchmark_reference"] == "baseline" else uniform_dir
            result = run_standard_experiment(
                manifest,
                control_path,
                experiment,
                prepared["experiments"][experiment_id]["path"],
                suite_root,
                reference,
                data,
                cache_root,
            )
            results.append(result)
            write_json(status_path, {"status": "running", "results": results})

        if experiments["exp016"].get("enabled", True):
            results.append(
                run_exp016(
                    manifest,
                    control_path,
                    experiments["exp016"],
                    prepared["experiments"]["exp016"]["path"],
                    suite_root,
                    data,
                    cache_root,
                    checkpoints,
                )
            )
        payload = {"status": "complete", "results": results, "auto_stop": manifest["auto_stop"]}
        write_json(status_path, payload)
        os.sync()
        print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            status_path,
            {
                "status": "infrastructure-failure",
                "error": repr(error),
                "results": results,
                "instance_action": "left-running-for-inspection",
            },
        )
        raise


if __name__ == "__main__":
    main()
