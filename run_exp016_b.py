#!/usr/bin/env python3
"""exp016-B: correctness plus control/treatment/control systems gate."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def source_metadata():
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tracked_dirty": subprocess.call(["git", "diff", "--quiet"]) != 0
        or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0,
    }


def run_streamed(command, environment, stdout_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=False)
    with stdout_path.open("w") as handle:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"benchmark subprocess failed with exit {return_code}")


def train_records(path):
    records = []
    with Path(path).open() as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "train":
                records.append(payload)
    selected = [record for record in records if 11 <= record["step"] <= 48]
    if len(selected) != 38:
        raise RuntimeError(f"expected 38 timed records, got {len(selected)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--input-val-bin", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_metadata()
    if source["tracked_dirty"]:
        raise RuntimeError("tracked worktree changes invalidate exp016-B")

    subprocess.check_call(
        [sys.executable, "-m", "unittest", "test_exp016_descent_budgeted.py", "-v"]
    )
    cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve()
    base_environment = os.environ.copy()
    base_environment.update({"WANDB_ENABLED": "0", "TORCH_LOGS": "recompiles"})
    stages = (
        ("control-before", []),
        ("treatment", ["--descent_budgeted_attn_v"]),
        ("control-after", []),
    )
    for label, treatment_args in stages:
        run_dir = output_dir / label
        environment = base_environment.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / label)
        command = [
            "bash",
            "run.sh",
            "smoke",
            "0",
            str(run_dir),
            "--run_mode",
            "custom",
            "--run_name",
            f"exp016-b-{label}",
            "--num_iterations",
            "50",
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
            "48",
            "--profile_warmup_steps",
            "1",
            "--profile_active_steps",
            "1",
            "--input_bin",
            args.input_bin,
            "--input_val_bin",
            args.input_val_bin,
            *treatment_args,
        ]
        run_streamed(command, environment, run_dir / "stdout.log")

    records = {
        label: train_records(output_dir / label / "metrics.jsonl")
        for label, _args in stages
    }
    medians = {
        label: statistics.median(record["step_time_ms"] for record in values)
        for label, values in records.items()
    }
    control_median = statistics.median(
        [medians["control-before"], medians["control-after"]]
    )
    overhead_pct = 100.0 * (medians["treatment"] / control_median - 1.0)
    control_drift_pct = 100.0 * abs(
        medians["control-after"] / medians["control-before"] - 1.0
    )
    treatment_records = records["treatment"]
    diagnostics_present = all(
        "spectral_descent_retention_median" in record for record in treatment_records
    )
    minimum_retention = min(
        (record.get("spectral_descent_retention_median", 0.0) for record in treatment_records),
        default=0.0,
    )
    if control_drift_pct > 2.0:
        decision = "invalid"
    else:
        decision = (
            "pass"
            if overhead_pct <= 1.0
            and diagnostics_present
            and minimum_retention >= 0.99 - 1e-6
            else "kill"
        )
    payload = {
        "experiment": "exp016-B",
        "decision": decision,
        "source": source,
        "unit_tests": "pass",
        "median_step_time_ms": medians,
        "control_median_step_time_ms": control_median,
        "treatment_overhead_pct": overhead_pct,
        "projected_proxy_optimizer_overhead_pct": overhead_pct,
        "control_drift_pct": control_drift_pct,
        "diagnostics_present": diagnostics_present,
        "minimum_median_descent_retention": minimum_retention,
        "checks": {
            "treatment_overhead_le_1pct": overhead_pct <= 1.0,
            "projected_overhead_le_1pct": overhead_pct <= 1.0,
            "control_drift_le_2pct": control_drift_pct <= 2.0,
            "descent_budget_preserved": minimum_retention >= 0.99 - 1e-6,
        },
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
