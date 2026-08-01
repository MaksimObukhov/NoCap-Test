#!/usr/bin/env python3
"""Summarize the exp011-B same-host control/treatment benchmark."""

import argparse
import json
import math
import statistics
from pathlib import Path


def load_records(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def median_absolute_deviation(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def run_metrics(run_dir):
    records = [record for record in load_records(run_dir) if record.get("event") == "train"]
    if len(records) != 50:
        raise ValueError(f"expected 50 train records in {run_dir}, got {len(records)}")
    steady = [
        record["step_time_ms"] for record in records if 11 <= record["step"] <= 48
    ]
    if len(steady) != 38:
        raise ValueError(f"expected 38 steady records in {run_dir}")
    center = statistics.median(steady)
    robust_cv = 1.4826 * median_absolute_deviation(steady) / center
    summary = json.loads((run_dir / "summary.json").read_text())
    stdout = (run_dir / "stdout.log").read_text().lower()
    return {
        "steady_median_ms": center,
        "steady_min_ms": min(steady),
        "steady_max_ms": max(steady),
        "steady_robust_cv": robust_cv,
        "first_update_ms": records[0]["step_time_ms"],
        "last_train_loss": records[-1]["train_loss"],
        "wall_time_seconds": summary["wall_time_seconds"],
        "peak_memory_mib": summary["peak_memory_mib"],
        "recompile_log_mentions": stdout.count("recompil"),
        "spectral_records_present": all(
            "spectral_cap_fraction_median" in record for record in records
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    control_before = run_metrics(run_root / "control-before")
    treatment = run_metrics(run_root / "treatment")
    control_after = run_metrics(run_root / "control-after")
    control_median = statistics.fmean(
        [control_before["steady_median_ms"], control_after["steady_median_ms"]]
    )
    steady_overhead = (
        treatment["steady_median_ms"] / control_median - 1.0
    )
    proxy_updates = 1788
    control_compile = statistics.fmean(
        [
            max(
                control_before["first_update_ms"]
                - control_before["steady_median_ms"],
                0.0,
            ),
            max(
                control_after["first_update_ms"]
                - control_after["steady_median_ms"],
                0.0,
            ),
        ]
    )
    treatment_compile = max(
        treatment["first_update_ms"] - treatment["steady_median_ms"], 0.0
    )
    projected_control = control_compile + proxy_updates * control_median
    projected_treatment = (
        treatment_compile + proxy_updates * treatment["steady_median_ms"]
    )
    projected_overhead = projected_treatment / projected_control - 1.0
    control_drift = abs(
        control_after["steady_median_ms"] / control_before["steady_median_ms"] - 1.0
    )
    checks = {
        "steady_overhead_le_0_01": steady_overhead <= 0.01,
        "projected_proxy_overhead_le_0_01": projected_overhead <= 0.01,
        "control_before_noise_resolves_1pct": control_before["steady_robust_cv"] <= 0.01,
        "control_after_noise_resolves_1pct": control_after["steady_robust_cv"] <= 0.01,
        "control_drift_le_0_01": control_drift <= 0.01,
        "treatment_noise_resolves_1pct": treatment["steady_robust_cv"] <= 0.01,
        "no_recompile_evidence": treatment["recompile_log_mentions"] == 0,
        "finite_losses": math.isfinite(control_before["last_train_loss"])
        and math.isfinite(control_after["last_train_loss"])
        and math.isfinite(treatment["last_train_loss"]),
        "spectral_diagnostics_present": treatment["spectral_records_present"],
    }
    payload = {
        "experiment": "exp011-B",
        "decision": "pass" if all(checks.values()) else "kill_or_invalid",
        "control_before": control_before,
        "control_after": control_after,
        "control_reference_median_ms": control_median,
        "control_drift_fraction": control_drift,
        "treatment": treatment,
        "steady_overhead_fraction": steady_overhead,
        "projected_proxy_overhead_fraction": projected_overhead,
        "checks": checks,
    }
    output = Path(args.output) if args.output else run_root / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
