#!/usr/bin/env python3
import json
import sys
from pathlib import Path


PROXY_SHORT_UPDATES = 896
PROXY_TOTAL_UPDATES = 1788
MIN_STEADY_SPEEDUP = 0.03
MIN_PROJECTED_NET_SPEEDUP = 0.01
MAX_LONG_CONTROL_DRIFT = 0.03


def load_summary(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"{path}: run is not complete")
    return value


def stage(summary: dict, name: str) -> dict:
    matches = [
        value for value in summary["train_shape_stages"] if value["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} stage, found {len(matches)}")
    return matches[0]


def compile_penalty_seconds(stage_record: dict) -> float:
    first = float(stage_record["first_step_time_ms"])
    steady = float(stage_record["steady_median_step_time_ms"])
    return max(0.0, first - steady) / 1000


def analyse_gate(fixed_summary: dict, scheduled_summary: dict) -> dict:
    fixed = stage(fixed_summary, "fixed")
    short = stage(scheduled_summary, "short")
    long = stage(scheduled_summary, "long")

    fixed_ms = float(fixed["steady_median_step_time_ms"])
    short_ms = float(short["steady_median_step_time_ms"])
    long_ms = float(long["steady_median_step_time_ms"])
    steady_speedup = long_ms / short_ms - 1
    long_control_drift = long_ms / fixed_ms - 1

    gross_saving_seconds = (long_ms - short_ms) * PROXY_SHORT_UPDATES / 1000
    additional_compile_seconds = (
        compile_penalty_seconds(short)
        + compile_penalty_seconds(long)
        - compile_penalty_seconds(fixed)
    )
    additional_compile_seconds = max(0.0, additional_compile_seconds)
    projected_net_saving_seconds = (
        gross_saving_seconds - additional_compile_seconds
    )
    projected_fixed_proxy_seconds = (
        fixed_ms * PROXY_TOTAL_UPDATES / 1000
        + compile_penalty_seconds(fixed)
    )
    projected_net_speedup = (
        projected_net_saving_seconds / projected_fixed_proxy_seconds
    )

    checks = {
        "steady_t512_speedup_at_least_3pct": steady_speedup
        >= MIN_STEADY_SPEEDUP,
        "projected_net_speedup_at_least_1pct": projected_net_speedup
        >= MIN_PROJECTED_NET_SPEEDUP,
        "scheduled_t1024_matches_fixed_within_3pct": abs(long_control_drift)
        <= MAX_LONG_CONTROL_DRIFT,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "fixed_t1024_steady_ms": fixed_ms,
        "scheduled_t512_steady_ms": short_ms,
        "scheduled_t1024_steady_ms": long_ms,
        "steady_t512_speedup": steady_speedup,
        "long_control_drift": long_control_drift,
        "gross_proxy_saving_seconds": gross_saving_seconds,
        "additional_compile_seconds": additional_compile_seconds,
        "projected_net_saving_seconds": projected_net_saving_seconds,
        "projected_net_speedup": projected_net_speedup,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_exp005_gate.py GATE_DIR")
    gate_dir = Path(sys.argv[1])
    result = analyse_gate(
        load_summary(gate_dir / "fixed-t1024" / "summary.json"),
        load_summary(gate_dir / "scheduled" / "summary.json"),
    )
    output_path = gate_dir / "gate-summary.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    if result["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
