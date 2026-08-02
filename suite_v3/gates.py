"""Gate decisions for the v3 proxy suite.

Every function here is pure: dicts and numbers in, a decision out. Nothing
imports torch, touches the filesystem, or talks to W&B, so the whole
decision surface is testable on a laptop without a GPU. That separation is
deliberate -- in the v2 suite the gate logic was interleaved with process
management, and the one confirmed false kill (exp015) lived in a branch that
had no test covering it.

Decisions are one of:

    "pass"          -- proceed to the next stage
    "warning"       -- proceed, but the report must say why
    "inconclusive"  -- stop this experiment, no claim either way
    "kill"          -- stop this experiment, negative result
    "invalid"       -- the measurement itself is untrustworthy; not a result
"""

import statistics

PASS = "pass"
WARNING = "warning"
INCONCLUSIVE = "inconclusive"
KILL = "kill"
INVALID = "invalid"


def _decision(name, decision, evidence):
    return {"stage": name, "decision": decision, "evidence": dict(evidence)}


# ---------------------------------------------------------------------------
# Accounting


def accounting_gate(report, expectations):
    """Compare a model's self-reported shape against what was preregistered.

    `report` comes from accounting.py running inside the experiment worktree;
    `expectations` comes from the suite manifest. Any mismatch is `invalid`
    rather than `kill`: a parameter count that disagrees with the
    preregistration means the experiment under test is not the experiment that
    was described, so there is no scientific claim to reject.
    """
    mismatches = []
    for key, expected in sorted(expectations.items()):
        if key not in report:
            mismatches.append(f"{key}: missing from accounting report")
            continue
        actual = report[key]
        if isinstance(expected, float) or isinstance(actual, float):
            if abs(float(actual) - float(expected)) > 1e-6:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            mismatches.append(f"{key}: expected {expected}, got {actual}")

    for flag in ("tying_preserved", "optimizer_groups_cover_all_parameters"):
        if flag in report and not report[flag]:
            mismatches.append(f"{flag} is false")

    equivalence = report.get("reference_equivalence")
    if equivalence is not None and not equivalence.get("passed", False):
        mismatches.append(
            "reference_equivalence failed: max abs difference "
            f"{equivalence.get('max_abs_difference')} exceeds tolerance "
            f"{equivalence.get('tolerance')}"
        )

    decision = PASS if not mismatches else INVALID
    return _decision(
        "accounting", decision, {"mismatches": mismatches, "report": report}
    )


# ---------------------------------------------------------------------------
# Benchmark


def steady_step_stats(records, trim_leading=10, trim_trailing=2):
    """Median steady-state step time over an exact-shape benchmark run.

    Benchmark stages run a flat effective batch for both control and
    treatment, so every retained update covers the same number of tokens and
    the medians are directly comparable. The v2 suite compared a ramped
    treatment against a flat reference and ended up taking a median over
    mixed batch phases with unequal record counts.
    """
    train = [r for r in records if r.get("event") == "train"]
    if trim_trailing > 0:
        kept = train[trim_leading:-trim_trailing]
    else:
        kept = train[trim_leading:]
    if not kept:
        return None
    batches = {r.get("effective_batch_tokens") for r in kept}
    step_times = [r["step_time_ms"] for r in kept]
    return {
        "updates": len(kept),
        "median_step_time_ms": statistics.median(step_times),
        "mean_step_time_ms": statistics.fmean(step_times),
        "distinct_effective_batches": sorted(b for b in batches if b is not None),
        "total_train_seconds": train[-1].get("training_time_seconds"),
    }


def benchmark_gate(
    treatment,
    control_before,
    control_after,
    warning_regression_pct=1.0,
    kill_regression_pct=2.0,
    drift_tolerance_pct=1.0,
):
    """Bracketed same-host benchmark: control, treatment, control again.

    The two controls bound clock and thermal drift across the measurement.
    If they disagree by more than `drift_tolerance_pct` the host moved under
    us and the treatment number means nothing, so the stage is `invalid`
    rather than a verdict. The v2 suite measured its references once at the
    start of the night and compared exp015 against them many hours later.
    """
    evidence = {
        "treatment": treatment,
        "control_before": control_before,
        "control_after": control_after,
        "warning_regression_pct": warning_regression_pct,
        "kill_regression_pct": kill_regression_pct,
        "drift_tolerance_pct": drift_tolerance_pct,
    }

    if not (treatment and control_before and control_after):
        evidence["reason"] = "missing one or more bracketed measurements"
        return _decision("benchmark", INVALID, evidence)

    if treatment["distinct_effective_batches"] != control_before[
        "distinct_effective_batches"
    ]:
        evidence["reason"] = (
            "treatment and control did not run the same effective batch shape"
        )
        return _decision("benchmark", INVALID, evidence)

    before = control_before["median_step_time_ms"]
    after = control_after["median_step_time_ms"]
    drift_pct = 100.0 * abs(after - before) / before
    evidence["control_drift_pct"] = drift_pct
    if drift_pct > drift_tolerance_pct:
        evidence["reason"] = (
            f"control drifted {drift_pct:.2f}% between brackets, above the "
            f"{drift_tolerance_pct:.2f}% tolerance"
        )
        return _decision("benchmark", INVALID, evidence)

    reference = statistics.fmean([before, after])
    treatment_time = treatment["median_step_time_ms"]
    # Positive means the treatment is slower than the control.
    regression_pct = 100.0 * (treatment_time / reference - 1.0)
    evidence["reference_median_step_time_ms"] = reference
    evidence["regression_pct"] = regression_pct
    evidence["speedup_pct"] = -regression_pct

    if regression_pct > kill_regression_pct:
        decision = KILL
    elif regression_pct > warning_regression_pct:
        decision = WARNING
    else:
        decision = PASS
    return _decision("benchmark", decision, evidence)


# ---------------------------------------------------------------------------
# Proxy


def proxy_gate(summary, pass_loss, kill_loss, reference_label, reference_loss):
    """Quality verdict for a completed proxy stage.

    A stage that aborted on a tripwire, or finished without retained final
    weights, is not scored: it is `invalid`. Only a run that actually reached
    its token budget gets a pass/kill/inconclusive verdict.
    """
    status = summary.get("status")
    evidence = {
        "status": status,
        "pass_loss": pass_loss,
        "kill_loss": kill_loss,
        "reference_label": reference_label,
        "reference_loss": reference_loss,
        "final_val_loss": summary.get("final_val_loss"),
        "tokens_seen": summary.get("tokens_seen"),
        "target_tokens": summary.get("target_tokens"),
        "training_time_seconds": summary.get("training_time_seconds"),
        "gradient_health_by_phase": summary.get("gradient_health_by_phase"),
    }

    if status != "complete":
        evidence["reason"] = f"stage did not complete (status {status!r})"
        return _decision("proxy", INVALID, evidence)

    if summary.get("tokens_seen") != summary.get("target_tokens"):
        evidence["reason"] = "token budget not reached"
        return _decision("proxy", INVALID, evidence)

    final_checkpoint = summary.get("final_checkpoint")
    if not final_checkpoint or not final_checkpoint.get("sha256"):
        evidence["reason"] = "no validated final checkpoint"
        return _decision("proxy", INVALID, evidence)
    evidence["final_checkpoint_sha256"] = final_checkpoint["sha256"]

    loss = summary.get("final_val_loss")
    if loss is None:
        evidence["reason"] = "no final validation loss"
        return _decision("proxy", INVALID, evidence)

    if reference_loss is not None:
        evidence["delta_vs_reference"] = loss - reference_loss

    if loss <= pass_loss:
        decision = PASS
    elif loss >= kill_loss:
        decision = KILL
    else:
        decision = INCONCLUSIVE
    return _decision("proxy", decision, evidence)


# ---------------------------------------------------------------------------
# Validation envelope and post-hoc health reporting


def envelope_from_curve(validation_records, margin, floor=None):
    """Build a tripwire envelope from a reference run's validation curve.

    The envelope stops divergence, it does not adjudicate quality. A merely
    worse candidate must be allowed to finish, because a finished worse run
    is evidence and an aborted one is not -- hence a deliberately loose
    default margin.
    """
    points = []
    for record in validation_records:
        if record.get("event") != "validation":
            continue
        tokens = record.get("tokens_seen")
        loss = record.get("val_loss")
        if tokens is None or loss is None:
            continue
        limit = loss + margin
        if floor is not None:
            limit = max(limit, floor)
        points.append([int(tokens), float(limit)])
    points.sort()
    return points


def spike_clusters(records, phase, spike_multiple=10.0):
    """Group gradient-norm excursions into events, within one phase only.

    Two properties the v2 gate lacked. First, the reference median is taken
    inside the phase, so a calm steady state cannot drag the threshold down
    for a noisy warmup. Second, consecutive excursions count as one event: a
    spike and the recovery update that follows it are one thing happening,
    not two independent pathologies.

    This is reporting, not a gate. Nothing in the v3 suite kills on it.
    """
    phase_records = [
        r
        for r in records
        if r.get("event") == "train"
        and r.get("phase") == phase
        and r.get("pre_clip_grad_norm") is not None
    ]
    if not phase_records:
        return {"phase": phase, "updates": 0, "clusters": []}

    norms = [r["pre_clip_grad_norm"] for r in phase_records]
    median = statistics.median(norms)
    threshold = spike_multiple * max(median, 1e-30)

    clusters = []
    current = None
    previous_step = None
    for record in phase_records:
        step = record.get("step")
        if record["pre_clip_grad_norm"] > threshold:
            if current is not None and previous_step == step - 1:
                current["steps"].append(step)
                current["peak"] = max(current["peak"], record["pre_clip_grad_norm"])
            else:
                if current is not None:
                    clusters.append(current)
                current = {
                    "steps": [step],
                    "peak": record["pre_clip_grad_norm"],
                }
            previous_step = step
        elif current is not None and previous_step is not None and step > previous_step + 1:
            clusters.append(current)
            current = None
            previous_step = None
    if current is not None:
        clusters.append(current)

    return {
        "phase": phase,
        "updates": len(phase_records),
        "median_grad_norm": median,
        "spike_threshold": threshold,
        "max_grad_norm": max(norms),
        "clusters": clusters,
    }


def checkpoint_field_failures(declared, observed):
    """Compare an exp016 replay checkpoint against its pinned declaration.

    exp016's own Stage A validator checks a few args and a commit prefix, but
    not finality, step, tokens or content hash, so a mid-run baseline
    checkpoint with the right run_mode could have passed it. The suite
    validates canonicity here first and treats the branch script's opinion as
    a second check rather than the gate.
    """
    failures = []
    if observed.get("missing"):
        return [f"missing at {observed.get('path')}"]
    if observed.get("bytes") != declared["bytes"]:
        failures.append(
            f"bytes {observed.get('bytes')} != {declared['bytes']}"
        )
    if observed.get("sha256") != declared["sha256"]:
        failures.append(f"sha256 {observed.get('sha256')} != {declared['sha256']}")
        # A content mismatch makes every other field meaningless.
        return failures
    for field, key in (
        ("next_step", "expect_next_step"),
        ("tokens_seen", "expect_tokens_seen"),
        ("run_mode", "expect_run_mode"),
        ("seed", "expect_seed"),
    ):
        if observed.get(field) != declared[key]:
            failures.append(
                f"{field} {observed.get(field)!r} != {declared[key]!r}"
            )
    for required in ("model", "optimizer", "train_loader", "rng_state"):
        if required not in observed.get("state_keys", []):
            failures.append(f"checkpoint missing {required}")
    return failures


def suite_should_continue(decision):
    """Whether a stage decision permits running the next stage downstream.

    Only `pass` and `warning` continue. `warning` deliberately continues:
    exp019's benchmark band between 1% and 2% is a "note it and proceed"
    case, not a stop.
    """
    return decision in {PASS, WARNING}
