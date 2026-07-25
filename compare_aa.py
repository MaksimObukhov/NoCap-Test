"""Compare two metrics.jsonl train-loss traces update by update.

Used as the A/A gate: the instrumented binary, with measurement disabled, must
reproduce the recorded baseline. Exits non-zero if it does not, so run_aa.sh
fails loudly rather than letting a drifted binary into a paid measurement run.

Bitwise equality is the expected outcome for identical code, seed and GPU model.
Tiny drift can still come from a different card or a different cuBLAS autotune
choice, so a small tolerance is allowed and reported rather than hidden.
"""

import json
import sys

TOLERANCE = 1e-6


def load_train_losses(path):
    losses = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("event") == "train":
                losses[record["step"]] = record["train_loss"]
    return losses


def main():
    if len(sys.argv) != 3:
        print("usage: compare_aa.py REFERENCE_METRICS CANDIDATE_METRICS", file=sys.stderr)
        return 2

    reference = load_train_losses(sys.argv[1])
    candidate = load_train_losses(sys.argv[2])
    shared = sorted(set(reference) & set(candidate))

    if not shared:
        print("FAIL: no overlapping updates between the two runs", file=sys.stderr)
        return 3

    exact = 0
    worst_step = None
    worst_delta = 0.0
    for step in shared:
        delta = abs(candidate[step] - reference[step])
        if delta == 0.0:
            exact += 1
        if delta > worst_delta:
            worst_delta, worst_step = delta, step

    print(f"compared {len(shared)} updates (steps {shared[0]}..{shared[-1]})")
    print(f"bitwise identical: {exact}/{len(shared)}")
    print(f"largest deviation: {worst_delta:.3e} at step {worst_step}")
    for step in shared[:5]:
        print(
            f"  step {step:4d}  reference {reference[step]:.8f}  "
            f"candidate {candidate[step]:.8f}"
        )

    if worst_delta == 0.0:
        print("PASS: bitwise identical to the baseline.")
        return 0
    if worst_delta <= TOLERANCE:
        print(f"PASS: within tolerance {TOLERANCE:.0e}, but not bitwise identical.")
        print("      note it in EXPERIMENTS.md - it means the card or the kernel")
        print("      selection differs from the baseline instance.")
        return 0

    print(
        f"FAIL: deviation {worst_delta:.3e} exceeds tolerance {TOLERANCE:.0e}.",
        file=sys.stderr,
    )
    print(
        "      do not start the measurement run; the binary is not reproducing "
        "the baseline.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
