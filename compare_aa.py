"""Compare two metrics.jsonl train-loss traces from runs that should be equivalent.

Used as the A/A gate before a paid measurement run.

Training is chaotic: a difference of one float32 ulp in the first forward pass is
amplified roughly tenfold every ten updates, so two runs of identical code on two
different rented cards will not agree bitwise, and demanding that they do is a
test that fails for reasons that do not matter. Different cards, drivers or
cuBLAS autotune choices change the reduction order, and that is enough.

So the gate asks the three questions that actually distinguish "same code,
different hardware" from "the code changed":

  1. does the very first update agree to within a few ulp? Before any optimizer
     step has compounded anything, this tests initialisation, data order and the
     forward path. A code change shows up here immediately.
  2. is the sign of the deviation symmetric? Chaotic divergence wanders either
     way; a code change biases it one way.
  3. does the late deviation stay inside the run's own step-to-step jitter? If
     the two traces differ by less than one update's natural movement, they are
     the same trajectory with a different rounding seed.
"""

import json
import statistics
import sys

FLOAT32_EPS = 1.1920929e-7
FIRST_STEP_ULP_BUDGET = 8.0
SIGN_BALANCE_RANGE = (0.2, 0.8)
LATE_JITTER_BUDGET = 3.0


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
    if len(shared) < 10:
        print(f"FAIL: only {len(shared)} overlapping updates; need at least 10",
              file=sys.stderr)
        return 3

    deviations = [(s, candidate[s] - reference[s]) for s in shared]
    print(f"compared {len(shared)} updates (steps {shared[0]}..{shared[-1]})")

    # 1. first update - the only point not yet contaminated by amplification
    first = shared[0]
    first_dev = abs(candidate[first] - reference[first])
    ulp = FLOAT32_EPS * abs(reference[first])
    first_ulps = first_dev / ulp
    print(f"\n1. first update (step {first})")
    print(f"   reference {reference[first]:.8f}  candidate {candidate[first]:.8f}")
    print(f"   deviation {first_dev:.3e} = {first_ulps:.2f} ulp "
          f"(1 ulp = {ulp:.3e} at this magnitude)")
    first_ok = first_ulps <= FIRST_STEP_ULP_BUDGET
    print(f"   {'ok' if first_ok else 'FAIL'}: budget {FIRST_STEP_ULP_BUDGET:.0f} ulp")

    # 2. sign symmetry - chaos wanders, a code change drifts
    positive = sum(1 for _, d in deviations if d > 0)
    fraction = positive / len(deviations)
    mean_dev = statistics.mean(d for _, d in deviations)
    print(f"\n2. sign balance")
    print(f"   candidate higher on {positive}/{len(deviations)} updates "
          f"({fraction:.0%}), mean deviation {mean_dev:+.4f}")
    sign_ok = SIGN_BALANCE_RANGE[0] <= fraction <= SIGN_BALANCE_RANGE[1]
    print(f"   {'ok' if sign_ok else 'FAIL'}: expected "
          f"{SIGN_BALANCE_RANGE[0]:.0%}-{SIGN_BALANCE_RANGE[1]:.0%} for chaotic drift")

    # 3. late deviation against the reference's own step-to-step movement
    late_from = shared[len(shared) * 2 // 3]
    late = [abs(d) for s, d in deviations if s >= late_from]
    consecutive = [
        reference[b] - reference[a]
        for a, b in zip(shared, shared[1:])
        if a >= late_from and b == a + 1
    ]
    print(f"\n3. late deviation (steps {late_from}..{shared[-1]})")
    if len(consecutive) < 3:
        print("   not enough consecutive updates to measure jitter; skipping")
        late_ok = True
    else:
        jitter = statistics.pstdev(consecutive)
        ratio = max(late) / jitter
        print(f"   max |deviation| {max(late):.4f}, mean {statistics.mean(late):.4f}")
        print(f"   reference's own step-to-step jitter {jitter:.4f}")
        print(f"   deviation is {ratio:.2f}x one update's natural movement")
        late_ok = ratio <= LATE_JITTER_BUDGET
        print(f"   {'ok' if late_ok else 'FAIL'}: budget {LATE_JITTER_BUDGET:.0f}x")

    print()
    if first_ok and sign_ok and late_ok:
        print("PASS: same trajectory, different rounding. The binary reproduces the")
        print("      baseline; the divergence is floating-point non-determinism from")
        print("      a different card or kernel choice, amplified by training.")
        if first_dev > 0.0:
            print("      Note this is not a bitwise match, so single-seed paired")
            print("      comparisons across instances carry this divergence floor.")
        return 0

    print("FAIL: the deviation does not look like hardware non-determinism.",
          file=sys.stderr)
    if not first_ok:
        print("      The very first update already disagrees - suspect the code,",
              file=sys.stderr)
        print("      the seed, or the data order, not the card.", file=sys.stderr)
    if not sign_ok:
        print("      The deviation is one-sided, which chaotic drift is not.",
              file=sys.stderr)
    if not late_ok:
        print("      The trajectories separated by more than natural jitter.",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
