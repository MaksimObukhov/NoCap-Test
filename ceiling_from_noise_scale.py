"""Turn the measured B_crit curve into a predicted token saving, before spending money.

McCandlish's model says that per unit of optimization progress, the token cost of
training at batch B is proportional to (B + B_noise): you pay B tokens per step,
and the step is worth 1/(1 + B_noise/B) of a noise-free step. Integrating that
along the measured B_noise(t) gives a predicted cost for any batch schedule.

The model is approximate, so it is calibrated against the one schedule whose real
answer we already know: exp001 delivered a measured 1.06x token advantage at the
end of the proxy. Whatever factor the model over-predicts that by is applied to
every other candidate.

usage: ceiling_from_noise_scale.py METRICS_JSONL
"""

import json
import statistics
import sys

BASELINE_BATCH = 524_288
MICRO_BATCH = 16_384
PROXY_TOKENS = 937_426_944
EMA_ALPHA = 0.3
EXP001_MEASURED_ADVANTAGE = 1.06


def load_curve(path):
    records = [json.loads(line) for line in open(path)]
    records = [
        r for r in records
        if r.get("event") == "noise_scale" and r.get("b_simple_tokens")
    ]
    records.sort(key=lambda r: r["tokens_seen"])
    curve, ema = [], None
    for r in records:
        b = r["b_simple_tokens"]
        ema = b if ema is None else EMA_ALPHA * b + (1 - EMA_ALPHA) * ema
        curve.append((r["tokens_seen"], ema))
    return curve


def main():
    if len(sys.argv) != 2:
        print("usage: ceiling_from_noise_scale.py METRICS_JSONL", file=sys.stderr)
        return 2
    curve = load_curve(sys.argv[1])

    def b_noise(tokens):
        if tokens <= curve[0][0]:
            return curve[0][1]
        for (t0, e0), (t1, e1) in zip(curve, curve[1:]):
            if t0 <= tokens <= t1:
                return e0 + (e1 - e0) * (tokens - t0) / (t1 - t0)
        return curve[-1][1]

    below = [t for t, e in curve if e < BASELINE_BATCH]
    middle = [e for t, e in curve if 0.1e9 < t < 0.7e9]
    print("=== how over-batched was the baseline? ===")
    print(f"  B_crit stays below {BASELINE_BATCH:,} up to {max(below):,} tokens "
          f"= {max(below)/PROXY_TOKENS:.0%} of the proxy budget")
    print(f"  mid-training (0.1-0.7B) median B_crit = {statistics.median(middle):,.0f} "
          f"= {statistics.median(middle)/BASELINE_BATCH:.2f}x the baseline batch")
    print(f"  i.e. the baseline ran ~{BASELINE_BATCH/statistics.median(middle):.1f}x "
          "larger than the noise scale for most of the run")

    def cost(schedule):
        total = 0.0
        for (t0, _), (t1, _) in zip(curve, curve[1:]):
            middle_t = (t0 + t1) / 2
            total += (t1 - t0) * (schedule(middle_t) + b_noise(middle_t))
        return total

    def ramp(start, end_fraction):
        end = end_fraction * PROXY_TOKENS

        def schedule(t):
            if t >= end:
                return BASELINE_BATCH
            return start + (BASELINE_BATCH - start) * (t / end)
        return schedule

    baseline_cost = cost(lambda t: BASELINE_BATCH)
    candidates = {
        "exp001 as run (131k -> 524k over 50%)": ramp(131_072, 0.50),
        "accum=1 flat (16,384 tokens)": lambda t: MICRO_BATCH,
        "track B_crit exactly": b_noise,
        "16k -> 524k over 90% of tokens": ramp(MICRO_BATCH, 0.90),
        "16k -> 524k over 97% of tokens": ramp(MICRO_BATCH, 0.97),
    }

    raw = {name: baseline_cost / cost(f) for name, f in candidates.items()}
    print("\n=== raw model prediction (token advantage vs baseline) ===")
    for name, value in raw.items():
        print(f"  {name:38s} {value:5.2f}x")

    exp001_key = "exp001 as run (131k -> 524k over 50%)"
    over = (raw[exp001_key] - 1) / (EXP001_MEASURED_ADVANTAGE - 1)
    print(f"\n=== calibration against the one schedule we actually ran ===")
    print(f"  model says exp001 = {raw[exp001_key]:.2f}x, "
          f"exp001 measured {EXP001_MEASURED_ADVANTAGE:.2f}x")
    print(f"  -> the model over-states the GAIN by {over:.1f}x on this problem")

    print("\n=== calibrated (divide each gain by that factor) ===")
    for name, value in raw.items():
        if name == exp001_key:
            continue
        calibrated = 1 + (value - 1) / over
        print(f"  {name:38s} {calibrated:5.2f}x  "
              f"({100*(1-1/calibrated):4.1f}% fewer tokens)")

    print("\nTreat these as an ordering, not as forecasts: the calibration comes")
    print("from a 6% effect and is being applied to much larger ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
