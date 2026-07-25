"""Read the B_simple curve out of a metrics.jsonl produced with --noise_scale_every.

B_simple is a ratio of two noisy quantities estimated from ~32 samples, so the
per-step values are jumpy by construction. The trend is what carries meaning,
so this prints both the raw values and an exponential moving average, and then
answers the question exp002 exists to answer: where does B_simple cross the
baseline's effective batch of 524,288 tokens?
"""

import json
import sys

BASELINE_BATCH_TOKENS = 524_288
EMA_ALPHA = 0.3


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_noise_scale.py METRICS_JSONL [...]", file=sys.stderr)
        return 2

    records = []
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("event") == "noise_scale":
                    records.append(record)
    records.sort(key=lambda r: r["tokens_seen"])

    usable = [r for r in records if r.get("b_simple_tokens") is not None]
    if not usable:
        print("no usable B_simple estimates", file=sys.stderr)
        print(f"({len(records)} measurement steps, all with a negative variance "
              "or signal estimate - the estimator needs more micro-batches)",
              file=sys.stderr)
        return 3

    print(f"{len(usable)}/{len(records)} measurement steps produced a usable estimate")
    print()
    print(f"{'update':>7} {'tokens':>13} {'train_loss':>11} "
          f"{'B_simple':>12} {'EMA':>12} {'vs 524288':>10}")

    ema = None
    crossing = None
    previous = None
    for r in usable:
        b = r["b_simple_tokens"]
        ema = b if ema is None else EMA_ALPHA * b + (1 - EMA_ALPHA) * ema
        if crossing is None and previous is not None:
            if previous[1] < BASELINE_BATCH_TOKENS <= ema:
                crossing = (previous[0], r["tokens_seen"])
        previous = (r["tokens_seen"], ema)
        print(
            f"{r['step']:>7} {r['tokens_seen']:>13,} {r['train_loss']:>11.5f} "
            f"{b:>12,.0f} {ema:>12,.0f} {ema / BASELINE_BATCH_TOKENS:>9.2f}x"
        )

    print()
    first, last = usable[0], usable[-1]
    print(f"first estimate: B_simple = {first['b_simple_tokens']:,.0f} tokens at "
          f"{first['tokens_seen']:,} tokens "
          f"({BASELINE_BATCH_TOKENS / first['b_simple_tokens']:.1f}x below the "
          f"baseline batch)")
    print(f"last  estimate: EMA      = {ema:,.0f} tokens at {last['tokens_seen']:,} tokens")

    if crossing is not None:
        print(f"\nEMA crosses {BASELINE_BATCH_TOKENS:,} between {crossing[0]:,} and "
              f"{crossing[1]:,} tokens.")
        print("Index any future batch ramp on that absolute token count, not on a "
              "fraction of the budget.")
    elif ema < BASELINE_BATCH_TOKENS:
        print(f"\nEMA never reaches {BASELINE_BATCH_TOKENS:,} within this run - the "
              "baseline batch is still above B_simple at the stopping point.")
        print("Either extend the run, or read it as: the whole measured stretch is "
              "over-batched.")
    else:
        print(f"\nEMA starts at or above {BASELINE_BATCH_TOKENS:,} - prediction (1) "
              "in EXPERIMENTS.md is falsified and the batch story is dead.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
