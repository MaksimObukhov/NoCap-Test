#!/usr/bin/env python3
"""Apply the pre-registered exp021/exp022 full-winner rule."""

import argparse
import json
from pathlib import Path


REFERENCE_FINAL_LOSS = 3.56718373298645
REFERENCE_TRAIN_SECONDS = 6782.063025594557
QUALITY_MARGIN = 0.004


def validation_records(path):
    records = []
    with Path(path).open() as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "validation" and payload.get("val_loss") is not None:
                records.append(payload)
    return sorted(records, key=lambda record: record["tokens_seen"])


def crossing_time(records, target_loss):
    previous = None
    for record in records:
        loss = record["val_loss"]
        if loss <= target_loss:
            if previous is None or previous["val_loss"] <= target_loss:
                return record["training_time_seconds"]
            loss_span = previous["val_loss"] - loss
            if loss_span <= 0:
                return record["training_time_seconds"]
            fraction = (previous["val_loss"] - target_loss) / loss_span
            return previous["training_time_seconds"] + fraction * (
                record["training_time_seconds"] - previous["training_time_seconds"]
            )
        previous = record
    return None


def choose(summary, metrics_path):
    complete = (
        summary.get("status") == "complete"
        and summary.get("tokens_seen") == 937_426_944
        and summary.get("final_checkpoint", {}).get("tokens_seen") == 937_426_944
    )
    if not complete:
        return {
            "winner": "exp021",
            "exp022_decision": "fallback-incomplete",
            "reason": "exp022 did not produce a complete exact-budget proxy",
        }
    final_loss = summary["final_val_loss"]
    training_time = summary["training_time_seconds"]
    crossing = crossing_time(validation_records(metrics_path), REFERENCE_FINAL_LOSS)
    quality_route = (
        final_loss <= REFERENCE_FINAL_LOSS - QUALITY_MARGIN
        and training_time <= REFERENCE_TRAIN_SECONDS * 1.02
    )
    time_route = (
        crossing is not None
        and crossing <= REFERENCE_TRAIN_SECONDS * 0.99
        and final_loss <= REFERENCE_FINAL_LOSS
    )
    winner = "exp022" if quality_route or time_route else "exp021"
    return {
        "winner": winner,
        "exp022_decision": "pass" if winner == "exp022" else "does-not-replace",
        "reference": {
            "experiment": "exp021",
            "final_loss": REFERENCE_FINAL_LOSS,
            "training_time_seconds": REFERENCE_TRAIN_SECONDS,
        },
        "exp022": {
            "final_loss": final_loss,
            "training_time_seconds": training_time,
            "time_to_exp021_final_loss_seconds": crossing,
        },
        "routes": {
            "quality_route": quality_route,
            "time_to_quality_route": time_route,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    if summary_path.is_file() and Path(args.metrics).is_file():
        result = choose(json.loads(summary_path.read_text()), args.metrics)
    else:
        result = {
            "winner": "exp021",
            "exp022_decision": "fallback-missing",
            "reason": "exp022 summary or metrics is missing",
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
