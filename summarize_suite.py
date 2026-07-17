#!/usr/bin/env python3
import argparse
import json
import os
import statistics


def write_json_atomic(path, payload):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir")
    args = parser.parse_args()

    seeds = [0, 1, 2]
    summaries = []
    for seed in seeds:
        path = os.path.join(args.suite_dir, f"proxy-seed-{seed}", "summary.json")
        with open(path) as f:
            summary = json.load(f)
        if summary.get("status") != "complete" or summary.get("run_mode") != "proxy":
            raise ValueError(f"not a completed proxy summary: {path}")
        if summary.get("seed") != seed:
            raise ValueError(f"unexpected seed in {path}")
        summaries.append(summary)

    losses = [summary["final_val_loss"] for summary in summaries]
    result = {
        "mode": "proxy",
        "n": len(losses),
        "seeds": seeds,
        "final_val_losses": losses,
        "mean_final_val_loss": statistics.mean(losses),
        "sample_std_final_val_loss": statistics.stdev(losses),
        "total_tokens": sum(summary["tokens_seen"] for summary in summaries),
        "total_training_time_seconds": sum(
            summary["training_time_seconds"] for summary in summaries
        ),
    }
    output_path = os.path.join(args.suite_dir, "proxy_statistics.json")
    write_json_atomic(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
