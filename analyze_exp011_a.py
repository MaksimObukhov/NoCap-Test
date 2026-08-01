#!/usr/bin/env python3
"""exp011-A: causal attribution gate for the attention-V spectral cap."""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from spectral_cap import (
    cap_leading_gap,
    leading_energy,
    preconditioned_adamw_update,
    rank_two_svd,
    synthetic_self_test,
)
from train_gpt2 import DistributedDataLoader, GPT, GPTConfig


BASELINE_COMMIT = "bd681a332"
EXPECTED_ARGS = {
    "model": "d12",
    "batch_size": 16,
    "sequence_length": 1024,
    "grad_accumulation_steps": 32,
    "seed": 0,
}


def parse_checkpoint_spec(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = spec.split("=", 1)
    return label, path


def validate_checkpoint(checkpoint, label, expected_commit):
    errors = []
    checkpoint_args = checkpoint.get("args", {})
    for key, expected in EXPECTED_ARGS.items():
        if checkpoint_args.get(key) != expected:
            errors.append(
                f"{key}: expected {expected!r}, got {checkpoint_args.get(key)!r}"
            )
    if checkpoint_args.get("run_mode") != label:
        errors.append(f"run_mode does not match label {label!r}")
    actual_commit = str(checkpoint.get("metadata", {}).get("git_commit", ""))
    if not actual_commit.startswith(expected_commit):
        errors.append(f"unexpected checkpoint commit: {actual_commit}")
    for key in ("model", "optimizer", "next_x", "next_y", "train_loader"):
        if key not in checkpoint:
            errors.append(f"missing checkpoint key: {key}")
    if errors:
        raise ValueError("; ".join(errors))


def git_metadata():
    try:
        return {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "tracked_dirty": subprocess.call(["git", "diff", "--quiet"]) != 0
            or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_dirty": None}


class AttentionInputCapture:
    def __init__(self, model, max_rows):
        self.enabled = False
        self.max_rows = max_rows
        self.inputs = {}
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            self.handles.append(
                block.attn.c_attn.register_forward_pre_hook(self._hook(layer))
            )

    def _hook(self, layer):
        def hook(_module, inputs):
            if not self.enabled or layer in self.inputs:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if values.shape[0] > self.max_rows:
                indices = torch.linspace(
                    0,
                    values.shape[0] - 1,
                    steps=self.max_rows,
                    device=values.device,
                ).long()
                values = values.index_select(0, indices)
            self.inputs[layer] = values.float().cpu()

        return hook

    def reset(self):
        self.inputs.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def median(values):
    return statistics.median(float(value) for value in values)


def parameters_equal_checkpoint(model, checkpoint_model):
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), checkpoint_model[name]):
            return False, name
    return True, None


def aggregate(records):
    functional_reduction = median(
        record["functional_leading_energy_reduction"] for record in records
    )
    descent_retention = median(record["descent_retention"] for record in records)
    retention_fraction = sum(
        record["descent_retention"] >= 0.95 for record in records
    ) / len(records)
    norm_rescale = median(record["norm_rescale"] for record in records)
    checks = {
        "median_functional_reduction_ge_0_10": functional_reduction >= 0.10,
        "median_descent_retention_ge_0_98": descent_retention >= 0.98,
        "retention_fraction_ge_0_75": retention_fraction >= 0.75,
        "median_norm_rescale_le_1_05": norm_rescale <= 1.05,
        "positive_total_descent": all(record["total_descent"] > 0 for record in records),
    }
    return {
        "records": len(records),
        "median_functional_leading_energy_reduction": functional_reduction,
        "median_descent_retention": descent_retention,
        "fraction_retaining_95pct_descent": retention_fraction,
        "median_norm_rescale": norm_rescale,
        "checks": checks,
        "passes": all(checks.values()),
    }


def analyze_checkpoint(
    label,
    checkpoint_path,
    input_bin,
    num_updates,
    sample_rows,
    power_iterations,
    expected_commit,
):
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, label, expected_commit)
    model = GPT(GPTConfig()).train().cuda()
    model.load_state_dict(checkpoint["model"])
    optimizer = model.configure_optimizers(
        checkpoint["args"]["weight_decay"],
        checkpoint["args"]["learning_rate"],
        (0.9, 0.95),
        "cuda",
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        group["weight_decay"] = 0.0

    loader = DistributedDataLoader(
        input_bin,
        checkpoint["args"]["batch_size"],
        checkpoint["args"]["sequence_length"],
        0,
        1,
    )
    loader.load_state_dict(checkpoint["train_loader"])
    x = checkpoint["next_x"].cuda()
    y = checkpoint["next_y"].cuda()
    accumulation = checkpoint["args"]["grad_accumulation_steps"]
    width = model.config.n_embd
    capture = AttentionInputCapture(model, sample_rows)
    update_records = []

    for update_index in range(num_updates):
        optimizer.zero_grad(set_to_none=True)
        capture.reset()
        train_loss = 0.0
        for micro_step in range(accumulation):
            capture.enabled = micro_step == 0
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, return_logits=False)
                loss = loss / accumulation
            train_loss += loss.detach().item()
            x, y = loader.next_batch()
            loss.backward()
        capture.enabled = False

        layer_records = []
        for layer, block in enumerate(model.transformer.h):
            parameter = block.attn.c_attn.weight
            value_update = preconditioned_adamw_update(
                parameter, optimizer, slice(2 * width, 3 * width)
            )
            value_gradient = parameter.grad[2 * width : 3 * width].float()
            singular_values, left, right = rank_two_svd(
                value_update,
                power_iterations=power_iterations,
                seed=21000 + 101 * update_index + layer,
            )
            capped, removed, scale = cap_leading_gap(
                value_update, singular_values, left, right
            )
            inputs = capture.inputs[layer].cuda(non_blocking=True)
            functional = inputs @ value_update.T
            capped_functional = inputs @ capped.T
            original_energy = leading_energy(
                functional, power_iterations=power_iterations, seed=31000 + layer
            )
            capped_energy = leading_energy(
                capped_functional,
                power_iterations=power_iterations,
                seed=41000 + layer,
            )
            total_descent = (value_gradient * value_update).sum().item()
            capped_descent = (value_gradient * capped).sum().item()
            layer_records.append(
                {
                    "layer": layer,
                    "sigma1": singular_values[0].item(),
                    "sigma2": singular_values[1].item(),
                    "cap_fraction": (
                        (singular_values[0] - singular_values[1])
                        / torch.linalg.vector_norm(value_update).clamp_min(1e-30)
                    ).item(),
                    "norm_rescale": scale.item(),
                    "removed_frobenius_fraction": (
                        torch.linalg.vector_norm(removed)
                        / torch.linalg.vector_norm(value_update).clamp_min(1e-30)
                    ).item(),
                    "functional_leading_energy": original_energy,
                    "capped_functional_leading_energy": capped_energy,
                    "functional_leading_energy_reduction": 1.0
                    - capped_energy / max(original_energy, 1e-30),
                    "total_descent": total_descent,
                    "capped_descent": capped_descent,
                    "descent_retention": capped_descent
                    / max(total_descent, 1e-30),
                }
            )
        update_records.append(
            {"update": update_index, "train_loss": train_loss, "layers": layer_records}
        )
        optimizer.step()
        print(f"{label}: causal replay {update_index + 1}/{num_updates}", flush=True)

    capture.remove()
    unchanged, changed_parameter = parameters_equal_checkpoint(model, checkpoint["model"])
    flat_records = [
        layer for update in update_records for layer in update["layers"]
    ]
    checkpoint_aggregate = aggregate(flat_records)
    decision = "pass" if checkpoint_aggregate["passes"] and unchanged else "kill"
    if not unchanged:
        decision = "invalid"
    result = {
        "label": label,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_next_step": checkpoint["next_step"],
        "checkpoint_args": checkpoint["args"],
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "checkpoint_train_loader": checkpoint["train_loader"],
        "parameters_unchanged": unchanged,
        "changed_parameter": changed_parameter,
        "updates": update_records,
        "aggregate": checkpoint_aggregate,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del capture, optimizer, model, checkpoint
    torch.cuda.empty_cache()
    return result


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_spec)
    parser.add_argument(
        "--input-bin", default="data/fineweb10B/fineweb_train_*.bin"
    )
    parser.add_argument("--output", default="runs/exp011-a/summary.json")
    parser.add_argument("--num-updates", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--power-iterations", type=int, default=8)
    parser.add_argument("--expected-checkpoint-commit", default=BASELINE_COMMIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        print("exp011-A self-test passed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint or [item[0] for item in args.checkpoint] != ["proxy", "full"]:
        parser.error("pass checkpoints in proxy, full order")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.perf_counter()
    checkpoints = [
        analyze_checkpoint(
            label,
            path,
            args.input_bin,
            args.num_updates,
            args.sample_rows,
            args.power_iterations,
            args.expected_checkpoint_commit,
        )
        for label, path in args.checkpoint
    ]
    if any(item["decision"] == "invalid" for item in checkpoints):
        decision = "invalid"
    elif all(item["decision"] == "pass" for item in checkpoints):
        decision = "pass"
    else:
        decision = "kill"
    payload = {
        "experiment": "exp011-A",
        "decision": decision,
        "checkpoints": checkpoints,
        "source": git_metadata(),
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "experiment": "exp011-A",
        "decision": decision,
        "aggregates": {
            item["label"]: item["aggregate"] for item in checkpoints
        },
        "output": str(Path(args.output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
