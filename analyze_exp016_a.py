#!/usr/bin/env python3
"""exp016-A: frozen-weight causal replay of the descent-budgeted direction."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from descent_budgeted_adamw import (
    descent_budgeted_direction,
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
    return tuple(spec.split("=", 1))


def validate_checkpoint(checkpoint, label, expected_commit):
    errors = []
    checkpoint_args = checkpoint.get("args", {})
    for key, expected in EXPECTED_ARGS.items():
        if checkpoint_args.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {checkpoint_args.get(key)!r}")
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


def source_metadata():
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tracked_dirty": subprocess.call(["git", "diff", "--quiet"]) != 0
        or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


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


def parameters_equal_checkpoint(model, checkpoint_model):
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), checkpoint_model[name]):
            return False, name
    return True, None


def aggregate(records):
    median_descent = statistics.median(record["descent_retention"] for record in records)
    median_norm = statistics.median(record["norm_retention"] for record in records)
    median_functional = statistics.median(
        record["functional_leading_energy_reduction"] for record in records
    )
    active_fraction = sum(record["alpha"] > 0 for record in records) / len(records)
    checks = {
        "median_descent_retention_ge_0_99": median_descent >= 0.99 - 1e-6,
        "median_norm_retention_ge_0_98": median_norm >= 0.98,
        "median_functional_reduction_ge_0_10": median_functional >= 0.10,
        "active_fraction_ge_0_25": active_fraction >= 0.25,
        "positive_total_descent": all(record["total_descent"] > 0 for record in records),
    }
    return {
        "records": len(records),
        "median_descent_retention": median_descent,
        "median_norm_retention": median_norm,
        "median_functional_leading_energy_reduction": median_functional,
        "active_fraction": active_fraction,
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
                _logits, loss = model(x, y, return_logits=False)
                loss = loss / accumulation
            train_loss += loss.detach().float().item()
            x, y = loader.next_batch()
            loss.backward()
        capture.enabled = False

        layer_records = []
        for layer, block in enumerate(model.transformer.h):
            parameter = block.attn.c_attn.weight
            row_slice = slice(2 * width, 3 * width)
            direction = preconditioned_adamw_update(parameter, optimizer, row_slice)
            gradient = parameter.grad[row_slice].float()
            singular_values, left, right = rank_two_svd(
                direction,
                power_iterations=power_iterations,
                seed=26000 + 101 * update_index + layer,
            )
            treated, evidence = descent_budgeted_direction(
                direction,
                gradient,
                singular_values,
                left,
                right,
                minimum_descent_retention=0.99,
            )
            inputs = capture.inputs[layer].cuda(non_blocking=True)
            functional = inputs @ direction.T
            treated_functional = inputs @ treated.T
            original_energy = leading_energy(
                functional, power_iterations=power_iterations, seed=36000 + layer
            )
            treated_energy = leading_energy(
                treated_functional,
                power_iterations=power_iterations,
                seed=46000 + layer,
            )
            layer_records.append(
                {
                    "layer": layer,
                    "sigma1": singular_values[0].item(),
                    "sigma2": singular_values[1].item(),
                    "alpha": evidence["alpha"].item(),
                    "total_descent": evidence["total_descent"].item(),
                    "descent_retention": evidence["descent_retention"].item(),
                    "norm_retention": evidence["norm_retention"].item(),
                    "functional_leading_energy": original_energy,
                    "treated_functional_leading_energy": treated_energy,
                    "functional_leading_energy_reduction": 1.0
                    - treated_energy / max(original_energy, 1e-30),
                }
            )
        update_records.append(
            {"update": update_index, "train_loss": train_loss, "layers": layer_records}
        )
        optimizer.step()
        print(f"{label}: causal replay {update_index + 1}/{num_updates}", flush=True)

    capture.remove()
    unchanged, changed_parameter = parameters_equal_checkpoint(model, checkpoint["model"])
    flat_records = [layer for update in update_records for layer in update["layers"]]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_spec)
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-updates", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--power-iterations", type=int, default=8)
    parser.add_argument("--expected-checkpoint-commit", default=BASELINE_COMMIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        print("exp016-A self-test passed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint or [item[0] for item in args.checkpoint] != ["proxy", "full"]:
        parser.error("pass checkpoints in proxy, full order")

    source = source_metadata()
    if source["tracked_dirty"]:
        raise RuntimeError("tracked worktree changes invalidate exp016-A")
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
        "experiment": "exp016-A",
        "decision": decision,
        "checkpoints": checkpoints,
        "source": source,
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
