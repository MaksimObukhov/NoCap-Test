#!/usr/bin/env python3
"""Frozen-checkpoint falsifier for Residual-Complement Attention (exp009)."""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

from train_gpt2 import DistributedDataLoader, GPT, GPTConfig


BASELINE_COMMIT = "bd681a332"
EXPECTED_ARGS = {
    "model": "d12",
    "batch_size": 16,
    "sequence_length": 1024,
    "grad_accumulation_steps": 32,
    "seed": 0,
}


def mean_se(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def rca_update(attention, residual_direction, gate):
    attention_fp32 = attention.float()
    residual_fp32 = residual_direction.float()
    coefficient = (attention_fp32 * residual_fp32).sum(-1, keepdim=True)
    coefficient = coefficient / residual_fp32.square().sum(-1, keepdim=True).clamp_min(1e-12)
    projection = (coefficient * residual_fp32).to(attention.dtype)
    return attention - torch.tanh(gate).to(attention.dtype) * projection


def self_test():
    torch.manual_seed(7)
    attention = torch.randn(3, 5, 11)
    residual = torch.randn_like(attention)
    zero = rca_update(attention, residual, torch.tensor(0.0))
    assert torch.equal(zero, attention)
    removed = rca_update(attention, residual, torch.tensor(20.0))
    dot = ((attention - removed) * residual).sum(-1)
    expected = (attention * residual).sum(-1)
    torch.testing.assert_close(dot, expected, rtol=2e-5, atol=2e-5)
    mean, se = mean_se([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and se > 0
    print("exp009 self-test passed")


def git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        tracked_dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0 or subprocess.call(
            ["git", "diff", "--cached", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return {"commit": commit, "tracked_dirty": tracked_dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_dirty": None}


def parse_checkpoint_spec(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path


def validate_checkpoint(checkpoint, label, expected_commit):
    errors = []
    checkpoint_args = checkpoint.get("args", {})
    for key, expected in EXPECTED_ARGS.items():
        actual = checkpoint_args.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")
    if checkpoint_args.get("run_mode") != label:
        errors.append(
            f"run_mode: expected label {label!r}, got {checkpoint_args.get('run_mode')!r}"
        )
    actual_commit = str(checkpoint.get("metadata", {}).get("git_commit", ""))
    if expected_commit and not actual_commit.startswith(expected_commit):
        errors.append(
            f"git_commit: expected prefix {expected_commit!r}, got {actual_commit!r}"
        )
    required = ("model", "next_x", "next_y", "train_loader", "next_step")
    for key in required:
        if key not in checkpoint:
            errors.append(f"missing checkpoint key: {key}")
    if errors:
        raise ValueError(f"invalid {label} checkpoint: " + "; ".join(errors))


class RCAHooks:
    def __init__(self, model):
        self.gates = torch.zeros(
            model.config.n_layer, device="cuda", dtype=torch.float32, requires_grad=True
        )
        self.collect_metrics = True
        self.current_inputs = {}
        self.current_metrics = {}
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            self.handles.append(
                block.attn.register_forward_pre_hook(self._pre_hook(layer))
            )
            self.handles.append(block.attn.register_forward_hook(self._hook(layer)))

    def _pre_hook(self, layer):
        def hook(_module, inputs):
            self.current_inputs[layer] = inputs[0]

        return hook

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            residual = self.current_inputs.pop(layer)
            if self.collect_metrics:
                attention = output.detach().float()
                direction = residual.detach().float()
                dot = (attention * direction).sum(-1)
                cosine_sq = dot.square() / (
                    attention.square().sum(-1)
                    * direction.square().sum(-1)
                ).clamp_min(1e-20)
                shifted = torch.roll(direction, shifts=1, dims=1)
                shifted_dot = (attention * shifted).sum(-1)
                shifted_cosine_sq = shifted_dot.square() / (
                    attention.square().sum(-1)
                    * shifted.square().sum(-1)
                ).clamp_min(1e-20)
                self.current_metrics[layer] = {
                    "alignment_cosine_sq": cosine_sq.mean().item(),
                    "shifted_null_cosine_sq": shifted_cosine_sq.mean().item(),
                }
            return rca_update(output, residual, self.gates[layer])

        return hook

    def clear_batch(self):
        self.current_inputs.clear()
        self.current_metrics.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def checkpoint_batches(checkpoint, input_bin, count):
    args = checkpoint["args"]
    loader = DistributedDataLoader(
        input_bin,
        args["batch_size"],
        args["sequence_length"],
        process_rank=0,
        num_processes=1,
    )
    loader.load_state_dict(checkpoint["train_loader"])
    x = checkpoint["next_x"]
    y = checkpoint["next_y"]
    for _ in range(count):
        yield x, y
        x, y = loader.next_batch()


def aggregate_layer(batch_records, finite_records, layer):
    alignment = [
        record["layers"][layer]["alignment_cosine_sq"] for record in batch_records
    ]
    shifted = [
        record["layers"][layer]["shifted_null_cosine_sq"] for record in batch_records
    ]
    excess = [value - null for value, null in zip(alignment, shifted)]
    gradients = [record["gate_gradients"][layer] for record in batch_records]
    slopes = [record["slopes"][layer] for record in finite_records]
    alignment_mean, alignment_se = mean_se(alignment)
    shifted_mean, shifted_se = mean_se(shifted)
    excess_mean, excess_se = mean_se(excess)
    gradient_mean, gradient_se = mean_se(gradients)
    slope_mean, slope_se = mean_se(slopes)
    negative_fraction = sum(value < 0 for value in gradients) / len(gradients)
    checks = {
        "alignment_excess_gt_2se": excess_mean - 2 * excess_se > 0,
        "gradient_plus_2se_lt_zero": gradient_mean + 2 * gradient_se < 0,
        "negative_gradient_fraction_ge_0_75": negative_fraction >= 0.75,
        "finite_difference_plus_2se_lt_zero": slope_mean + 2 * slope_se < 0,
        "autograd_finite_difference_sign_agrees": gradient_mean * slope_mean > 0,
    }
    return {
        "layer": layer,
        "alignment_cosine_sq_mean": alignment_mean,
        "alignment_cosine_sq_se": alignment_se,
        "shifted_null_cosine_sq_mean": shifted_mean,
        "shifted_null_cosine_sq_se": shifted_se,
        "alignment_excess_mean": excess_mean,
        "alignment_excess_se": excess_se,
        "gate_gradient_mean": gradient_mean,
        "gate_gradient_se": gradient_se,
        "negative_gradient_fraction": negative_fraction,
        "finite_difference_slope_mean": slope_mean,
        "finite_difference_slope_se": slope_se,
        "checks": checks,
        "passes": all(checks.values()),
    }


def parameters_equal_checkpoint(model, checkpoint_model):
    for name, parameter in model.state_dict().items():
        reference = checkpoint_model[name]
        if not torch.equal(parameter.detach().cpu(), reference):
            return False, name
    return True, None


def analyze_checkpoint(
    label,
    checkpoint_path,
    input_bin,
    num_batches,
    finite_difference_batches,
    finite_difference_gate,
    expected_commit,
):
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, label, expected_commit)
    config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.train().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hooks = RCAHooks(model)
    batch_records = []
    finite_batches = []
    for batch_index, (x_cpu, y_cpu) in enumerate(
        checkpoint_batches(checkpoint, input_bin, num_batches)
    ):
        if batch_index < finite_difference_batches:
            finite_batches.append((x_cpu.clone(), y_cpu.clone()))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        hooks.gates.data.zero_()
        hooks.gates.grad = None
        hooks.collect_metrics = True
        hooks.clear_batch()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, return_logits=False)
        loss.backward()
        if hooks.gates.grad is None:
            raise RuntimeError("gate gradients were not produced")
        layer_metrics = [hooks.current_metrics[layer] for layer in range(config.n_layer)]
        gradients = hooks.gates.grad.detach().float().cpu().tolist()
        batch_records.append(
            {
                "batch": batch_index,
                "loss": loss.item(),
                "gate_gradients": gradients,
                "layers": layer_metrics,
            }
        )
        print(f"{label}: gradient batch {batch_index + 1}/{num_batches}", flush=True)

    finite_records = []
    hooks.collect_metrics = False
    gate_value = float(finite_difference_gate)
    gate_delta = 2.0 * math.tanh(gate_value)
    for batch_index, (x_cpu, y_cpu) in enumerate(finite_batches):
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        slopes = []
        for layer in range(config.n_layer):
            hooks.gates.data.zero_()
            hooks.gates.data[layer] = gate_value
            hooks.clear_batch()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                _, plus_loss = model(x, y, return_logits=False)
            hooks.gates.data[layer] = -gate_value
            hooks.clear_batch()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                _, minus_loss = model(x, y, return_logits=False)
            slopes.append((plus_loss.item() - minus_loss.item()) / gate_delta)
        finite_records.append({"batch": batch_index, "slopes": slopes})
        print(
            f"{label}: finite-difference batch {batch_index + 1}/{len(finite_batches)}",
            flush=True,
        )

    hooks.gates.data.zero_()
    hooks.remove()
    unchanged, changed_parameter = parameters_equal_checkpoint(model, checkpoint["model"])
    layers = [
        aggregate_layer(batch_records, finite_records, layer)
        for layer in range(config.n_layer)
    ]
    passed_layers = [entry["layer"] for entry in layers if entry["passes"]]
    result = {
        "label": label,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_next_step": checkpoint["next_step"],
        "checkpoint_args": checkpoint["args"],
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "parameters_unchanged": unchanged,
        "changed_parameter": changed_parameter,
        "num_batches": num_batches,
        "finite_difference_batches": finite_difference_batches,
        "finite_difference_gate": finite_difference_gate,
        "batch_records": batch_records,
        "finite_difference_records": finite_records,
        "layers": layers,
        "passed_layers": passed_layers,
        "checkpoint_decision": "pass" if passed_layers and unchanged else "kill",
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not unchanged:
        result["checkpoint_decision"] = "invalid"
    del hooks, model, checkpoint
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
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        help="checkpoint as LABEL=PATH; pass proxy and full",
    )
    parser.add_argument(
        "--input-bin", default="data/fineweb10B/fineweb_train_*.bin"
    )
    parser.add_argument("--output", default="runs/exp009-rca-gate/summary.json")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--finite-difference-batches", type=int, default=4)
    parser.add_argument("--finite-difference-gate", type=float, default=0.05)
    parser.add_argument("--expected-checkpoint-commit", default=BASELINE_COMMIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exp009")
    if not args.checkpoint:
        parser.error("pass both --checkpoint proxy=... and --checkpoint full=...")
    labels = [label for label, _ in args.checkpoint]
    if labels != ["proxy", "full"]:
        parser.error("checkpoint order must be proxy, then full")
    if args.num_batches < 2:
        parser.error("--num-batches must be at least 2")
    if not 1 <= args.finite_difference_batches <= args.num_batches:
        parser.error("finite-difference batches must be within num-batches")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    results = []
    for label, path in args.checkpoint:
        results.append(
            analyze_checkpoint(
                label,
                path,
                args.input_bin,
                args.num_batches,
                args.finite_difference_batches,
                args.finite_difference_gate,
                args.expected_checkpoint_commit,
            )
        )
    result_by_label = {result["label"]: result for result in results}
    shared_layers = sorted(
        set(result_by_label["proxy"]["passed_layers"])
        & set(result_by_label["full"]["passed_layers"])
    )
    any_invalid = any(
        result["checkpoint_decision"] == "invalid" for result in results
    )
    if any_invalid:
        decision = "invalid"
    elif shared_layers:
        decision = "pass"
    elif result_by_label["full"]["passed_layers"]:
        decision = "weak_inconclusive"
    else:
        decision = "kill"
    payload = {
        "experiment": "exp009",
        "measurement": "residual-complement-attention-offline-gate",
        "decision": decision,
        "shared_passed_layers": shared_layers,
        "checkpoints": results,
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "source": git_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "experiment": payload["experiment"],
        "decision": decision,
        "shared_passed_layers": shared_layers,
        "output": str(Path(args.output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
