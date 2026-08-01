#!/usr/bin/env python3
"""Frozen-weight AdamW update-spectrum falsifier for exp010."""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
import zlib
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
    required = (
        "model",
        "optimizer",
        "next_x",
        "next_y",
        "train_loader",
        "next_step",
    )
    for key in required:
        if key not in checkpoint:
            errors.append(f"missing checkpoint key: {key}")
    if errors:
        raise ValueError(f"invalid {label} checkpoint: " + "; ".join(errors))


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


def top_singular_subspace(matrix, rank=8, power_iterations=6, seed=0):
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank two")
    rows, columns = matrix.shape
    rank = min(rank, rows, columns)
    if rank < 2:
        raise ValueError("matrix is too small for a leading and second mode")
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(seed)
    right = torch.randn(
        columns, rank, device=matrix.device, dtype=torch.float32, generator=generator
    )
    right = torch.linalg.qr(right, mode="reduced").Q
    for _ in range(power_iterations):
        left = torch.linalg.qr(matrix @ right, mode="reduced").Q
        right = torch.linalg.qr(matrix.T @ left, mode="reduced").Q
    compressed = matrix @ right
    _left, singular_values, small_vh = torch.linalg.svd(
        compressed, full_matrices=False
    )
    right_vectors = right @ small_vh.T
    return singular_values, right_vectors


def spectral_metrics(matrix, rank, power_iterations, seed):
    matrix = matrix.float()
    singular_values, right_vectors = top_singular_subspace(
        matrix, rank=rank, power_iterations=power_iterations, seed=seed
    )
    frobenius_sq = matrix.square().sum().item()
    sigma1 = singular_values[0].item()
    sigma2 = singular_values[1].item()
    leading_energy = sigma1 * sigma1 / max(frobenius_sq, 1e-30)
    rows, columns = matrix.shape
    isotropic_null = (math.sqrt(rows) + math.sqrt(columns)) ** 2 / (rows * columns)
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "frobenius_norm": math.sqrt(frobenius_sq),
        "leading_energy": leading_energy,
        "stable_rank": frobenius_sq / max(sigma1 * sigma1, 1e-30),
        "cap_fraction": (sigma1 - sigma2) / max(math.sqrt(frobenius_sq), 1e-30),
        "isotropic_null_leading_energy": isotropic_null,
        "leading_energy_over_null": leading_energy / isotropic_null,
    }, right_vectors


def subspace_overlap(previous, current):
    rank = min(previous.shape[1], current.shape[1])
    if rank == 0:
        return None
    cross = previous[:, :rank].T @ current[:, :rank]
    return cross.square().sum().item() / rank


def self_test():
    torch.manual_seed(13)
    matrix = torch.diag(torch.tensor([9.0, 4.0, 2.0, 1.0, 0.5]))
    metrics, vectors = spectral_metrics(matrix, rank=4, power_iterations=12, seed=4)
    assert abs(metrics["sigma1"] - 9.0) < 1e-4
    assert abs(metrics["sigma2"] - 4.0) < 1e-4
    weak = vectors[:, 1:]
    assert abs(subspace_overlap(weak, weak) - 1.0) < 1e-5
    expected_energy = 81.0 / matrix.square().sum().item()
    assert abs(metrics["leading_energy"] - expected_energy) < 1e-5

    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.randn_like(parameter)
    predicted = preconditioned_adamw_update(parameter, optimizer)
    before = parameter.detach().clone()
    learning_rate = 1e-3
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.step()
    observed = (before - parameter.detach()) / learning_rate
    torch.testing.assert_close(predicted, observed, rtol=2e-4, atol=2e-4)
    print("exp010 self-test passed")


class InputCapture:
    def __init__(self, modules, max_rows):
        self.max_rows = max_rows
        self.enabled = False
        self.inputs = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_module, inputs):
            if not self.enabled or name in self.inputs:
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
            self.inputs[name] = values.float().cpu()

        return hook

    def reset(self):
        self.inputs.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def target_specs(model):
    modules = {}
    specs_by_parameter = defaultdict(list)
    width = model.config.n_embd
    for layer, block in enumerate(model.transformer.h):
        entries = (
            ("attn.c_attn", block.attn.c_attn, None),
            ("attn.c_proj", block.attn.c_proj, "attn_out"),
            ("mlp.c_fc", block.mlp.c_fc, "mlp_up"),
            ("mlp.c_proj", block.mlp.c_proj, "mlp_down"),
        )
        for suffix, module, family in entries:
            module_name = f"transformer.h.{layer}.{suffix}"
            modules[module_name] = module
            if suffix == "attn.c_attn":
                for part, part_family in enumerate(("attn_q", "attn_k", "attn_v")):
                    specs_by_parameter[id(module.weight)].append(
                        {
                            "name": f"{module_name}.{part_family[-1]}",
                            "family": part_family,
                            "layer": layer,
                            "module_name": module_name,
                            "parameter": module.weight,
                            "row_start": part * width,
                            "row_end": (part + 1) * width,
                        }
                    )
            else:
                specs_by_parameter[id(module.weight)].append(
                    {
                        "name": module_name,
                        "family": family,
                        "layer": layer,
                        "module_name": module_name,
                        "parameter": module.weight,
                        "row_start": 0,
                        "row_end": module.weight.shape[0],
                    }
                )
    return modules, specs_by_parameter


def preconditioned_adamw_update(parameter, optimizer):
    state = optimizer.state.get(parameter)
    if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
        raise RuntimeError("missing AdamW moment state for a measured parameter")
    if parameter.grad is None:
        raise RuntimeError("missing gradient for a measured parameter")
    group = next(
        group
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    )
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise RuntimeError("the diagnostic supports baseline AdamW only")
    beta1, beta2 = group["betas"]
    step_value = state["step"]
    step = int(step_value.item() if torch.is_tensor(step_value) else step_value) + 1
    first = state["exp_avg"] * beta1 + parameter.grad * (1.0 - beta1)
    second = state["exp_avg_sq"] * beta2 + parameter.grad.square() * (1.0 - beta2)
    first = first / (1.0 - beta1**step)
    second = second / (1.0 - beta2**step)
    return first / (second.sqrt() + group["eps"])


def parameters_equal_checkpoint(model, checkpoint_model):
    for name, parameter in model.state_dict().items():
        if not torch.equal(parameter.detach().cpu(), checkpoint_model[name]):
            return False, name
    return True, None


def family_aggregates(update_records):
    grouped = defaultdict(list)
    for update in update_records:
        for matrix in update["matrices"]:
            grouped[matrix["family"]].append(matrix)
    aggregates = []
    for family, records in sorted(grouped.items()):
        cap_fraction = statistics.median(record["cap_fraction"] for record in records)
        functional_energy = statistics.median(
            record["functional_leading_energy"] for record in records
        )
        overlaps = [
            record["weak_subspace_overlap"]
            for record in records
            if record["weak_subspace_overlap"] is not None
        ]
        overlap = statistics.median(overlaps) if overlaps else 0.0
        random_expectation = statistics.median(
            record["random_subspace_overlap_expectation"] for record in records
        )
        overlap_threshold = max(0.05, 3.0 * random_expectation)
        checks = {
            "median_cap_fraction_ge_0_05": cap_fraction >= 0.05,
            "median_functional_leading_energy_ge_0_10": functional_energy >= 0.10,
            "median_weak_subspace_overlap_ge_threshold": overlap >= overlap_threshold,
        }
        aggregates.append(
            {
                "family": family,
                "matrix_records": len(records),
                "overlap_records": len(overlaps),
                "median_cap_fraction": cap_fraction,
                "median_functional_leading_energy": functional_energy,
                "median_weak_subspace_overlap": overlap,
                "median_random_subspace_overlap_expectation": random_expectation,
                "overlap_threshold": overlap_threshold,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    return aggregates


def analyze_checkpoint(
    label,
    checkpoint_path,
    input_bin,
    num_updates,
    sample_rows,
    rank,
    power_iterations,
    expected_commit,
):
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, label, expected_commit)
    config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.train().cuda()
    optimizer = model.configure_optimizers(
        weight_decay=checkpoint["args"]["weight_decay"],
        learning_rate=checkpoint["args"]["learning_rate"],
        betas=(0.9, 0.95),
        device_type="cuda",
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        group["weight_decay"] = 0.0

    modules, specs_by_parameter = target_specs(model)
    capture = InputCapture(modules, sample_rows)
    loader = DistributedDataLoader(
        input_bin,
        checkpoint["args"]["batch_size"],
        checkpoint["args"]["sequence_length"],
        process_rank=0,
        num_processes=1,
    )
    loader.load_state_dict(checkpoint["train_loader"])
    x = checkpoint["next_x"].cuda()
    y = checkpoint["next_y"].cuda()
    accumulation = checkpoint["args"]["grad_accumulation_steps"]
    previous_weak_modes = {}
    update_records = []

    for update_index in range(num_updates):
        optimizer.zero_grad(set_to_none=True)
        capture.reset()
        total_loss = 0.0
        for micro_step in range(accumulation):
            capture.enabled = micro_step == 0
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, return_logits=False)
                scaled_loss = loss / accumulation
            total_loss += scaled_loss.detach().item()
            x, y = loader.next_batch()
            scaled_loss.backward()
        capture.enabled = False
        missing_inputs = sorted(set(modules) - set(capture.inputs))
        if missing_inputs:
            raise RuntimeError(f"failed to capture module inputs: {missing_inputs}")

        matrix_records = []
        parameter_by_id = {
            id(parameter): parameter for parameter in model.parameters()
        }
        for parameter_id, specs in specs_by_parameter.items():
            parameter = parameter_by_id[parameter_id]
            update_matrix = preconditioned_adamw_update(parameter, optimizer)
            for spec in specs:
                matrix = update_matrix[spec["row_start"] : spec["row_end"]]
                seed = zlib.crc32(spec["name"].encode()) + 1009 * update_index
                metrics, right_vectors = spectral_metrics(
                    matrix, rank=rank, power_iterations=power_iterations, seed=seed
                )
                inputs = capture.inputs[spec["module_name"]].cuda(non_blocking=True)
                functional = inputs @ matrix.T
                functional_metrics, _ = spectral_metrics(
                    functional,
                    rank=rank,
                    power_iterations=power_iterations,
                    seed=seed + 1,
                )
                weak_modes = right_vectors[:, 1:rank].detach()
                previous = previous_weak_modes.get(spec["name"])
                overlap = (
                    subspace_overlap(previous, weak_modes)
                    if previous is not None
                    else None
                )
                previous_weak_modes[spec["name"]] = weak_modes
                random_expectation = weak_modes.shape[1] / weak_modes.shape[0]
                matrix_records.append(
                    {
                        "name": spec["name"],
                        "family": spec["family"],
                        "layer": spec["layer"],
                        "shape": list(matrix.shape),
                        **metrics,
                        "functional_shape": list(functional.shape),
                        "functional_leading_energy": functional_metrics[
                            "leading_energy"
                        ],
                        "functional_stable_rank": functional_metrics["stable_rank"],
                        "functional_leading_energy_over_null": functional_metrics[
                            "leading_energy_over_null"
                        ],
                        "weak_subspace_overlap": overlap,
                        "random_subspace_overlap_expectation": random_expectation,
                    }
                )
                del functional, inputs, right_vectors
            del update_matrix
        update_records.append(
            {"update": update_index, "train_loss": total_loss, "matrices": matrix_records}
        )
        optimizer.step()
        print(
            f"{label}: replay update {update_index + 1}/{num_updates}, loss {total_loss:.6f}",
            flush=True,
        )

    capture.remove()
    unchanged, changed_parameter = parameters_equal_checkpoint(model, checkpoint["model"])
    aggregates = family_aggregates(update_records)
    passed_families = [entry["family"] for entry in aggregates if entry["passes"]]
    result = {
        "label": label,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_next_step": checkpoint["next_step"],
        "checkpoint_args": checkpoint["args"],
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "checkpoint_train_loader": checkpoint["train_loader"],
        "input_bin": input_bin,
        "parameters_unchanged": unchanged,
        "changed_parameter": changed_parameter,
        "num_updates": num_updates,
        "sample_rows": sample_rows,
        "rank": rank,
        "power_iterations": power_iterations,
        "updates": update_records,
        "families": aggregates,
        "passed_families": passed_families,
        "checkpoint_decision": "pass" if passed_families and unchanged else "kill",
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not unchanged:
        result["checkpoint_decision"] = "invalid"
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
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        help="checkpoint as LABEL=PATH; pass proxy and full",
    )
    parser.add_argument(
        "--input-bin", default="data/fineweb10B/fineweb_train_*.bin"
    )
    parser.add_argument(
        "--output", default="runs/exp010-spectral-adamw-gate/summary.json"
    )
    parser.add_argument("--num-updates", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=6)
    parser.add_argument("--expected-checkpoint-commit", default=BASELINE_COMMIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exp010")
    if not args.checkpoint:
        parser.error("pass both --checkpoint proxy=... and --checkpoint full=...")
    labels = [label for label, _ in args.checkpoint]
    if labels != ["proxy", "full"]:
        parser.error("checkpoint order must be proxy, then full")
    if args.num_updates < 2 or args.sample_rows < args.rank or args.rank < 2:
        parser.error("need updates >=2, sample rows >= rank, and rank >=2")

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
                args.num_updates,
                args.sample_rows,
                args.rank,
                args.power_iterations,
                args.expected_checkpoint_commit,
            )
        )
    result_by_label = {result["label"]: result for result in results}
    shared_families = sorted(
        set(result_by_label["proxy"]["passed_families"])
        & set(result_by_label["full"]["passed_families"])
    )
    any_invalid = any(
        result["checkpoint_decision"] == "invalid" for result in results
    )
    if any_invalid:
        decision = "invalid"
    elif shared_families:
        decision = "pass"
    elif result_by_label["full"]["passed_families"]:
        decision = "weak_inconclusive"
    else:
        decision = "kill"
    payload = {
        "experiment": "exp010",
        "measurement": "selective-spectral-adamw-offline-gate",
        "decision": decision,
        "shared_passed_families": shared_families,
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
        "shared_passed_families": shared_families,
        "output": str(Path(args.output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
