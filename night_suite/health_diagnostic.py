#!/usr/bin/env python3
"""Short, deliberately untimed gradient-health diagnostic for a treatment branch."""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import torch

from train_gpt2 import DistributedDataLoader, GPT, GPTConfig


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def git_metadata():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked_dirty = (
        subprocess.call(["git", "diff", "--quiet"]) != 0
        or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0
    )
    return commit, tracked_dirty


def squared_norm(tensors):
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    elements = 0
    maximum = torch.zeros((), device="cuda", dtype=torch.float32)
    nonfinite = torch.zeros((), device="cuda", dtype=torch.int64)
    for tensor in tensors:
        if tensor is None:
            continue
        value = tensor.detach()
        finite = torch.isfinite(value)
        nonfinite += (~finite).sum()
        safe = torch.where(finite, value, torch.zeros_like(value)).float()
        total += safe.double().square().sum()
        maximum = torch.maximum(maximum, safe.abs().max())
        elements += value.numel()
    return total, elements, maximum, nonfinite


def layer_index(name):
    prefix = "transformer.h."
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    head = remainder.split(".", 1)[0]
    return int(head) if head.isdigit() else None


@torch.no_grad()
def parameter_norms(named_parameters):
    total, elements, _maximum, nonfinite = squared_norm(
        parameter for _name, parameter in named_parameters
    )
    return math.sqrt(total.item()), elements, int(nonfinite.item())


def gradient_metrics(named_parameters):
    pairs = [(name, parameter.grad) for name, parameter in named_parameters]
    total, elements, maximum, nonfinite = squared_norm(
        gradient for _name, gradient in pairs
    )
    layers = {}
    for name, gradient in pairs:
        index = layer_index(name)
        if index is None or gradient is None:
            continue
        entry = layers.setdefault(index, {"sum_sq": 0.0, "elements": 0})
        safe = torch.where(torch.isfinite(gradient), gradient, torch.zeros_like(gradient))
        entry["sum_sq"] += safe.float().square().sum().item()
        entry["elements"] += gradient.numel()
    return {
        "global_grad_norm": math.sqrt(total.item()),
        "grad_rms": math.sqrt(total.item() / max(elements, 1)),
        "max_abs_grad": maximum.item(),
        "nonfinite_grad_count": int(nonfinite.item()),
        "per_layer_grad_rms": {
            str(index): math.sqrt(entry["sum_sq"] / max(entry["elements"], 1))
            for index, entry in sorted(layers.items())
        },
    }


@torch.no_grad()
def update_metrics(named_parameters, before):
    update_sq = 0.0
    weight_sq = 0.0
    per_layer = {}
    for (name, parameter), old_value in zip(named_parameters, before):
        delta = parameter.detach().float() - old_value.float()
        update_sq += delta.square().sum().item()
        weight_sq += old_value.float().square().sum().item()
        index = layer_index(name)
        if index is None:
            continue
        entry = per_layer.setdefault(index, {"update_sq": 0.0, "weight_sq": 0.0})
        entry["update_sq"] += delta.square().sum().item()
        entry["weight_sq"] += old_value.float().square().sum().item()
    return {
        "update_weight_ratio": math.sqrt(update_sq / max(weight_sq, 1e-30)),
        "per_layer_update_weight_ratio": {
            str(index): math.sqrt(entry["update_sq"] / max(entry["weight_sq"], 1e-30))
            for index, entry in sorted(per_layer.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-project", default="nocap-baseline")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--num-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--grad-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0018)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--warmup-updates", type=int, default=96)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--update-ratio-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--descent-budgeted-attn-v",
        action="store_true",
        help="enable the exp016 optimizer treatment when its module is present",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(
        args.num_updates,
        args.batch_size,
        args.sequence_length,
        args.grad_accumulation_steps,
        args.update_ratio_every,
    ) <= 0:
        parser.error("update and shape arguments must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "gradient_diagnostics.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    commit, tracked_dirty = git_metadata()
    if tracked_dirty:
        raise RuntimeError("tracked worktree changes invalidate the diagnostic")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)

    config = {"args": vars(args), "git_commit": commit, "git_dirty": tracked_dirty}
    write_json(output_dir / "config.json", config)
    shutil.copy2(__file__, output_dir / "health_diagnostic.py")
    shutil.copy2("train_gpt2.py", output_dir / "train_gpt2.py")

    wandb_run = None
    if args.log_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            job_type="health-diagnostic",
            config=config,
        )

    loader = DistributedDataLoader(
        args.input_bin,
        args.batch_size,
        args.sequence_length,
        0,
        1,
    )
    x, y = loader.next_batch()
    base_model = GPT(GPTConfig()).train().cuda()
    named_parameters = list(base_model.named_parameters())
    model = torch.compile(base_model)
    optimizer = base_model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (0.9, 0.95),
        "cuda",
    )
    spectral_treatment = None
    if args.descent_budgeted_attn_v:
        from descent_budgeted_adamw import AttentionVDescentBudgetedCap

        spectral_treatment = AttentionVDescentBudgetedCap(base_model, optimizer)
    context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    previous_loss = None
    grad_norms = []
    loss_jumps = []
    nonfinite_total = 0
    started = time.perf_counter()

    for update in range(args.num_updates):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _micro_step in range(args.grad_accumulation_steps):
            with context:
                _logits, loss = model(x, y, return_logits=False)
                loss = loss / args.grad_accumulation_steps
            train_loss += loss.detach().float().item()
            x, y = loader.next_batch()
            loss.backward()

        learning_rate = args.learning_rate * args.lr_scale
        if update < args.warmup_updates:
            learning_rate *= (update + 1) / args.warmup_updates
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        record = {
            "event": "gradient_health",
            "step": update + 1,
            "tokens_seen": (update + 1)
            * args.batch_size
            * args.sequence_length
            * args.grad_accumulation_steps,
            "train_loss": train_loss,
            "loss_jump": 0.0 if previous_loss is None else train_loss - previous_loss,
            "learning_rate": learning_rate,
            **gradient_metrics(named_parameters),
        }
        before = None
        if (update + 1) % args.update_ratio_every == 0:
            before = [parameter.detach().clone() for _name, parameter in named_parameters]
        if spectral_treatment is not None:
            spectral_treatment.prepare()
        optimizer.step()
        if spectral_treatment is not None:
            spectral_treatment.apply()
        if before is not None:
            record.update(update_metrics(named_parameters, before))
        optimizer.zero_grad(set_to_none=True)

        parameter_norm, parameter_elements, parameter_nonfinite = parameter_norms(
            named_parameters
        )
        record.update(
            {
                "parameter_norm": parameter_norm,
                "parameter_elements": parameter_elements,
                "nonfinite_parameter_count": parameter_nonfinite,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        append_jsonl(diagnostics_path, record)
        append_jsonl(metrics_path, record)
        if wandb_run is not None:
            wandb_run.log(record, step=update + 1)
        print(
            f"health {update + 1}/{args.num_updates} loss={train_loss:.6f} "
            f"grad={record['global_grad_norm']:.6g} "
            f"max={record['max_abs_grad']:.6g}",
            flush=True,
        )
        previous_loss = train_loss
        grad_norms.append(record["global_grad_norm"])
        if update > 0:
            loss_jumps.append(abs(record["loss_jump"]))
        nonfinite_total += record["nonfinite_grad_count"] + parameter_nonfinite

    grad_median = statistics.median(grad_norms)
    jump_median = statistics.median(loss_jumps) if loss_jumps else 0.0
    grad_spikes = sum(value > 10.0 * max(grad_median, 1e-30) for value in grad_norms)
    jump_spikes = sum(value > 10.0 * max(jump_median, 1e-30) for value in loss_jumps)
    pathological = grad_spikes >= 2 or jump_spikes >= 2
    decision = "pass" if nonfinite_total == 0 and not pathological else "kill"
    summary = {
        "status": "complete",
        "experiment_stage": "health-diagnostic",
        "decision": decision,
        "git_commit": commit,
        "git_dirty": tracked_dirty,
        "num_updates": args.num_updates,
        "tokens_seen": args.num_updates
        * args.batch_size
        * args.sequence_length
        * args.grad_accumulation_steps,
        "median_global_grad_norm": grad_median,
        "max_global_grad_norm": max(grad_norms),
        "gradient_spikes_over_10x_median": grad_spikes,
        "median_absolute_loss_jump": jump_median,
        "loss_spikes_over_10x_median": jump_spikes,
        "nonfinite_count": nonfinite_total,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "gate_decision.json",
        {"stage": "health-diagnostic", "decision": decision, "evidence": summary},
    )
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
