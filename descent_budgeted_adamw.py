"""Descent-budgeted spectral reweighting for AdamW attention-V updates."""

import math
import statistics

import torch


SPECTRAL_STATE_KEY = "exp016_attn_v_right_subspace"


def optimizer_group_for(parameter, optimizer):
    for group in optimizer.param_groups:
        if any(candidate is parameter for candidate in group["params"]):
            return group
    raise KeyError("parameter is not present in the optimizer")


def preconditioned_adamw_update(parameter, optimizer, row_slice=None):
    """Return positive P in the baseline update p <- decay(p) - lr*P."""
    if parameter.grad is None:
        raise RuntimeError("missing gradient for spectral target")
    group = optimizer_group_for(parameter, optimizer)
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise RuntimeError("exp016 supports the baseline AdamW configuration only")
    state = optimizer.state.get(parameter, {})
    gradient = parameter.grad if row_slice is None else parameter.grad[row_slice]
    beta1, beta2 = group["betas"]
    if "exp_avg" in state:
        old_first = state["exp_avg"] if row_slice is None else state["exp_avg"][row_slice]
        old_second = (
            state["exp_avg_sq"]
            if row_slice is None
            else state["exp_avg_sq"][row_slice]
        )
        first = old_first * beta1 + gradient * (1.0 - beta1)
        second = old_second * beta2 + gradient.square() * (1.0 - beta2)
        old_step = state["step"]
        old_step = int(old_step.item() if torch.is_tensor(old_step) else old_step)
    else:
        first = gradient * (1.0 - beta1)
        second = gradient.square() * (1.0 - beta2)
        old_step = 0
    step = old_step + 1
    first = first / (1.0 - beta1**step)
    second = second / (1.0 - beta2**step)
    return first / (second.sqrt() + group["eps"])


def rank_two_svd(matrix, right=None, power_iterations=6, seed=0):
    matrix = matrix.float()
    _rows, columns = matrix.shape
    if min(matrix.shape) < 2:
        raise ValueError("rank-two decomposition requires both dimensions >= 2")
    if right is None:
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(seed)
        right = torch.randn(
            columns,
            2,
            generator=generator,
            device=matrix.device,
            dtype=torch.float32,
        )
    else:
        right = right.to(device=matrix.device, dtype=torch.float32)
    right = torch.linalg.qr(right, mode="reduced").Q
    for _ in range(power_iterations):
        left = torch.linalg.qr(matrix @ right, mode="reduced").Q
        right = torch.linalg.qr(matrix.T @ left, mode="reduced").Q
    compressed = matrix @ right
    left, singular_values, small_vh = torch.linalg.svd(compressed, full_matrices=False)
    right = right @ small_vh.T
    return singular_values, left, right


def descent_budgeted_direction(
    matrix,
    gradient,
    singular_values,
    left,
    right,
    minimum_descent_retention=0.99,
):
    """Apply the largest partial leading-gap cap inside the descent budget."""
    matrix = matrix.float()
    gradient = gradient.float()
    gap = torch.clamp(singular_values[0] - singular_values[1], min=0.0)
    full_removal = gap * torch.outer(left[:, 0], right[:, 0])
    total_descent = (gradient * matrix).sum()
    if not torch.isfinite(total_descent) or total_descent <= 0:
        raise ValueError("positive finite predicted descent is required")
    removal_descent = (gradient * full_removal).sum()
    if removal_descent <= 0:
        alpha = torch.ones((), device=matrix.device, dtype=torch.float32)
    else:
        budget = (1.0 - minimum_descent_retention) * total_descent
        alpha = torch.clamp(budget / removal_descent, min=0.0, max=1.0)
    removed = alpha * full_removal
    treated = matrix - removed
    treated_descent = (gradient * treated).sum()
    descent_retention = treated_descent / total_descent
    norm_retention = torch.linalg.vector_norm(treated) / torch.linalg.vector_norm(
        matrix
    ).clamp_min(1e-30)
    return treated, {
        "alpha": alpha,
        "gap": gap,
        "full_removal": full_removal,
        "removed": removed,
        "total_descent": total_descent,
        "removal_descent": removal_descent,
        "treated_descent": treated_descent,
        "descent_retention": descent_retention,
        "norm_retention": norm_retention,
    }


def leading_energy(matrix, power_iterations=6, seed=0):
    values, _left, _right = rank_two_svd(
        matrix, power_iterations=power_iterations, seed=seed
    )
    frobenius_sq = matrix.float().square().sum().item()
    return values[0].item() ** 2 / max(frobenius_sq, 1e-30)


class AttentionVDescentBudgetedCap:
    """Prepare the AdamW correction before step and apply it immediately after."""

    def __init__(
        self,
        model,
        optimizer,
        minimum_descent_retention=0.99,
        bootstrap_iterations=4,
        tracking_iterations=1,
    ):
        self.optimizer = optimizer
        self.width = model.config.n_embd
        self.minimum_descent_retention = minimum_descent_retention
        self.bootstrap_iterations = bootstrap_iterations
        self.tracking_iterations = tracking_iterations
        self.targets = [
            (layer, block.attn.c_attn.weight)
            for layer, block in enumerate(model.transformer.h)
        ]
        self.pending = []

    @torch.no_grad()
    def prepare(self):
        if self.pending:
            raise RuntimeError("previous spectral corrections were not applied")
        diagnostics = []
        for layer, parameter in self.targets:
            row_slice = slice(2 * self.width, 3 * self.width)
            direction = preconditioned_adamw_update(
                parameter, self.optimizer, row_slice
            )
            gradient = parameter.grad[row_slice]
            state = self.optimizer.state.get(parameter, {})
            previous_right = state.get(SPECTRAL_STATE_KEY)
            iterations = (
                self.tracking_iterations
                if previous_right is not None
                else self.bootstrap_iterations
            )
            singular_values, left, right = rank_two_svd(
                direction,
                right=previous_right,
                power_iterations=iterations,
                seed=16000 + layer,
            )
            treated, evidence = descent_budgeted_direction(
                direction,
                gradient,
                singular_values,
                left,
                right,
                self.minimum_descent_retention,
            )
            group = optimizer_group_for(parameter, self.optimizer)
            self.pending.append(
                {
                    "parameter": parameter,
                    "correction": direction - treated,
                    "right": right.detach(),
                    "learning_rate": float(group["lr"]),
                }
            )
            diagnostics.append(
                {
                    "layer": layer,
                    "sigma1": singular_values[0].item(),
                    "sigma2": singular_values[1].item(),
                    "alpha": evidence["alpha"].item(),
                    "descent_retention": evidence["descent_retention"].item(),
                    "norm_retention": evidence["norm_retention"].item(),
                    "power_iterations": iterations,
                }
            )
        return diagnostics

    @torch.no_grad()
    def apply(self):
        if len(self.pending) != len(self.targets):
            raise RuntimeError("spectral corrections were not prepared for every layer")
        for entry in self.pending:
            parameter = entry["parameter"]
            parameter[2 * self.width : 3 * self.width].add_(
                entry["correction"], alpha=entry["learning_rate"]
            )
            self.optimizer.state[parameter][SPECTRAL_STATE_KEY] = entry["right"]
        self.pending.clear()


def aggregate_diagnostics(records):
    if not records:
        return {}
    return {
        "spectral_alpha_median": statistics.median(
            record["alpha"] for record in records
        ),
        "spectral_active_fraction": sum(record["alpha"] > 0 for record in records)
        / len(records),
        "spectral_descent_retention_median": statistics.median(
            record["descent_retention"] for record in records
        ),
        "spectral_norm_retention_median": statistics.median(
            record["norm_retention"] for record in records
        ),
    }


def synthetic_self_test():
    matrix = torch.diag(torch.tensor([9.0, 4.0, 2.0, 1.0]))
    gradient = torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0]))
    values, left, right = rank_two_svd(matrix, power_iterations=12, seed=5)
    treated, evidence = descent_budgeted_direction(
        matrix, gradient, values, left, right, minimum_descent_retention=0.99
    )
    if evidence["descent_retention"].item() < 0.99 - 1e-6:
        raise AssertionError("descent budget was violated")
    if evidence["alpha"].item() <= 0:
        raise AssertionError("synthetic treatment should be active")
    if torch.linalg.vector_norm(treated) > torch.linalg.vector_norm(matrix) + 1e-6:
        raise AssertionError("treatment unexpectedly amplified the update")
    identity = torch.diag(torch.tensor([4.0, 4.0, 2.0, 1.0]))
    values, left, right = rank_two_svd(identity, power_iterations=12, seed=3)
    unchanged, evidence = descent_budgeted_direction(
        identity, torch.eye(4), values, left, right
    )
    torch.testing.assert_close(unchanged, identity, rtol=2e-4, atol=2e-4)
    if evidence["gap"].item() > 2e-3:
        raise AssertionError("zero-gap self-test did not recover identity")
    if not math.isfinite(leading_energy(matrix)):
        raise AssertionError("leading energy is not finite")
