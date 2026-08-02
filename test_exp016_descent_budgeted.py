import copy
import unittest
from types import SimpleNamespace

import torch

from descent_budgeted_adamw import (
    SPECTRAL_STATE_KEY,
    AttentionVDescentBudgetedCap,
    descent_budgeted_direction,
    preconditioned_adamw_update,
    rank_two_svd,
    synthetic_self_test,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_embd=2)
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.c_attn = torch.nn.Linear(2, 6, bias=False)
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([block])


class DescentBudgetedTest(unittest.TestCase):
    def test_synthetic_rule(self):
        synthetic_self_test()

    def test_budget_and_zero_gap(self):
        matrix = torch.diag(torch.tensor([8.0, 3.0, 2.0]))
        gradient = torch.diag(torch.tensor([3.0, 1.0, 1.0]))
        values, left, right = rank_two_svd(matrix, power_iterations=16, seed=2)
        treated, evidence = descent_budgeted_direction(
            matrix, gradient, values, left, right, 0.99
        )
        self.assertGreater(evidence["alpha"].item(), 0)
        self.assertGreaterEqual(evidence["descent_retention"].item(), 0.99 - 1e-6)
        self.assertLessEqual(torch.linalg.vector_norm(treated), torch.linalg.vector_norm(matrix))

    def test_only_v_rows_change_and_state_resumes(self):
        torch.manual_seed(9)
        treatment_model = TinyModel()
        control_model = copy.deepcopy(treatment_model)
        treatment_parameter = treatment_model.transformer.h[0].attn.c_attn.weight
        control_parameter = control_model.transformer.h[0].attn.c_attn.weight
        treatment_optimizer = torch.optim.AdamW(
            treatment_model.parameters(), lr=0.01, weight_decay=0.1, betas=(0.9, 0.95)
        )
        control_optimizer = torch.optim.AdamW(
            control_model.parameters(), lr=0.01, weight_decay=0.1, betas=(0.9, 0.95)
        )
        gradient = torch.tensor(
            [[1.0, 0.5], [0.2, -0.1], [0.3, 0.7], [-0.2, 0.4], [2.0, 0.2], [0.1, 0.8]]
        )
        treatment_parameter.grad = gradient.clone()
        control_parameter.grad = gradient.clone()
        capper = AttentionVDescentBudgetedCap(
            treatment_model,
            treatment_optimizer,
            bootstrap_iterations=12,
            tracking_iterations=1,
        )
        diagnostics = capper.prepare()
        self.assertGreaterEqual(diagnostics[0]["descent_retention"], 0.99 - 1e-6)
        treatment_optimizer.step()
        capper.apply()
        control_optimizer.step()
        torch.testing.assert_close(
            treatment_parameter[:4], control_parameter[:4], rtol=0, atol=0
        )
        self.assertIn(SPECTRAL_STATE_KEY, treatment_optimizer.state[treatment_parameter])

        saved = copy.deepcopy(treatment_optimizer.state_dict())
        resumed_model = copy.deepcopy(treatment_model)
        resumed_optimizer = torch.optim.AdamW(
            resumed_model.parameters(), lr=0.01, weight_decay=0.1, betas=(0.9, 0.95)
        )
        resumed_optimizer.load_state_dict(saved)
        resumed_parameter = resumed_model.transformer.h[0].attn.c_attn.weight
        torch.testing.assert_close(
            resumed_optimizer.state[resumed_parameter][SPECTRAL_STATE_KEY],
            treatment_optimizer.state[treatment_parameter][SPECTRAL_STATE_KEY],
        )

        next_gradient = torch.flip(gradient, dims=[0])
        treatment_parameter.grad = next_gradient.clone()
        resumed_parameter.grad = next_gradient.clone()
        treatment_capper = AttentionVDescentBudgetedCap(
            treatment_model, treatment_optimizer
        )
        resumed_capper = AttentionVDescentBudgetedCap(resumed_model, resumed_optimizer)
        treatment_capper.prepare()
        resumed_capper.prepare()
        treatment_optimizer.step()
        resumed_optimizer.step()
        treatment_capper.apply()
        resumed_capper.apply()
        torch.testing.assert_close(treatment_parameter, resumed_parameter, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
