import copy
import unittest
from types import SimpleNamespace

import torch

from spectral_cap import (
    SPECTRAL_STATE_KEY,
    AttentionVSpectralCap,
    cap_leading_gap,
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


class SpectralCapTest(unittest.TestCase):
    def test_synthetic_cap(self):
        synthetic_self_test()

    def test_only_v_rows_change_relative_to_adamw(self):
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

        update = preconditioned_adamw_update(
            treatment_parameter, treatment_optimizer
        )[4:6]
        values, left, right = rank_two_svd(
            update, power_iterations=12, seed=11000
        )
        expected_capped, _removed, _scale = cap_leading_gap(
            update, values, left, right
        )
        before = treatment_parameter.detach().clone()

        capper = AttentionVSpectralCap(
            treatment_model,
            treatment_optimizer,
            bootstrap_iterations=12,
            tracking_iterations=1,
        )
        capper.prepare()
        treatment_optimizer.step()
        capper.apply()
        control_optimizer.step()

        torch.testing.assert_close(
            treatment_parameter[:4], control_parameter[:4], rtol=0, atol=0
        )
        expected_v = before[4:6] * (1.0 - 0.01 * 0.1) - 0.01 * expected_capped
        torch.testing.assert_close(
            treatment_parameter[4:6], expected_v, rtol=2e-5, atol=2e-5
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
        treatment_capper = AttentionVSpectralCap(treatment_model, treatment_optimizer)
        resumed_capper = AttentionVSpectralCap(resumed_model, resumed_optimizer)
        treatment_capper.prepare()
        resumed_capper.prepare()
        treatment_optimizer.step()
        resumed_optimizer.step()
        treatment_capper.apply()
        resumed_capper.apply()
        torch.testing.assert_close(
            treatment_parameter, resumed_parameter, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
