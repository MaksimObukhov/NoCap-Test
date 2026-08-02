import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


TRAIN_PATH = Path(__file__).with_name("train_gpt2.py")
ORIGINAL_ARGV0 = sys.argv[0]
sys.argv[0] = str(TRAIN_PATH)
SPEC = importlib.util.spec_from_file_location("exp022_train", TRAIN_PATH)
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)
sys.argv[0] = ORIGINAL_ARGV0


class Exp022Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TRAIN.GPT(TRAIN.GPTConfig())

    def test_parameter_count_tying_and_fused_shape(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 109_376_256)
        self.assertIs(self.model.transformer.wte.weight, self.model.lm_head.weight)
        mlp = self.model.transformer.h[0].mlp
        self.assertEqual((mlp.c_gate_value.in_features, mlp.c_gate_value.out_features), (768, 3072))
        self.assertEqual((mlp.c_proj.in_features, mlp.c_proj.out_features), (1536, 768))

    def test_selective_weight_decay_covers_each_parameter_once(self):
        optimizer = self.model.configure_optimizers(0.1, 0.0018, (0.9, 0.95), "cpu")
        groups = {group["weight_decay"]: group["params"] for group in optimizer.param_groups}
        self.assertEqual(set(groups), {0.0, 0.1})
        self.assertEqual(groups[0.0], [self.model.transformer.wte.weight])
        parameter_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(set(parameter_ids), {id(p) for p in self.model.parameters()})
        self.assertEqual(sum(p.numel() for p in groups[0.0]), 38_597_376)
        self.assertEqual(sum(p.numel() for p in groups[0.1]), 70_778_880)

    def test_fused_projection_matches_two_projection_form(self):
        torch.manual_seed(22)
        config = TRAIN.GPTConfig(n_embd=8, n_head=2, n_layer=1, vocab_size=32)
        fused = TRAIN.MLP(config)
        gate = torch.nn.Linear(8, 16, bias=False)
        value = torch.nn.Linear(8, 16, bias=False)
        projection = torch.nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            fused.c_gate_value.weight.copy_(torch.cat((gate.weight, value.weight), dim=0))
            fused.c_proj.weight.copy_(projection.weight)
        inputs = torch.randn(2, 5, 8)
        expected = projection(F.silu(gate(inputs)) * value(inputs))
        torch.testing.assert_close(fused(inputs), expected)


if __name__ == "__main__":
    unittest.main()
