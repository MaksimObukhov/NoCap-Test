import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("fp8_integration_benchmark.py")
SPEC = importlib.util.spec_from_file_location("fp8_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FP8IntegrationTests(unittest.TestCase):
    def test_padded_head_exposes_only_true_vocab(self):
        linear = torch.nn.Linear(8, MODULE.PADDED_VOCAB_SIZE, bias=False)
        head = MODULE.TrueVocabHead(linear)
        output = head(torch.randn(2, 3, 8))
        self.assertEqual(output.shape, (2, 3, MODULE.TRUE_VOCAB_SIZE))
        self.assertIs(head.weight, linear.weight)

    def test_registered_vocab_padding(self):
        self.assertEqual(MODULE.TRUE_VOCAB_SIZE, 50_257)
        self.assertEqual(MODULE.PADDED_VOCAB_SIZE, 50_304)
        self.assertEqual(MODULE.PADDED_VOCAB_SIZE % 16, 0)


if __name__ == "__main__":
    unittest.main()
