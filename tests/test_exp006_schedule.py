import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def import_trainer():
    path = REPOSITORY_ROOT / "train_gpt2.py"
    spec = importlib.util.spec_from_file_location(
        "train_gpt2_for_exp006_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    original_argv_zero = sys.argv[0]
    try:
        sys.argv[0] = str(path)
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = original_argv_zero
    return module


trainer = import_trainer()


class SequenceLengthQualityScheduleTest(unittest.TestCase):
    def test_proxy_schedule_uses_validation_aligned_early_transition(self):
        stages = trainer.build_train_shape_stages(
            num_iterations=1788,
            initial_batch_size=32,
            initial_sequence_length=512,
            transition_step=384,
            final_batch_size=16,
            final_sequence_length=1024,
        )

        self.assertEqual(384 % 128, 0)
        self.assertEqual(
            trainer.train_shape_stage_for_step(stages, 383, 1788).name,
            "short",
        )
        self.assertEqual(
            trainer.train_shape_stage_for_step(stages, 384, 1788).name,
            "long",
        )
        self.assertEqual(
            sum(
                (stage.end_step_exclusive - stage.start_step)
                * stage.tokens_per_micro_batch
                * 32
                for stage in stages
            ),
            937_426_944,
        )


if __name__ == "__main__":
    unittest.main()
