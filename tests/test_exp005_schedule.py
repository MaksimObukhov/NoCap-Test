import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    original_argv_zero = sys.argv[0]
    try:
        sys.argv[0] = str(path)
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = original_argv_zero
    return module


trainer = import_from_path(
    "train_gpt2_for_exp005_test", REPOSITORY_ROOT / "train_gpt2.py"
)
gate = import_from_path(
    "summarize_exp005_gate_for_test",
    REPOSITORY_ROOT / "summarize_exp005_gate.py",
)


class SequenceLengthScheduleTest(unittest.TestCase):
    def test_proxy_schedule_preserves_tokens_and_transition(self):
        stages = trainer.build_train_shape_stages(
            num_iterations=1788,
            initial_batch_size=32,
            initial_sequence_length=512,
            transition_step=896,
            final_batch_size=16,
            final_sequence_length=1024,
        )

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0].tokens_per_micro_batch, 16_384)
        self.assertEqual(stages[1].tokens_per_micro_batch, 16_384)
        self.assertEqual(
            trainer.train_shape_stage_for_step(stages, 895, 1788).name,
            "short",
        )
        self.assertEqual(
            trainer.train_shape_stage_for_step(stages, 896, 1788).name,
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

    def test_loader_shape_change_rejects_token_count_change(self):
        loader = object.__new__(trainer.DistributedDataLoader)
        loader.B = 32
        loader.T = 512

        loader.set_batch_shape(16, 1024)
        self.assertEqual((loader.B, loader.T), (16, 1024))
        with self.assertRaisesRegex(ValueError, "preserve"):
            loader.set_batch_shape(16, 512)

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve"):
            trainer.build_train_shape_stages(
                num_iterations=1788,
                initial_batch_size=32,
                initial_sequence_length=512,
                transition_step=896,
                final_batch_size=16,
                final_sequence_length=512,
            )

    def test_gate_projection_passes_expected_profiler_scenario(self):
        fixed_summary = {
            "status": "complete",
            "train_shape_stages": [
                {
                    "name": "fixed",
                    "first_step_time_ms": 67_500,
                    "steady_median_step_time_ms": 4_000,
                }
            ],
        }
        scheduled_summary = {
            "status": "complete",
            "train_shape_stages": [
                {
                    "name": "short",
                    "first_step_time_ms": 67_500,
                    "steady_median_step_time_ms": 3_850,
                },
                {
                    "name": "long",
                    "first_step_time_ms": 64_000,
                    "steady_median_step_time_ms": 4_030,
                },
            ],
        }

        result = gate.analyse_gate(fixed_summary, scheduled_summary)
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["steady_t512_speedup"], 0.03)
        self.assertGreater(result["projected_net_speedup"], 0.01)


if __name__ == "__main__":
    unittest.main()
