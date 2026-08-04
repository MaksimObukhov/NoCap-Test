import re
import unittest
from pathlib import Path

from exp024_schedule import (
    MILESTONE_INTERVAL_TOKENS,
    MILESTONE_START_TOKENS,
    PHASES,
    SAVE_INTERVAL_TOKENS,
    TARGET_TOKENS,
    TOKEN_CLOCK_BATCH,
    VALIDATION_INTERVAL_TOKENS,
    WARMDOWN_TOKENS,
    WARMUP_TOKENS,
    accumulation_steps_at,
    validate_schedule,
)


RUN_SCRIPT = Path(__file__).with_name("run.sh").read_text()


class Exp024ScheduleTests(unittest.TestCase):
    def test_preregistered_phase_accounting(self):
        records = validate_schedule()
        self.assertEqual(
            [record["effective_batch_tokens"] for record in records],
            [65_536, 131_072, 262_144, 524_288],
        )
        self.assertEqual(
            [record["updates"] for record in records], [3_072, 2_048, 1_792, 3_358]
        )
        self.assertEqual(records[-1]["cumulative_updates"], 10_270)
        self.assertEqual(records[-1]["end_tokens"], TARGET_TOKENS)

    def test_every_transition_is_exact(self):
        boundaries = [boundary for boundary, _accumulation in PHASES]
        probes = (
            (0, 4),
            (boundaries[0] - 65_536, 4),
            (boundaries[0], 8),
            (boundaries[1], 16),
            (boundaries[2], 32),
            (TARGET_TOKENS - 524_288, 32),
        )
        for tokens_seen, expected in probes:
            self.assertEqual(accumulation_steps_at(tokens_seen), expected)

    def test_full_token_clock_hits_every_registered_artifact_boundary(self):
        tokens_seen = 0
        updates = 0
        last_validation_tokens = -1
        validations = []
        latest_boundaries = []
        milestones = []
        phase_boundaries = []
        registered_phases = {boundary for boundary, _accumulation in PHASES[:-1]}
        while True:
            if (
                last_validation_tokens < 0
                or tokens_seen // VALIDATION_INTERVAL_TOKENS
                > last_validation_tokens // VALIDATION_INTERVAL_TOKENS
                or tokens_seen == TARGET_TOKENS
            ):
                validations.append(tokens_seen)
                last_validation_tokens = tokens_seen
            if tokens_seen == TARGET_TOKENS:
                break
            previous = tokens_seen
            tokens_seen += 16_384 * accumulation_steps_at(tokens_seen)
            updates += 1
            if tokens_seen // SAVE_INTERVAL_TOKENS > previous // SAVE_INTERVAL_TOKENS:
                latest_boundaries.append(tokens_seen)
            if tokens_seen >= MILESTONE_START_TOKENS:
                if previous < MILESTONE_START_TOKENS or (
                    (tokens_seen - MILESTONE_START_TOKENS)
                    // MILESTONE_INTERVAL_TOKENS
                    > (previous - MILESTONE_START_TOKENS)
                    // MILESTONE_INTERVAL_TOKENS
                ):
                    milestones.append(tokens_seen)
            if tokens_seen in registered_phases:
                phase_boundaries.append(tokens_seen)

        self.assertEqual(updates, 10_270)
        self.assertEqual(phase_boundaries, sorted(registered_phases))
        self.assertEqual(len(milestones), 9)
        self.assertEqual(milestones[0], MILESTONE_START_TOKENS)
        self.assertEqual(validations[0], 0)
        self.assertEqual(validations[-1], TARGET_TOKENS)
        self.assertTrue(registered_phases.issubset(validations))
        self.assertGreater(len(latest_boundaries), 0)

    def test_token_clock_preserves_exp021_wsd_spans(self):
        self.assertEqual(553 * TOKEN_CLOCK_BATCH, WARMUP_TOKENS)
        self.assertEqual(2_212 * TOKEN_CLOCK_BATCH, WARMDOWN_TOKENS)
        self.assertEqual(256 * TOKEN_CLOCK_BATCH, VALIDATION_INTERVAL_TOKENS)
        self.assertEqual(8_088 * TOKEN_CLOCK_BATCH, MILESTONE_START_TOKENS)
        self.assertEqual(256 * TOKEN_CLOCK_BATCH, MILESTONE_INTERVAL_TOKENS)
        self.assertEqual(10_300 * TOKEN_CLOCK_BATCH, TARGET_TOKENS)

    def test_launcher_is_full_seed_zero_and_wandb_enabled_by_default(self):
        self.assertIn('"$MODE" != "full"', RUN_SCRIPT)
        self.assertIn('"$SEED" != "0"', RUN_SCRIPT)
        self.assertIn('WANDB_ENABLED:-1', RUN_SCRIPT)
        self.assertIn('exp024 requires W&B logging', RUN_SCRIPT)
        self.assertIn("--wandb_checkpoint_artifact", RUN_SCRIPT)
        self.assertIn("--exp024_staircase", RUN_SCRIPT)
        self.assertIn("--token_clock_batch_tokens 262144", RUN_SCRIPT)
        self.assertEqual(
            int(re.search(r"NUM_ITERATIONS=([0-9]+)", RUN_SCRIPT).group(1)), 10_300
        )
        self.assertIn('only an absolute --resume PATH', RUN_SCRIPT)


if __name__ == "__main__":
    unittest.main()
