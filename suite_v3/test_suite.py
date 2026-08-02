"""Regression tests for the v3 proxy suite.

Runs on a laptop: no torch, no GPU, no network. Everything under test is
either a pure function in gates.py or a module-level helper lifted out of
train_gpt2.py, which is loaded here by extracting the function definitions
with ast rather than importing the module (importing would pull in torch).

Each test names the failure it prevents. Several encode a specific v2 defect
and would have failed against that code.

    python -m suite_v3.test_suite
"""

import ast
import json
import os
import tempfile
import unittest

from suite_v3 import gates, suite

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def read_text(path):
    with open(path) as handle:
        return handle.read()


def load_train_helpers(names):
    """Exec selected top-level functions from train_gpt2.py without torch."""
    source = read_text(os.path.join(REPO_ROOT, "train_gpt2.py"))
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in wanted}
    if missing:
        raise AssertionError(f"train_gpt2.py is missing helpers: {sorted(missing)}")
    namespace = {"os": os, "json": json}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "train_gpt2", "exec"), namespace)
    return namespace


HELPERS = load_train_helpers(
    {
        "checkpoint_destination",
        "truncate_metrics_records",
        "parse_validation_envelope",
        "envelope_limit_at",
        "training_phase",
    }
)


def train_records(specs):
    """Build minimal train records: (step, phase, grad_norm)."""
    return [
        {
            "event": "train",
            "step": step,
            "phase": phase,
            "pre_clip_grad_norm": norm,
        }
        for step, phase, norm in specs
    ]


def bench(median, batches=(524288,), updates=38):
    return {
        "updates": updates,
        "median_step_time_ms": median,
        "mean_step_time_ms": median,
        "distinct_effective_batches": list(batches),
        "total_train_seconds": 350.0,
    }


class BenchmarkGate(unittest.TestCase):
    def test_kill_stops_downstream_stages(self):
        """A killed benchmark must never be followed by a two-hour proxy."""
        decision = gates.benchmark_gate(bench(3800.0), bench(3700.0), bench(3700.0))
        self.assertEqual(decision["decision"], gates.KILL)
        self.assertFalse(gates.suite_should_continue(decision["decision"]))

    def test_warning_band_still_continues(self):
        """exp019's 1-2% band is 'note it and proceed', not a stop."""
        decision = gates.benchmark_gate(bench(3755.0), bench(3717.0), bench(3717.0))
        self.assertEqual(decision["decision"], gates.WARNING)
        self.assertTrue(gates.suite_should_continue(decision["decision"]))

    def test_small_regression_passes(self):
        decision = gates.benchmark_gate(bench(3720.0), bench(3717.0), bench(3717.0))
        self.assertEqual(decision["decision"], gates.PASS)

    def test_control_drift_invalidates_rather_than_verdicts(self):
        """v2 measured references once and compared hours later.

        If the host moved between the brackets the treatment number is not
        evidence, so the stage must be invalid rather than a pass or a kill.
        """
        decision = gates.benchmark_gate(bench(3700.0), bench(3700.0), bench(3800.0))
        self.assertEqual(decision["decision"], gates.INVALID)
        self.assertIn("drift", decision["evidence"]["reason"])

    def test_mismatched_batch_shape_invalidates(self):
        """v2 compared a ramped treatment against a flat reference."""
        decision = gates.benchmark_gate(
            bench(3700.0, batches=(131072, 524288)),
            bench(3700.0),
            bench(3700.0),
        )
        self.assertEqual(decision["decision"], gates.INVALID)

    def test_steady_stats_trims_compile_and_tail(self):
        records = [
            {
                "event": "train",
                "step": index + 1,
                "step_time_ms": 150000.0 if index == 0 else 3700.0,
                "effective_batch_tokens": 524288,
                "training_time_seconds": 350.0,
            }
            for index in range(50)
        ]
        stats = gates.steady_step_stats(records)
        self.assertEqual(stats["updates"], 38)
        self.assertAlmostEqual(stats["median_step_time_ms"], 3700.0)


class SpikeReporting(unittest.TestCase):
    """The exp015 regression.

    Real numbers from night-20260802-v2/exp015/health: warmup carried a
    63.94 excursion at update 80 that recovered in one update, while the
    post-warmup phase was calm with a max of 1.92. The v2 gate pooled both
    phases into one median of 0.5044, putting its threshold at 5.04, which
    ordinary warmup behaviour cleared -- and killed the experiment.
    """

    def test_phase_medians_are_not_pooled(self):
        records = train_records(
            [(step, "warmup", 0.875) for step in range(1, 97)]
            + [(step, "steady", 0.4175) for step in range(97, 257)]
        )
        warmup = gates.spike_clusters(records, "warmup")
        steady = gates.spike_clusters(records, "steady")
        self.assertAlmostEqual(warmup["median_grad_norm"], 0.875)
        self.assertAlmostEqual(steady["median_grad_norm"], 0.4175)
        # Pooling would have put the warmup threshold near 5.0 instead of 8.75.
        self.assertGreater(warmup["spike_threshold"], 8.0)

    def test_warmup_excursion_is_reported_and_does_not_gate(self):
        specs = [(step, "warmup", 0.875) for step in range(1, 97)]
        specs[79] = (80, "warmup", 63.94)
        records = train_records(specs)
        warmup = gates.spike_clusters(records, "warmup")
        self.assertEqual(len(warmup["clusters"]), 1)
        self.assertEqual(warmup["clusters"][0]["steps"], [80])
        # Nothing in the v3 suite converts spike reporting into a decision.
        self.assertNotIn("decision", warmup)

    def test_adjacent_excursions_are_one_event(self):
        """A spike and its recovery update are one thing happening.

        v2 counted them as two independent pathologies, which is how two
        adjacent updates tripped a "two or more" kill rule.
        """
        specs = [(step, "warmup", 1.0) for step in range(1, 21)]
        specs[9] = (10, "warmup", 40.0)
        specs[10] = (11, "warmup", 35.0)
        records = train_records(specs)
        clusters = gates.spike_clusters(records, "warmup")["clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["steps"], [10, 11])
        self.assertAlmostEqual(clusters[0]["peak"], 40.0)

    def test_separated_excursions_are_distinct_events(self):
        specs = [(step, "steady", 1.0) for step in range(1, 41)]
        specs[9] = (10, "steady", 40.0)
        specs[29] = (30, "steady", 45.0)
        records = train_records(specs)
        clusters = gates.spike_clusters(records, "steady")["clusters"]
        self.assertEqual(len(clusters), 2)
        self.assertEqual([c["steps"] for c in clusters], [[10], [30]])


class ProxyGate(unittest.TestCase):
    BASE = {
        "status": "complete",
        "tokens_seen": 937426944,
        "target_tokens": 937426944,
        "final_val_loss": 3.57,
        "final_checkpoint": {"sha256": "a" * 64},
    }

    def summary(self, **overrides):
        merged = dict(self.BASE)
        merged.update(overrides)
        return merged

    def test_pass_kill_and_inconclusive(self):
        for loss, expected in (
            (3.5770, gates.PASS),
            (3.5900, gates.KILL),
            (3.5820, gates.INCONCLUSIVE),
        ):
            decision = gates.proxy_gate(
                self.summary(final_val_loss=loss),
                3.577923,
                3.585923,
                "exp012_proxy_seed0",
                3.581923,
            )
            self.assertEqual(decision["decision"], expected, loss)

    def test_missing_final_checkpoint_is_invalid_not_a_result(self):
        """The v2 defect: proxy stages ran with --skip_final_checkpoint.

        exp012-exp014 produced numbers whose weights no longer exist. A loss
        without retained weights is not reproducible, so it is not scored.
        """
        decision = gates.proxy_gate(
            self.summary(final_checkpoint=None), 3.577923, 3.585923, "exp012", 3.581923
        )
        self.assertEqual(decision["decision"], gates.INVALID)
        self.assertIn("final checkpoint", decision["evidence"]["reason"])

    def test_aborted_stage_is_not_scored(self):
        decision = gates.proxy_gate(
            self.summary(status="aborted-envelope"),
            3.577923,
            3.585923,
            "exp012",
            3.581923,
        )
        self.assertEqual(decision["decision"], gates.INVALID)

    def test_short_token_budget_is_not_scored(self):
        decision = gates.proxy_gate(
            self.summary(tokens_seen=800000000),
            3.577923,
            3.585923,
            "exp012",
            3.581923,
        )
        self.assertEqual(decision["decision"], gates.INVALID)


class CheckpointRouting(unittest.TestCase):
    def test_latest_and_milestone_never_resolve_onto_final(self):
        """v2 rotated every save onto one checkpoint.pt."""
        final = HELPERS["checkpoint_destination"]("/ckpt", "final", 1788)
        latest = HELPERS["checkpoint_destination"]("/ckpt", "latest", 1000)
        milestone = HELPERS["checkpoint_destination"]("/ckpt", "milestone", 1024)
        self.assertTrue(final.endswith("final.pt"))
        self.assertNotEqual(latest, final)
        self.assertNotEqual(milestone, final)
        self.assertNotEqual(latest, milestone)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            HELPERS["checkpoint_destination"]("/ckpt", "scratch", 1)

    def test_milestones_are_distinct_per_step(self):
        first = HELPERS["checkpoint_destination"]("/ckpt", "milestone", 256)
        second = HELPERS["checkpoint_destination"]("/ckpt", "milestone", 512)
        self.assertNotEqual(first, second)


class ResumeTruncation(unittest.TestCase):
    def test_records_past_the_cursor_are_dropped(self):
        """A checkpoint can lag the last written lines; resume must be idempotent."""
        records = [{"tokens_seen": t, "step": i} for i, t in enumerate([100, 200, 300, 400])]
        kept = HELPERS["truncate_metrics_records"](records, 200)
        self.assertEqual([r["tokens_seen"] for r in kept], [100, 200])

    def test_no_duplicates_after_replaying_the_same_span(self):
        original = [{"tokens_seen": t} for t in (100, 200, 300)]
        kept = HELPERS["truncate_metrics_records"](original, 200)
        replayed = kept + [{"tokens_seen": 300}, {"tokens_seen": 400}]
        self.assertEqual([r["tokens_seen"] for r in replayed], [100, 200, 300, 400])


class Envelope(unittest.TestCase):
    def test_disabled_when_unset(self):
        self.assertEqual(HELPERS["parse_validation_envelope"](""), [])
        self.assertIsNone(HELPERS["envelope_limit_at"]([], 500))

    def test_interpolates_between_points(self):
        points = [(0, 10.0), (100, 5.0)]
        self.assertAlmostEqual(HELPERS["envelope_limit_at"](points, 50), 7.5)

    def test_clamps_outside_the_range(self):
        points = [(10, 9.0), (100, 5.0)]
        self.assertAlmostEqual(HELPERS["envelope_limit_at"](points, 0), 9.0)
        self.assertAlmostEqual(HELPERS["envelope_limit_at"](points, 1000), 5.0)

    def test_built_envelope_sits_above_the_reference_curve(self):
        curve = [
            {"event": "validation", "tokens_seen": 100, "val_loss": 4.0},
            {"event": "validation", "tokens_seen": 200, "val_loss": 3.6},
        ]
        envelope = gates.envelope_from_curve(curve, margin=0.25)
        self.assertEqual(envelope, [[100, 4.25], [200, 3.85]])

    def test_margin_is_loose_enough_to_let_a_worse_run_finish(self):
        """A merely worse candidate is evidence; an aborted one is not."""
        envelope = gates.envelope_from_curve(
            [{"event": "validation", "tokens_seen": 937426944, "val_loss": 3.581923}],
            margin=0.25,
        )
        limit = HELPERS["envelope_limit_at"](
            [tuple(point) for point in envelope], 937426944
        )
        # exp014, the worst real proxy so far, must still be allowed to run.
        self.assertGreater(limit, 3.624233)


class Phase(unittest.TestCase):
    def test_boundaries(self):
        phase = HELPERS["training_phase"]
        self.assertEqual(phase(0, 100, 1000, 200), "warmup")
        self.assertEqual(phase(99, 100, 1000, 200), "warmup")
        self.assertEqual(phase(100, 100, 1000, 200), "steady")
        self.assertEqual(phase(799, 100, 1000, 200), "steady")
        self.assertEqual(phase(800, 100, 1000, 200), "warmdown")


class StageReuse(unittest.TestCase):
    def test_stale_stage_from_another_manifest_is_not_reused(self):
        """v2 accepted any directory holding five expected filenames.

        That is how a stage measured at control SHA 4d734df was reused under
        a manifest that had already moved to 5d440e4.
        """
        with tempfile.TemporaryDirectory() as directory:
            identity = {"manifest_hash": "aaa", "sha": "111", "stage": "proxy"}
            suite.write_json_atomic(
                os.path.join(directory, "stage.json"), {"identity": identity}
            )
            suite.write_json_atomic(
                os.path.join(directory, "gate.json"), {"decision": "pass"}
            )
            ok, _ = suite.stage_is_reusable(directory, identity)
            self.assertTrue(ok)

            moved = dict(identity, manifest_hash="bbb")
            ok, why = suite.stage_is_reusable(directory, moved)
            self.assertFalse(ok)
            self.assertIn("identity", why)

    def test_incomplete_stage_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            suite.write_json_atomic(
                os.path.join(directory, "stage.json"), {"identity": {}}
            )
            ok, why = suite.stage_is_reusable(directory, {})
            self.assertFalse(ok)
            self.assertIn("not complete", why)


class Exp016Checkpoints(unittest.TestCase):
    DECLARED = {
        "sha256": "b" * 64,
        "bytes": 1482759539,
        "expect_next_step": 1788,
        "expect_tokens_seen": 937426944,
        "expect_run_mode": "proxy",
        "expect_seed": 0,
    }
    GOOD = {
        "sha256": "b" * 64,
        "bytes": 1482759539,
        "next_step": 1788,
        "tokens_seen": 937426944,
        "run_mode": "proxy",
        "seed": 0,
        "state_keys": ["args", "model", "optimizer", "rng_state", "train_loader"],
    }

    def test_canonical_checkpoint_passes(self):
        self.assertEqual(gates.checkpoint_field_failures(self.DECLARED, self.GOOD), [])

    def test_mid_run_checkpoint_is_rejected(self):
        """The exact hole in exp016's own validator.

        A mid-run baseline checkpoint has the right run_mode and plausible
        args, so a step-and-token check is what separates it from the
        canonical final state.
        """
        observed = dict(self.GOOD, next_step=1536, tokens_seen=805306368)
        failures = gates.checkpoint_field_failures(self.DECLARED, observed)
        self.assertTrue(any("next_step" in f for f in failures))
        self.assertTrue(any("tokens_seen" in f for f in failures))

    def test_wrong_content_short_circuits(self):
        observed = dict(self.GOOD, sha256="c" * 64)
        failures = gates.checkpoint_field_failures(self.DECLARED, observed)
        self.assertEqual(len(failures), 1)
        self.assertIn("sha256", failures[0])

    def test_missing_state_is_rejected(self):
        observed = dict(self.GOOD, state_keys=["model"])
        failures = gates.checkpoint_field_failures(self.DECLARED, observed)
        self.assertTrue(any("optimizer" in f for f in failures))
        self.assertTrue(any("rng_state" in f for f in failures))

    def test_missing_file_is_rejected(self):
        failures = gates.checkpoint_field_failures(
            self.DECLARED, {"missing": True, "path": "/nope.pt"}
        )
        self.assertEqual(len(failures), 1)


class AccountingGate(unittest.TestCase):
    def test_parameter_mismatch_is_invalid_not_a_kill(self):
        """A model that is not the preregistered model has no claim to reject."""
        decision = gates.accounting_gate(
            {"parameter_count": 123533568, "tying_preserved": True},
            {"parameter_count": 109376256},
        )
        self.assertEqual(decision["decision"], gates.INVALID)

    def test_broken_tying_fails(self):
        decision = gates.accounting_gate({"tying_preserved": False}, {})
        self.assertEqual(decision["decision"], gates.INVALID)

    def test_failed_equivalence_fails(self):
        decision = gates.accounting_gate(
            {
                "tying_preserved": True,
                "reference_equivalence": {
                    "passed": False,
                    "max_abs_difference": 3.1,
                    "tolerance": 0.02,
                },
            },
            {},
        )
        self.assertEqual(decision["decision"], gates.INVALID)

    def test_matching_report_passes(self):
        decision = gates.accounting_gate(
            {
                "parameter_count": 109376256,
                "mlp_hidden_width": 1536,
                "tying_preserved": True,
                "optimizer_groups_cover_all_parameters": True,
                "reference_equivalence": {"passed": True},
            },
            {"parameter_count": 109376256, "mlp_hidden_width": 1536},
        )
        self.assertEqual(decision["decision"], gates.PASS)


class NoFullRun(unittest.TestCase):
    def test_shipped_manifest_has_no_full_stage(self):
        manifest = suite.load_manifest(os.path.join(HERE, "manifest.json"))
        self.assertTrue(manifest["experiments"])

    def test_full_run_mode_in_arguments_is_rejected(self):
        manifest = {
            "common_args": {"proxy": ["--run_mode", "full"]},
            "experiments": [{"id": "x", "stages": ["proxy"], "proxy": {}}],
        }
        with self.assertRaises(suite.InfrastructureFailure):
            suite.assert_no_full_run(manifest)

    def test_full_run_mode_in_extra_args_is_rejected(self):
        manifest = {
            "common_args": {},
            "experiments": [
                {
                    "id": "x",
                    "stages": ["proxy"],
                    "proxy": {"extra_args": ["--run_mode", "full"]},
                }
            ],
        }
        with self.assertRaises(suite.InfrastructureFailure):
            suite.assert_no_full_run(manifest)

    def test_readonly_full_checkpoint_reference_is_allowed(self):
        """Reading exp000's full-run checkpoint is not launching a full run."""
        manifest = {
            "common_args": {"proxy": ["--run_mode", "proxy"]},
            "canonical_checkpoints": {"x": {"expect_run_mode": "full"}},
            "experiments": [{"id": "x", "stages": ["proxy"], "proxy": {}}],
        }
        suite.assert_no_full_run(manifest)

    def test_launcher_contains_no_full_stage(self):
        launcher = read_text(os.path.join(REPO_ROOT, "run_proxy_suite_v3.sh"))
        self.assertNotIn('"--run_mode", "full"', launcher)
        self.assertIn("--run-paid", launcher)
        self.assertIn("REFUSING TO START", launcher)

    def test_legacy_launcher_hard_fails(self):
        legacy = read_text(os.path.join(REPO_ROOT, "run_suite.sh"))
        self.assertIn("exit 64", legacy)
        self.assertIn("LEGACY", legacy)
        self.assertNotIn("run_one full", legacy)


class ManifestArithmetic(unittest.TestCase):
    """Derive the preregistered parameter counts instead of trusting them.

    The accounting gate compares a real model against these numbers, so a
    typo here would either wave through a wrong model or block a correct
    one. Caught exactly that: exp020 was written as 123,533,568.
    """

    VOCAB, EMBD, LAYERS = 50257, 768, 12

    def expected_total(self, ratio):
        attention = self.EMBD * 3 * self.EMBD + self.EMBD * self.EMBD
        mlp = 2 * self.EMBD * int(ratio * self.EMBD)
        return self.VOCAB * self.EMBD + self.LAYERS * (attention + mlp)

    def expectations(self, experiment_id):
        manifest = suite.load_manifest(os.path.join(HERE, "manifest.json"))
        experiment = next(
            e for e in manifest["experiments"] if e["id"] == experiment_id
        )
        return experiment["accounting"]["expectations"]

    def test_exp019_matches_uniform_3d(self):
        expectations = self.expectations("exp019")
        self.assertEqual(expectations["parameter_count"], self.expected_total(3.0))
        # Fused SwiGLU at hidden 1536: c_fc 768x3072 plus c_proj 1536x768.
        self.assertEqual(
            expectations["mlp_parameters_per_layer"],
            self.EMBD * 2 * 1536 + 1536 * self.EMBD,
        )
        self.assertEqual(
            expectations["mlp_parameters_per_layer"], 2 * self.EMBD * 2304
        )

    def test_exp017_matches_uniform_3d(self):
        expectations = self.expectations("exp017")
        self.assertEqual(expectations["parameter_count"], self.expected_total(3.0))
        self.assertEqual(
            expectations["no_decay_parameter_count"], self.VOCAB * self.EMBD
        )

    def test_exp020_matches_4d_baseline(self):
        expectations = self.expectations("exp020")
        self.assertEqual(expectations["parameter_count"], self.expected_total(4.0))

    def test_envelopes_end_at_the_reference_final_loss_plus_margin(self):
        manifest = suite.load_manifest(os.path.join(HERE, "manifest.json"))
        for name, reference_key in (
            ("exp012", "exp012_proxy_seed0"),
            ("baseline", "baseline_proxy_seed0"),
        ):
            final_tokens, final_limit = manifest["envelopes"][name][-1]
            self.assertEqual(final_tokens, manifest["token_budget"])
            reference = manifest["references"][reference_key]["final_val_loss"]
            self.assertAlmostEqual(final_limit - reference, 0.25, places=5)


class UploadRetry(unittest.TestCase):
    def test_failed_upload_reports_without_touching_training(self):
        """An upload failure must never mean recomputing a proxy."""
        from suite_v3 import upload

        with tempfile.TemporaryDirectory() as stage_dir:
            with open(os.path.join(stage_dir, "summary.json"), "w") as handle:
                json.dump({"status": "complete"}, handle)
            upload.RETRY_SECONDS = 0
            receipt = upload.upload_stage(
                project="p", group="g", run_name="r", stage_dir=stage_dir
            )
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(len(receipt["attempts"]), upload.MAX_ATTEMPTS)
            self.assertIn("no training needs to be repeated", receipt["note"])
            self.assertTrue(
                os.path.exists(os.path.join(stage_dir, "wandb_upload.json"))
            )
            # Training outputs are untouched.
            self.assertTrue(os.path.exists(os.path.join(stage_dir, "summary.json")))

    def test_live_log_and_checkpoints_are_not_artifact_members(self):
        """v2 shipped the uploader's own live log inside the artifact."""
        with tempfile.TemporaryDirectory() as stage_dir:
            os.makedirs(os.path.join(stage_dir, "checkpoints"))
            for name in ("summary.json", "upload.log", "metrics.jsonl"):
                open(os.path.join(stage_dir, name), "w").close()
            open(os.path.join(stage_dir, "checkpoints", "final.pt"), "w").close()
            names = {relative for _, relative in upload.collect_files(stage_dir)}
            self.assertIn("summary.json", names)
            self.assertIn("metrics.jsonl", names)
            self.assertNotIn("upload.log", names)
            self.assertNotIn(os.path.join("checkpoints", "final.pt"), names)


from suite_v3 import upload  # noqa: E402  (imported late for the test above)


if __name__ == "__main__":
    unittest.main(verbosity=2)
