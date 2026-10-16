"""Tests for experiment-CLI discovery labels."""
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "robomimic" / "scripts" / "experiment_cli.py"
SPEC = importlib.util.spec_from_file_location("robomimic_experiment_cli", CLI_PATH)
experiment_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment_cli
SPEC.loader.exec_module(experiment_cli)


class TestExperimentCli(unittest.TestCase):
    def test_discovery_uses_explicit_lte_config_files(self):
        experiments = experiment_cli.discover_experiments()
        lte_experiments = [
            experiment for experiment in experiments if experiment.algorithm == "lte_diffusion_policy_mod"
        ]
        self.assertEqual({experiment.environment for experiment in lte_experiments}, {
            "can", "lift", "square", "transport", "waitatgoal", "liftqa",
        })
        self.assertTrue(all(experiment.config_path.name == "lte_diffusion_policy_mod.json" for experiment in lte_experiments))
        self.assertFalse(any("_generated" in experiment.config_path.parts for experiment in experiments))

    def test_screen_session_name_is_short_and_unique(self):
        session = experiment_cli.make_screen_session(
            "train-lift-learned_temporal_diffusion_policy-lte_idle_direct_time_direct_idle-seed1000"
        )
        next_session = experiment_cli.make_screen_session(
            "train-lift-learned_temporal_diffusion_policy-lte_idle_direct_time_direct_idle-seed1001"
        )
        self.assertLessEqual(len(session), 64)
        self.assertTrue(session.startswith("robomimic-train-lift-"))
        self.assertNotEqual(session, next_session)

    def test_multi_seed_screen_batch_runs_direct_commands_in_order(self):
        experiment = experiment_cli.Experiment(
            "liftqa",
            "tc_diffusion_policy_mod_nocrop",
            (
                Path(__file__).resolve().parents[1]
                / "robomimic/exps/temporaldp/temporal/liftqa/tc_diffusion_policy_mod_nocrop.json"
            ),
        )
        with (
            mock.patch.object(experiment_cli, "select_experiment", return_value=experiment),
            mock.patch.object(experiment_cli, "prompt_gpu", return_value="1"),
            mock.patch.object(experiment_cli, "prompt_seeds", return_value=["1", "2", "3"]),
            mock.patch.object(experiment_cli, "prompt_execution_mode", return_value=experiment_cli.ExecutionMode(True)),
            mock.patch.object(experiment_cli, "prompt_yes_no", return_value=True),
            mock.patch.object(experiment_cli, "make_screen_session", return_value="batch"),
            mock.patch.object(experiment_cli, "start_screen_session", return_value=0) as start,
        ):
            experiment_cli.run_training([experiment], dry_run=False)

        command = start.call_args.args[1]
        self.assertNotIn("run-training-plan", command)
        self.assertLess(command.index("--seed 1"), command.index("--seed 2"))
        self.assertLess(command.index("--seed 2"), command.index("--seed 3"))
        self.assertIn("vla_alert", command)
        self.assertIn('exit "$status"', command)

    def test_learned_temporal_variant_is_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "observation": {"temporal_encodings": {
                    "learned_encoder": {"enabled": True, "input_signal": "velocity"},
                    "use_direct_time": True,
                    "use_direct_idleness": True,
                    "use_idleness_encoding": True,
                }},
            }), encoding="utf-8")
            experiment = experiment_cli.Experiment("lift", "tc_diffusion_policy", config_path, "ph")
            self.assertIn("LTE-velocity", experiment.label)
            self.assertIn("direct-time", experiment.label)
            self.assertIn("direct-idle", experiment.label)

    def test_explicit_lte_config_is_image_without_time(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "robomimic"
            / "exps"
            / "temporaldp"
            / "lift"
            / "ph"
            / "image"
            / "lte_diffusion_policy_mod.json"
        )
        experiment = experiment_cli.Experiment(
            "lift",
            "lte_diffusion_policy_mod",
            config_path,
            "ph",
        )
        learned = experiment_cli.load_config(experiment)["observation"]["temporal_encodings"]["learned_encoder"]
        self.assertTrue(learned["enabled"])
        self.assertEqual(learned["input_signal"], "image_embedding")
        self.assertTrue(learned["use_image_embedding"])
        self.assertFalse(learned["use_timestep_signal"])
        self.assertEqual(learned["num_images"], 1)
        self.assertFalse(learned["multi_image_fusion"])
        self.assertTrue(learned["cache_image_embeddings"])


if __name__ == "__main__":
    unittest.main()
