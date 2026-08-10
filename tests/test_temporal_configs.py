"""Configuration-level checks for the TemporalDiffusionPolicy BC baselines."""
import json
from pathlib import Path
import unittest

from robomimic.config import config_factory


CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "robomimic"
    / "exps"
    / "temporaldp"
    / "temporal"
)

VARIANTS = {
    "bc": ("SpatialSoftmax", "CropRandomizer"),
    "bc_rnn": ("SpatialSoftmax", "CropRandomizer"),
    "bc_mod": ("SpatialMeanPool", "CropRandomizer"),
    "bc_rnn_mod": ("SpatialMeanPool", "CropRandomizer"),
    "bc_mod_nocrop": ("SpatialMeanPool", None),
    "bc_rnn_mod_nocrop": ("SpatialMeanPool", None),
}


class TestTemporalConfigs(unittest.TestCase):
    def test_configs_load_with_expected_observations(self):
        config_paths = sorted(CONFIG_ROOT.glob("*/*.json"))
        self.assertEqual(len(config_paths), 12)
        for path in config_paths:
            with path.open() as handle:
                external = json.load(handle)
            config = config_factory(external["algo_name"])
            with config.values_unlocked():
                config.update(external)
            self.assertEqual(config.all_obs_keys, ["agent_pose", "image"], path)
            self.assertTrue(config.experiment.rollout.enabled, path)
            self.assertTrue(config.experiment.save.on_best_rollout_success_rate, path)
            self.assertEqual(config.train.data, f"datasets/temporal/{path.parent.name}_image.hdf5", path)
            expected_pool, expected_randomizer = VARIANTS[path.stem]
            self.assertEqual(config.observation.encoder.rgb.core_kwargs.pool_class, expected_pool, path)
            self.assertEqual(config.observation.encoder.rgb.obs_randomizer_class, expected_randomizer, path)


if __name__ == "__main__":
    unittest.main()
