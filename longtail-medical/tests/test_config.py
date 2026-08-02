import tempfile
import unittest
from pathlib import Path

import yaml

from longtail_medical.config import load_config


class ConfigTest(unittest.TestCase):
    def test_stage1_test_lock(self):
        payload = {
            "experiment": {}, "data": {}, "model": {}, "training": {},
            "checkpoint": {
                "primary_policy": "last",
                "secondary_policy": "best_validation",
                "monitor": None,
            },
            "evaluation": {"run_test": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test evaluation is locked"):
                load_config(path)

    def test_validation_selected_cannot_be_primary(self):
        payload = {
            "experiment": {}, "data": {}, "model": {},
            "training": {"monitor": "balanced_accuracy"},
            "checkpoint": {
                "primary_policy": "best_validation",
                "secondary_policy": "best_validation",
                "monitor": "balanced_accuracy",
            },
            "evaluation": {"run_test": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "last checkpoint"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
