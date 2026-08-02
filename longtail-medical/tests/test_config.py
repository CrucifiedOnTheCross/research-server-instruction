import tempfile
import unittest
from pathlib import Path

import yaml

from longtail_medical.config import load_config


class ConfigTest(unittest.TestCase):
    def test_stage1_test_lock(self):
        payload = {
            "experiment": {}, "data": {}, "model": {}, "training": {},
            "evaluation": {"run_test": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test evaluation is locked"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
