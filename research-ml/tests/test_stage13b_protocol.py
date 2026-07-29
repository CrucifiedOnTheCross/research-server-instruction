from __future__ import annotations

import unittest

from src.config import load_config
from tools.check_stage11b_gate import compare_recipe
from tools.check_stage13b_gate import validate_manifest


class Stage13BProtocolTests(unittest.TestCase):
    def test_recipes_match_qualified_convnext_small(self) -> None:
        baseline = load_config("configs/ham10000_stage11_convnext_small_regularized_384.yaml")
        for path in (
            "configs/ham10000_stage13b_replay_coverage_convnext_small_384.yaml",
            "configs/ham10000_stage13b_synthetic_coverage_convnext_small_384.yaml",
        ):
            config = load_config(path)
            self.assertEqual(compare_recipe(baseline, config, path), [])
            self.assertFalse(config["evaluation"]["run_test"])

    def test_manifest_must_have_open_gate_without_test(self) -> None:
        valid = {
            "protocol": "stage13_multiencoder_coverage_targeted",
            "locked_test_used": False,
            "gate": {
                "gate_open": True,
                "checks": {
                    "balanced_dose": True,
                    "locked_test_used": False,
                },
            },
        }
        self.assertEqual(validate_manifest(valid), [])
        invalid = {**valid, "locked_test_used": True}
        self.assertTrue(validate_manifest(invalid))


if __name__ == "__main__":
    unittest.main()
