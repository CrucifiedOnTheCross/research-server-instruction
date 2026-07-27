from __future__ import annotations

import unittest

from src.tracking import flatten_mapping, flatten_numeric


class TrackingTests(unittest.TestCase):
    def test_flatten_mapping_preserves_reproducible_config_values(self) -> None:
        flattened = flatten_mapping(
            {
                "training": {"lr": 1e-4, "seeds": [42, 43, 44]},
                "model": {"checkpoint": None},
            }
        )
        self.assertEqual(flattened["training.lr"], 1e-4)
        self.assertEqual(flattened["training.seeds"], "[42, 43, 44]")
        self.assertEqual(flattened["model.checkpoint"], "null")

    def test_flatten_numeric_excludes_booleans_and_arrays(self) -> None:
        flattened = flatten_numeric(
            {
                "macro_f1": 0.8,
                "test_evaluated": False,
                "per_class": {"mel": {"recall": 0.7}},
                "confusion_matrix": [[1, 2], [3, 4]],
            }
        )
        self.assertEqual(
            flattened,
            {"macro_f1": 0.8, "per_class.mel.recall": 0.7},
        )


if __name__ == "__main__":
    unittest.main()
