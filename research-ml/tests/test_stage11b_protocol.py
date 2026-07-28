from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import load_config
from tools.analyze_stage11b_results import decision_summary
from tools.check_stage11b_gate import compare_recipe, validate_pair


class Stage11BProtocolTests(unittest.TestCase):
    def test_arm_recipes_match_qualified_convnext_small(self) -> None:
        baseline = load_config("configs/ham10000_stage11_convnext_small_regularized_384.yaml")
        synthetic = load_config(
            "configs/ham10000_stage11b_synthetic_strict_id_convnext_small_384.yaml"
        )
        replay = load_config(
            "configs/ham10000_stage11b_replay_strict_id_convnext_small_384.yaml"
        )
        self.assertEqual(compare_recipe(baseline, synthetic, "synthetic"), [])
        self.assertEqual(compare_recipe(baseline, replay, "replay"), [])
        self.assertFalse(synthetic["evaluation"]["run_test"])
        self.assertFalse(replay["evaluation"]["run_test"])
        self.assertEqual(synthetic["training"]["synthetic_weight"], 0.5)
        self.assertTrue(replay["training"]["use_sample_weights"])

    def make_pair(self, root: Path) -> tuple[dict, dict]:
        split = root / "splits/stage10"
        split.mkdir(parents=True)
        eval_dir = root / "splits/stage8"
        eval_dir.mkdir(parents=True)
        base = pd.DataFrame(
            [
                {
                    "image_path": "base-a.jpg",
                    "label": "nv",
                    "is_synthetic": 0,
                    "image_id": "base-a",
                    "group_id": "train-a",
                    "source": "real",
                },
                {
                    "image_path": "base-b.jpg",
                    "label": "bcc",
                    "is_synthetic": 0,
                    "image_id": "base-b",
                    "group_id": "train-b",
                    "source": "real",
                },
            ]
        )
        selected_rows = []
        replay_rows = []
        for label in ("mel", "akiec", "bkl"):
            for index in range(30):
                synthetic_id = f"syn-{label}-{index}"
                source_id = f"real-{label}-{index}"
                selected_rows.append(
                    {
                        "image_path": f"{synthetic_id}.jpg",
                        "label": label,
                        "is_synthetic": 1,
                        "image_id": synthetic_id,
                        "group_id": f"syn-group-{label}-{index}",
                        "source": "synthetic",
                        "source_image_id": source_id,
                        "source_group_id": f"train-source-{label}-{index}",
                        "passes_geometry_filter": 1,
                        "stage10_stratum": "strict_id",
                    }
                )
                replay_rows.append(
                    {
                        "image_path": f"{source_id}.jpg",
                        "label": label,
                        "is_synthetic": 0,
                        "image_id": source_id,
                        "group_id": f"train-source-{label}-{index}",
                        "source": "real",
                        "is_replay": 1,
                        "sample_weight": 0.5,
                        "replay_source_image_id": source_id,
                        "replay_for_synthetic_image_id": synthetic_id,
                    }
                )
        selected = pd.DataFrame(selected_rows)
        replay = pd.DataFrame(replay_rows)
        synthetic_train = pd.concat([base, selected], ignore_index=True, sort=False)
        replay_base = base.assign(is_replay=0, sample_weight=1.0)
        replay_train = pd.concat([replay_base, replay], ignore_index=True, sort=False)
        selected.to_csv(split / "selected_synthetic_strict_id.csv", index=False)
        replay.to_csv(split / "source_replay_rows_strict_id.csv", index=False)
        synthetic_train.to_csv(split / "train_synthetic_strict_id.csv", index=False)
        replay_train.to_csv(split / "train_source_replay_strict_id.csv", index=False)
        (split / "stage10_strata_manifest.json").write_text(
            json.dumps(
                {
                    "protocol": "stage10_equal_dose_geometry_strata",
                    "rows_per_stratum": 90,
                    "sample_weight": 0.5,
                }
            ),
            encoding="utf-8",
        )
        empty_eval = pd.DataFrame(columns=["image_id", "group_id"])
        empty_eval.to_csv(eval_dir / "val_real.csv", index=False)
        empty_eval.to_csv(eval_dir / "locked_test_real.csv", index=False)
        synthetic_config = {
            "data": {
                "train_csv": "splits/stage10/train_synthetic_strict_id.csv",
                "val_csv": "splits/stage8/val_real.csv",
                "test_csv": "splits/stage8/locked_test_real.csv",
            },
            "training": {
                "synthetic_weight": 0.5,
                "synthetic_weight_per_class": {"mel": 0.5, "akiec": 0.5, "bkl": 0.5},
            },
        }
        replay_config = {
            "data": {
                "train_csv": "splits/stage10/train_source_replay_strict_id.csv",
                "val_csv": "splits/stage8/val_real.csv",
                "test_csv": "splits/stage8/locked_test_real.csv",
            },
            "training": {"use_sample_weights": True},
        }
        return synthetic_config, replay_config

    def test_pair_gate_accepts_exact_source_matched_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic, replay = self.make_pair(root)
            report, errors = validate_pair(root, synthetic, replay)
            self.assertEqual(errors, [])
            self.assertEqual(report["row_counts"]["synthetic_added"], 90)
            self.assertEqual(report["row_counts"]["replay_added"], 90)

    def test_pair_gate_rejects_broken_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic, replay = self.make_pair(root)
            path = root / "splits/stage10/source_replay_rows_strict_id.csv"
            rows = pd.read_csv(path)
            rows.loc[0, "replay_source_image_id"] = "wrong-source"
            rows.to_csv(path, index=False)
            _, errors = validate_pair(root, synthetic, replay)
            self.assertTrue(any("source mapping" in error for error in errors))

    def test_decision_requires_melanoma_guardrail(self) -> None:
        paired = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "mean_difference_synthetic_minus_replay": value,
                    "synthetic_wins": wins,
                }
                for metric, value, wins in (
                    ("macro_f1", 0.02, 3),
                    ("mcc", 0.03, 3),
                    ("balanced_accuracy", 0.01, 3),
                    ("mel_recall", -0.08, 0),
                    ("mel_auprc", 0.01, 3),
                )
            ]
        )
        bootstrap = pd.DataFrame(
            [
                {"metric": "macro_f1", "ci95_low": 0.001},
                {"metric": "mcc", "ci95_low": 0.002},
            ]
        )
        decision = decision_summary(paired, bootstrap)
        self.assertEqual(decision["classification"], "mixed_positive")
        self.assertFalse(decision["melanoma_guardrail_passed"])


if __name__ == "__main__":
    unittest.main()
