from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from src.datasets import ClassBalancedUndersampler, CsvImageDataset
from src.metrics import compute_metrics
from src.models import configure_classifier_only
from tools.calibrate_stage9_predictions import fit_temperature, group_calibration_split
from tools.make_source_matched_replay import build_source_replay


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.head = nn.Linear(3, num_classes)


class Stage9ControlTests(unittest.TestCase):
    def test_undersampler_is_balanced_unique_and_epoch_deterministic(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 2, 2, 2])
        sampler = ClassBalancedUndersampler(labels, seed=17)
        sampler.set_epoch(3)
        first = list(sampler)
        sampler.set_epoch(3)
        second = list(sampler)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)
        selected_labels = labels[first]
        self.assertEqual(
            {label: int((selected_labels == label).sum()) for label in (0, 1, 2)},
            {0: 2, 1: 2, 2: 2},
        )

    def test_source_replay_matches_selected_sources(self) -> None:
        train = pd.DataFrame(
            [
                {
                    "image_id": "real_a",
                    "image_path": "raw/real_a.jpg",
                    "label": "mel",
                    "group_id": "lesion_a",
                    "is_synthetic": 0,
                },
                {
                    "image_id": "real_b",
                    "image_path": "raw/real_b.jpg",
                    "label": "nv",
                    "group_id": "lesion_b",
                    "is_synthetic": 0,
                },
            ]
        )
        scores = pd.DataFrame(
            [
                {
                    "image_id": "synth_a",
                    "label": "mel",
                    "source_image_id": "real_a",
                    "source_image_path": "raw/real_a.jpg",
                    "source_group_id": "lesion_a",
                    "selected_by_stage6": 1,
                },
                {
                    "image_id": "synth_unused",
                    "label": "nv",
                    "source_image_id": "real_b",
                    "source_image_path": "raw/real_b.jpg",
                    "source_group_id": "lesion_b",
                    "selected_by_stage6": 0,
                },
            ]
        )
        augmented, replay, metadata = build_source_replay(
            train,
            scores,
            sample_weight=0.5,
            expected_selected=1,
        )
        self.assertEqual(len(augmented), 3)
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay.iloc[0]["image_path"], "raw/real_a.jpg")
        self.assertEqual(replay.iloc[0]["group_id"], "lesion_a")
        self.assertEqual(float(replay.iloc[0]["sample_weight"]), 0.5)
        self.assertEqual(int(replay.iloc[0]["is_synthetic"]), 0)
        self.assertEqual(metadata["unique_replay_sources"], 1)

    def test_dataset_exposes_row_sample_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), color=(120, 80, 60)).save(root / "image.jpg")
            pd.DataFrame(
                [
                    {
                        "image_path": "image.jpg",
                        "label": "mel",
                        "is_synthetic": 0,
                        "sample_weight": 0.25,
                    }
                ]
            ).to_csv(root / "data.csv", index=False)
            dataset = CsvImageDataset(
                root / "data.csv",
                root,
                "image_path",
                "label",
                "is_synthetic",
                "sample_weight",
                {"mel": 0},
                transform=None,
            )
            self.assertEqual(dataset[0]["sample_weight"], 0.25)

    def test_classifier_only_freezes_encoder_and_reinitializes_head(self) -> None:
        model = TinyClassifier()
        old_head_weight = model.head.weight.detach().clone()
        trainable = configure_classifier_only(
            model,
            num_classes=2,
            reinitialize_classifier=True,
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))
        self.assertTrue(all(name.startswith("head.") for name in trainable))
        self.assertFalse(torch.equal(old_head_weight, model.head.weight.detach()))

    def test_metrics_include_per_class_ranking(self) -> None:
        logits = np.asarray([[3.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 3.0]])
        targets = np.asarray([0, 1, 0, 1])
        metrics = compute_metrics(logits, targets, {0: "mel", 1: "nv"}, ece_bins=5)
        self.assertEqual(metrics["per_class"]["mel"]["auroc"], 1.0)
        self.assertEqual(metrics["per_class"]["mel"]["auprc"], 1.0)

    def test_calibration_split_keeps_groups_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": [f"group_{index // 2}" for index in range(24)],
            }
        )
        targets = np.asarray([(index // 2) % 2 for index in range(24)])
        calibration, evaluation = group_calibration_split(frame, targets, seed=11)
        calibration_groups = set(frame.iloc[calibration]["group_id"])
        evaluation_groups = set(frame.iloc[evaluation]["group_id"])
        self.assertFalse(calibration_groups & evaluation_groups)

    def test_temperature_fit_returns_positive_value(self) -> None:
        logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
        targets = np.asarray([0, 1, 0, 1])
        self.assertGreater(fit_temperature(logits, targets), 0.0)


if __name__ == "__main__":
    unittest.main()
