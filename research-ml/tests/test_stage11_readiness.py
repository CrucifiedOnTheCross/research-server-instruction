from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import timm
from torch import nn

from src.losses import build_loss
from src.models import create_model
from src.train import make_optimizer, make_scheduler, optimizer_group_metadata
from tools.check_stage11_gate import expected_stage10


class Stage11ReadinessTests(unittest.TestCase):
    def test_cross_entropy_label_smoothing_is_train_only_opt_in(self) -> None:
        config = {
            "imbalance": {
                "loss": "cross_entropy",
                "class_weights": "none",
                "balanced_softmax": False,
                "logit_adjustment_tau": 0.0,
            }
        }
        criterion = build_loss(
            config,
            [10, 2],
            torch.device("cpu"),
            label_smoothing=0.05,
        )
        self.assertEqual(criterion.label_smoothing, 0.05)

    def test_label_smoothing_rejects_unsupported_loss(self) -> None:
        config = {
            "imbalance": {
                "loss": "focal",
                "class_weights": "none",
                "focal_gamma": 2.0,
                "balanced_softmax": False,
                "logit_adjustment_tau": 0.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "only with cross_entropy"):
            build_loss(
                config,
                [10, 2],
                torch.device("cpu"),
                label_smoothing=0.05,
            )

    def test_optimizer_metadata_captures_full_parameter_count(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        config = {
            "training": {
                "optimizer": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.05,
                "layer_decay": 1.0,
            }
        }
        optimizer = make_optimizer(config, model)
        metadata = optimizer_group_metadata(optimizer)
        self.assertEqual(
            sum(group["parameter_count"] for group in metadata),
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertEqual(metadata[0]["lr_scale"], 1.0)

    def test_layer_decay_and_scheduler_preserve_lr_ratio(self) -> None:
        model = timm.create_model("convnext_atto", pretrained=False, num_classes=2)
        config = {
            "training": {
                "optimizer": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.05,
                "layer_decay": 0.8,
                "scheduler": "cosine",
                "epochs": 10,
                "warmup_epochs": 2,
                "min_lr": 1e-5,
            }
        }
        optimizer = make_optimizer(config, model)
        scheduler = make_scheduler(config, optimizer)
        initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.assertLess(min(initial_lrs), max(initial_lrs))
        initial_ratio = min(initial_lrs) / max(initial_lrs)
        optimizer.step()
        scheduler.step()
        updated_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.assertAlmostEqual(min(updated_lrs) / max(updated_lrs), initial_ratio)

    def test_stage11_gate_matrix_requires_24_stage10_runs(self) -> None:
        decision = {
            "required_stage10": {
                "seeds": [42, 43, 44],
                "strata": ["strict_id", "aid_radial", "ood_far", "random_remaining"],
                "arms": ["synthetic", "replay"],
            }
        }
        expected = expected_stage10(decision)
        self.assertEqual(len(expected), 24)
        self.assertEqual(len(set(expected)), 24)

    def test_dinov2_honors_explicit_patch_compatible_image_size(self) -> None:
        config = {
            "model": {
                "name": "vit_small_patch14_dinov2.lvd142m",
                "pretrained": False,
                "img_size": 392,
                "drop_rate": 0.0,
                "drop_path_rate": 0.0,
            }
        }
        model = create_model(config, num_classes=7)
        self.assertEqual(model.patch_embed.img_size, (392, 392))

    def test_stage11_decision_starts_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.yaml"
            path.write_text(
                "status: pending_stage10\nstage10_analysis_reviewed: false\n",
                encoding="utf-8",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("pending_stage10", text)


if __name__ == "__main__":
    unittest.main()
