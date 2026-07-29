from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.check_stage16a_gate import validate
from src.config import load_config
from src.image_transforms import CropDarkFieldOfView
from tools.prepare_isic2019 import (
    CLASS_NAMES,
    assign_connected_groups,
    audit_image,
    label_from_ground_truth,
    split_groups,
    validate_disjoint,
)


def row(image_id: str, label: str, lesion_id: str = "", digest: str = "") -> dict:
    return {
        "image_id": image_id,
        "label": label,
        "lesion_id": lesion_id,
        "sha256": digest or f"sha_{image_id}",
        "visual_hash": f"visual_{image_id}",
    }


class Stage16ISIC2019Tests(unittest.TestCase):
    def test_scc_is_not_mapped_to_ak(self) -> None:
        ground_truth = {"image": "ISIC_x", **{name.upper(): "0" for name in CLASS_NAMES}}
        ground_truth["SCC"] = "1"
        self.assertEqual(label_from_ground_truth(ground_truth), "scc")

    def test_ground_truth_rejects_multiple_labels(self) -> None:
        ground_truth = {"image": "ISIC_x", **{name.upper(): "0" for name in CLASS_NAMES}}
        ground_truth["MEL"] = "1"
        ground_truth["NV"] = "1"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            label_from_ground_truth(ground_truth)

    def test_connected_groups_join_lesion_and_hash_duplicates(self) -> None:
        rows = [
            row("a", "mel", lesion_id="lesion_1"),
            row("b", "mel", lesion_id="lesion_1"),
            row("c", "mel", digest="sha_b"),
        ]
        rows[1]["sha256"] = "sha_b"
        kept, quarantine = assign_connected_groups(rows)
        self.assertFalse(quarantine)
        self.assertEqual(len({item["group_id"] for item in kept}), 1)

    def test_visual_hash_collision_does_not_connect_distinct_images(self) -> None:
        rows = [row("a", "mel"), row("b", "nv")]
        rows[1]["visual_hash"] = rows[0]["visual_hash"]
        kept, quarantine = assign_connected_groups(rows)
        self.assertFalse(quarantine)
        self.assertEqual(len({item["group_id"] for item in kept}), 2)

    def test_conflicting_duplicate_labels_are_quarantined(self) -> None:
        rows = [row("a", "mel", digest="same"), row("b", "nv", digest="same")]
        kept, quarantine = assign_connected_groups(rows)
        self.assertFalse(kept)
        self.assertEqual(len(quarantine), 2)

    def test_splits_are_group_disjoint_and_cover_classes(self) -> None:
        rows = []
        for label in CLASS_NAMES:
            for index in range(20):
                rows.append(row(f"{label}_{index}", label))
        kept, quarantine = assign_connected_groups(rows)
        self.assertFalse(quarantine)
        splits = split_groups(kept, val_size=0.2, test_size=0.2, seed=42)
        validate_disjoint(splits)
        for split_rows in splits.values():
            self.assertEqual({item["label"] for item in split_rows}, set(CLASS_NAMES))

    def test_image_audit_records_geometry_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(path)
            audit = audit_image(path, skip=False)
        self.assertEqual(audit["decode_ok"], 1)
        self.assertEqual((audit["width"], audit["height"]), (12, 8))
        self.assertEqual(len(audit["sha256"]), 64)
        self.assertEqual(len(audit["visual_hash"]), 16)

    def test_gate_reports_missing_artifacts_without_import_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "tools.check_stage16a_gate.EXPECTED_IMAGES", 1
            ):
                details, errors = validate(Path(directory), check_files=False)
        self.assertFalse(details)
        self.assertTrue(errors)

    def test_stage16p_is_a_single_factor_geometry_ablation(self) -> None:
        baseline = load_config(
            "configs/isic2019_stage16_real_ce_natural_384.yaml"
        )
        aspect_pad = load_config(
            "configs/isic2019_stage16p_real_aspect_pad_natural_384.yaml"
        )
        dark_fov = load_config(
            "configs/isic2019_stage16p_real_dark_fov_pad_natural_384.yaml"
        )
        for section in ("data", "model", "training", "imbalance", "evaluation"):
            self.assertEqual(baseline[section], aspect_pad[section])
            self.assertEqual(baseline[section], dark_fov[section])
        self.assertFalse(
            baseline["augmentation"]["train"].get(
                "aspect_preserving_pad", False
            )
        )
        self.assertTrue(
            aspect_pad["augmentation"]["train"]["aspect_preserving_pad"]
        )
        self.assertFalse(
            aspect_pad["augmentation"]["train"]["random_resized_crop"]
        )
        self.assertTrue(
            aspect_pad["augmentation"]["eval"]["aspect_preserving_pad"]
        )
        self.assertFalse(aspect_pad["augmentation"]["eval"]["center_crop"])
        self.assertFalse(aspect_pad["evaluation"]["run_test"])
        self.assertTrue(dark_fov["augmentation"]["train"]["crop_dark_field"])
        self.assertTrue(dark_fov["augmentation"]["eval"]["crop_dark_field"])
        self.assertEqual(
            dark_fov["augmentation"]["train"]["dark_field_threshold"], 8
        )

    def test_dark_field_crop_removes_only_external_black_frame(self) -> None:
        image = Image.new("RGB", (100, 80), (0, 0, 0))
        image.paste((80, 120, 160), (10, 15, 90, 65))
        cropped = CropDarkFieldOfView(
            threshold=8, margin_fraction=0.0
        )(image)
        self.assertEqual(cropped.size, (80, 50))
        self.assertEqual(cropped.getpixel((0, 0)), (80, 120, 160))

    def test_dark_field_crop_rejects_implausibly_small_foreground(self) -> None:
        image = Image.new("RGB", (100, 80), (0, 0, 0))
        image.paste((255, 255, 255), (45, 35, 55, 45))
        cropped = CropDarkFieldOfView(
            threshold=8, margin_fraction=0.0
        )(image)
        self.assertEqual(cropped.size, image.size)


if __name__ == "__main__":
    unittest.main()
