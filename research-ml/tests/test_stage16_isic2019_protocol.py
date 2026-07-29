from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.check_stage16a_gate import validate
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


if __name__ == "__main__":
    unittest.main()
