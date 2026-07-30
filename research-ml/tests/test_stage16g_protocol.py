from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from tools.check_stage16g_readiness import validate
from tools.prepare_stage16g_masks import (
    build_train_mask_manifest,
    choose_mask_member,
    extract_selected_masks,
    image_id_from_mask_name,
)


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "label", "group_id"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


class Stage16GProtocolTests(unittest.TestCase):
    def test_mask_name_parser_is_case_insensitive(self) -> None:
        self.assertEqual(
            image_id_from_mask_name("nested/isic_0012345_segmentation.PNG"),
            "ISIC_0012345",
        )

    def test_ambiguous_noncanonical_masks_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            choose_mask_member(
                "ISIC_0000001",
                ["ISIC_0000001_a.png", "ISIC_0000001_b.png"],
            )

    def test_archive_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "masks.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../ISIC_0000001_segmentation.png", b"mask")
            with self.assertRaisesRegex(RuntimeError, "Unsafe ZIP"):
                extract_selected_masks(archive, Path(directory) / "output")

    def test_manifest_uses_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "train": [
                    {"image_id": "ISIC_0000001", "label": "mel", "group_id": "g1"}
                ],
                "val": [
                    {"image_id": "ISIC_0000002", "label": "mel", "group_id": "g2"}
                ],
                "locked_test": [
                    {"image_id": "ISIC_0000003", "label": "mel", "group_id": "g3"}
                ],
            }
            for name, values in rows.items():
                write_split(root / "splits" / f"{name}.csv", values)
            masks = {}
            for image_id in ("ISIC_0000001", "ISIC_0000002", "ISIC_0000003"):
                path = root / f"{image_id}_segmentation.png"
                path.write_bytes(image_id.encode())
                masks[image_id] = path
            manifest, inventory = build_train_mask_manifest(
                root, masks, "a" * 64, "https://example.invalid/masks.zip"
            )
            self.assertEqual([row["image_id"] for row in manifest], ["ISIC_0000001"])
            self.assertEqual(inventory["mask_ids_in_validation_or_test_not_used"], 2)
            self.assertFalse(inventory["locked_test_evaluated"])

    def test_readiness_fails_when_a_target_class_has_too_few_lesions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_rows = [
                {"image_id": "ISIC_0000001", "label": "mel", "group_id": "g1"}
            ]
            for name in ("train", "val", "locked_test"):
                write_split(root / "splits" / f"{name}.csv", split_rows if name == "train" else [])
            mask_root = root / "splits" / "stage16g"
            mask_root.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        **split_rows[0],
                        "mask_sha256": "a" * 64,
                        "mask_path": "mask.png",
                    }
                ]
            ).to_csv(mask_root / "mask_qualified_train.csv", index=False)
            (mask_root / "mask_inventory.json").write_text(
                json.dumps({"locked_test_evaluated": False}), encoding="utf-8"
            )
            config = yaml.safe_load(
                Path("configs/stage16g_generator_qualification.yaml").read_text(
                    encoding="utf-8"
                )
            )
            config["data"]["root"] = str(root)
            config["targeting"]["candidate_classes"] = ["mel"]
            config["targeting"]["minimum_unique_lesions_per_class"] = 2
            _, errors = validate(config, Path.cwd(), check_hardware=False)
            self.assertTrue(any("mel:" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
