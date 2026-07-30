from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from tools.check_stage16g_readiness import validate
from tools.stage16g_mask_metrics import (
    deterministic_group_sample,
    mask_characteristics,
    overlap_metrics,
)
from tools.prepare_stage16g_masks import (
    build_train_mask_manifest,
    choose_mask_member,
    extract_selected_masks,
    image_id_from_mask_name,
)
from tools.prepare_stage16g_segmentation_data import split_segmentation_rows
from tools.select_stage16g_generator_masks import select_masks
from tools.prepare_stage16g_generator_smoke import select_morphology_anchors
from tools.generate_stage16g_generator_smoke import prepare_mask


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "label", "group_id"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


class Stage16GProtocolTests(unittest.TestCase):
    def test_generator_smoke_selection_is_balanced_and_group_unique(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "image_id": f"{label}_{index}",
                    "group_id": f"{label}_g{index}",
                    "label": label,
                    "area_fraction": index / 10,
                }
                for label in ("mel", "scc")
                for index in range(1, 7)
            ]
        )
        selected = select_morphology_anchors(
            rows, ["mel", "scc"], anchors_per_class=4, morphology_field="area_fraction"
        )
        self.assertEqual(selected["label"].value_counts().to_dict(), {"mel": 4, "scc": 4})
        self.assertEqual(selected["group_id"].nunique(), 8)

    def test_generator_smoke_selection_fails_for_small_class(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "image_id": "mel_1",
                    "group_id": "mel_g1",
                    "label": "mel",
                    "area_fraction": 0.2,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "only 1"):
            select_morphology_anchors(
                rows, ["mel"], anchors_per_class=2, morphology_field="area_fraction"
            )

    def test_inpainting_mask_is_binary_before_feathering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[3:5, 3:5] = 255
            Image.fromarray(mask).save(path)
            prepared = np.asarray(prepare_mask(path, 8, dilation=1, blur_radius=0))
            self.assertEqual(set(np.unique(prepared)), {0, 255})
            self.assertGreater(int((prepared > 0).sum()), 4)

    def test_mask_metrics_reward_exact_overlap(self) -> None:
        truth = np.zeros((8, 8), dtype=bool)
        truth[2:6, 2:6] = True
        metrics = overlap_metrics(truth, truth)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)

    def test_mask_characteristics_detect_fragmentation(self) -> None:
        probability = np.zeros((8, 8), dtype=np.float32)
        probability[1:3, 1:3] = 0.9
        probability[5:7, 5:7] = 0.9
        metrics = mask_characteristics(probability, threshold=0.5)
        self.assertEqual(metrics["component_count"], 2)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 0.5)

    def test_pseudo_mask_sampling_is_group_unique(self) -> None:
        rows = [
            {"image_id": "a", "group_id": "g1", "label": "mel"},
            {"image_id": "b", "group_id": "g1", "label": "mel"},
            {"image_id": "c", "group_id": "g2", "label": "mel"},
        ]
        selected = deterministic_group_sample(rows, ["mel"], maximum=10)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["group_id"] for row in selected}), 2)

    def test_generator_mask_gate_uses_qualification_area_range(self) -> None:
        qualification = pd.DataFrame({"gt_area_fraction": [0.1, 0.2, 0.8, 0.9]})
        pseudo = pd.DataFrame(
            [
                {
                    "image_id": "good",
                    "group_id": "g1",
                    "pseudo_mask_passed": 1,
                    "area_fraction": 0.5,
                    "border_foreground_fraction": 0.0,
                },
                {
                    "image_id": "border",
                    "group_id": "g2",
                    "pseudo_mask_passed": 1,
                    "area_fraction": 0.5,
                    "border_foreground_fraction": 0.2,
                },
                {
                    "image_id": "large",
                    "group_id": "g3",
                    "pseudo_mask_passed": 1,
                    "area_fraction": 0.99,
                    "border_foreground_fraction": 0.0,
                },
            ]
        )
        selected, summary = select_masks(
            pseudo,
            qualification,
            lower_quantile=0.0,
            upper_quantile=1.0,
            maximum_border_fraction=0.1,
        )
        self.assertEqual(selected["image_id"].tolist(), ["good"])
        self.assertEqual(summary["generator_mask_eligible"], 1)
        self.assertFalse(summary["locked_test_evaluated"])
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

    def test_segmentation_bootstrap_excludes_stage16_eval_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = {}
            mask_rows = []
            for identifier in (
                "ISIC_0000001",
                "ISIC_0000002",
                "ISIC_0000003",
                "ISIC_0000004",
            ):
                image = root / f"{identifier}.jpg"
                image.write_bytes(identifier.encode())
                image_paths[identifier] = image
                mask_rows.append(
                    {
                        "image_id": identifier,
                        "mask_path": f"{identifier}.png",
                        "mask_sha256": "a" * 64,
                    }
                )
            splits = {
                "train": [
                    {
                        "image_id": "ISIC_0000001",
                        "sha256": "train",
                    }
                ],
                "val": [
                    {
                        "image_id": "ISIC_0000002",
                        "sha256": "val",
                    }
                ],
                "locked_test": [
                    {
                        "image_id": "ISIC_0000003",
                        "sha256": "test",
                    }
                ],
            }
            train, qualification, report = split_segmentation_rows(
                mask_rows, splits, image_paths, root
            )
            self.assertEqual(
                [row["image_id"] for row in qualification], ["ISIC_0000001"]
            )
            self.assertEqual([row["image_id"] for row in train], ["ISIC_0000004"])
            self.assertEqual(
                report["excluded_stage16_validation_or_test_ids"], 2
            )
            self.assertFalse(report["locked_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
