from __future__ import annotations

import argparse
import json
from pathlib import Path

import fiftyone as fo
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a persistent FiftyOne audit dataset for Stage 16G masks."
    )
    parser.add_argument(
        "--csv",
        default=(
            "/srv/research/projects/default/research-ml/outputs/reports/"
            "stage16g_segmentation_audit/pseudo_mask_manifest.csv"
        ),
    )
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument(
        "--dataset-name", default="isic2019-stage16g-mask-audit"
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else root / path)


def build_dataset(
    frame: pd.DataFrame, root: Path, name: str, replace: bool
) -> fo.Dataset:
    if name in fo.list_datasets():
        if not replace:
            raise RuntimeError(f"Dataset already exists: {name}; use --replace")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    dataset.persistent = True
    samples: list[fo.Sample] = []
    for _, row in frame.iterrows():
        image_path = resolve(root, str(row["image_path"]))
        mask_path = resolve(root, str(row["pseudo_mask_path"]))
        if not Path(image_path).is_file() or not Path(mask_path).is_file():
            raise RuntimeError(f"Missing image/mask pair: {image_path} / {mask_path}")
        sample = fo.Sample(filepath=image_path)
        sample["diagnosis"] = fo.Classification(label=str(row["label"]))
        sample["pseudo_mask"] = fo.Segmentation(mask_path=mask_path)
        sample["image_id"] = str(row["image_id"])
        sample["group_id"] = str(row["group_id"])
        sample["source"] = str(row["source"])
        sample["passed"] = bool(int(row["pseudo_mask_passed"]))
        for field in (
            "area_fraction",
            "foreground_confidence",
            "largest_component_fraction",
            "border_foreground_fraction",
        ):
            sample[field] = float(row[field])
        sample["component_count"] = int(row["component_count"])
        samples.append(sample)
    dataset.add_samples(samples)
    dataset.info = {
        "protocol": "stage16g_segmentation_audit_v1",
        "source_csv": str(Path(frame.attrs.get("source_csv", ""))),
        "rows": len(frame),
        "locked_test_evaluated": False,
        "purpose": "Blinded visual audit before generator qualification",
    }
    dataset.save()
    dataset.save_view("01_passed", dataset.match(fo.ViewField("passed") == True))
    dataset.save_view("02_failed", dataset.match(fo.ViewField("passed") == False))
    dataset.save_view(
        "03_fragmented",
        dataset.match(fo.ViewField("component_count") > 3),
    )
    dataset.save_view(
        "04_border_touching",
        dataset.sort_by("border_foreground_fraction", reverse=True),
    )
    dataset.save_view(
        "05_low_confidence",
        dataset.sort_by("foreground_confidence"),
    )
    return dataset


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    frame.attrs["source_csv"] = args.csv
    dataset = build_dataset(
        frame, Path(args.data_root), args.dataset_name, args.replace
    )
    print(
        json.dumps(
            {
                "dataset": dataset.name,
                "samples": len(dataset),
                "saved_views": dataset.list_saved_views(),
                "app": "http://10.200.1.180:5151",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
