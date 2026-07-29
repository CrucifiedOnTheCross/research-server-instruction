from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_isic2019 import CLASS_NAMES, EXPECTED_IMAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate curated ISIC 2019.")
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument(
        "--report", default="outputs/reports/stage16a_data_gate.json"
    )
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def validate(data_root: Path, check_files: bool) -> tuple[dict, list[str]]:
    errors: list[str] = []
    summary_path = data_root / "dataset_summary.json"
    manifest_path = data_root / "manifest.csv"
    split_paths = {
        name: data_root / "splits" / f"{name}.csv"
        for name in ("train", "val", "locked_test")
    }
    required = [summary_path, manifest_path, *split_paths.values()]
    missing = [path for path in required if not path.is_file()]
    if missing:
        return {}, [f"Missing Stage 16A artifact: {path}" for path in missing]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
    splits = {
        name: pd.read_csv(path, dtype={"image_id": str})
        for name, path in split_paths.items()
    }
    expected_classes = set(CLASS_NAMES)
    if summary.get("protocol") != "stage16a_isic2019_curated_v1":
        errors.append("Unexpected dataset protocol")
    if summary.get("locked_test_evaluated") is not False:
        errors.append("Dataset summary does not keep locked test closed")
    if len(manifest) + int(summary.get("quarantined_label_conflicts", 0)) != EXPECTED_IMAGES:
        errors.append("Usable plus quarantined rows do not match official size")
    if set(manifest["label"].astype(str)) != expected_classes:
        errors.append("Manifest class ontology differs from ISIC 2019")
    if manifest["image_id"].astype(str).nunique() != len(manifest):
        errors.append("Manifest image IDs are not unique")
    if not (manifest["decode_ok"].astype(int) == 1).all():
        errors.append("Manifest contains undecodable images")
    if not (manifest["is_synthetic"].astype(int) == 0).all():
        errors.append("Stage 16A real manifest contains synthetic images")
    if not (manifest["sample_weight"].astype(float) == 1.0).all():
        errors.append("Stage 16A real sample weights differ from one")

    combined_ids: set[str] = set()
    for name, frame in splits.items():
        if set(frame["label"].astype(str)) != expected_classes:
            errors.append(f"{name}: one or more classes are absent")
        ids = set(frame["image_id"].astype(str))
        if combined_ids & ids:
            errors.append(f"{name}: image IDs overlap a previous split")
        combined_ids |= ids
        if name != "train" and (frame["is_synthetic"].astype(int) != 0).any():
            errors.append(f"{name}: synthetic images found in evaluation split")
    if combined_ids != set(manifest["image_id"].astype(str)):
        errors.append("Split image IDs do not exactly cover the usable manifest")

    for column in ("group_id", "lesion_id", "sha256", "visual_hash"):
        values: dict[str, set[str]] = {}
        for name, frame in splits.items():
            values[name] = {
                value
                for value in frame[column].fillna("").astype(str)
                if value and value.lower() != "nan"
            }
        names = list(values)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = values[left] & values[right]
                if overlap:
                    errors.append(
                        f"{column}: {len(overlap)} values overlap {left}/{right}"
                    )

    missing_files: list[str] = []
    if check_files:
        missing_files = [
            path
            for path in manifest["image_path"].astype(str)
            if not (data_root / path).is_file()
        ]
        if missing_files:
            errors.append(f"{len(missing_files)} manifest image files are missing")

    details = {
        "usable_images": len(manifest),
        "unique_groups": int(manifest["group_id"].nunique()),
        "classes": sorted(manifest["label"].astype(str).unique()),
        "splits": {
            name: {
                "images": len(frame),
                "groups": int(frame["group_id"].nunique()),
                "class_counts": {
                    str(label): int(count)
                    for label, count in frame["label"].value_counts().sort_index().items()
                },
            }
            for name, frame in splits.items()
        },
        "missing_files": len(missing_files),
    }
    return details, errors


def main() -> None:
    args = parse_args()
    details, errors = validate(Path(args.data_root).resolve(), args.check_files)
    report = {
        "protocol": "stage16a_isic2019_data_gate",
        "gate_open": not errors,
        "locked_test_evaluated": False,
        "details": details,
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
