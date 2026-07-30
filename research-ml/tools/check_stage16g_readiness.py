from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness gate for Stage 16G generator pilots."
    )
    parser.add_argument(
        "--config", default="configs/stage16g_generator_qualification.yaml"
    )
    parser.add_argument(
        "--report", default="outputs/reports/stage16g_readiness.json"
    )
    parser.add_argument("--skip-hardware", action="store_true")
    return parser.parse_args()


def _read_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype={"image_id": str})
    return set(frame["image_id"].astype(str).str.upper())


def query_free_vram_gib() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value.strip()) / 1024 for value in result.stdout.splitlines()]
    return max(values) if values else None


def validate(
    config: dict[str, Any], repository_root: Path, check_hardware: bool
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = (repository_root / data_root).resolve()
    paths = {
        key: data_root / config["data"][key]
        for key in (
            "train_split",
            "validation_split",
            "locked_test_split",
            "mask_manifest",
            "mask_inventory",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {"missing_artifacts": missing}, [
            f"Missing Stage 16G artifact: {path}" for path in missing
        ]

    train_ids = _read_ids(paths["train_split"])
    validation_ids = _read_ids(paths["validation_split"])
    test_ids = _read_ids(paths["locked_test_split"])
    masks = pd.read_csv(paths["mask_manifest"], dtype={"image_id": str})
    inventory = json.loads(paths["mask_inventory"].read_text(encoding="utf-8"))
    mask_ids = set(masks["image_id"].astype(str).str.upper())

    if mask_ids - train_ids:
        errors.append("Mask manifest contains IDs outside train")
    if mask_ids & (validation_ids | test_ids):
        errors.append("Mask manifest leaks validation or locked-test IDs")
    if masks["image_id"].nunique() != len(masks):
        errors.append("Mask manifest image IDs are not unique")
    if "group_id" not in masks:
        errors.append("Mask manifest has no lesion-aware group_id")
    if masks.get("mask_sha256", pd.Series(dtype=str)).astype(str).str.len().ne(64).any():
        errors.append("One or more mask SHA256 values are invalid")
    if inventory.get("locked_test_evaluated") is not False:
        errors.append("Mask inventory does not keep locked test closed")

    minimum = int(config["targeting"]["minimum_unique_lesions_per_class"])
    class_counts: dict[str, int] = {}
    for label in config["targeting"]["candidate_classes"]:
        rows = masks[masks["label"].astype(str) == str(label)]
        unique_lesions = int(rows["group_id"].fillna(rows["image_id"]).nunique())
        class_counts[str(label)] = unique_lesions
        if unique_lesions < minimum:
            errors.append(
                f"{label}: {unique_lesions} mask-qualified train lesions < {minimum}"
            )

    hardware: dict[str, float | None] = {}
    if check_hardware:
        free_disk = shutil.disk_usage(data_root).free / 1024**3
        free_vram = query_free_vram_gib()
        hardware = {
            "free_disk_gib": round(free_disk, 2),
            "free_vram_gib": None if free_vram is None else round(free_vram, 2),
        }
        if free_disk < float(config["qualification"]["gpu"]["minimum_free_disk_gib"]):
            errors.append("Insufficient free disk for generator qualification")
        if free_vram is None:
            errors.append("nvidia-smi did not report free VRAM")
        elif free_vram < float(
            config["qualification"]["gpu"]["minimum_free_vram_gib"]
        ):
            errors.append("Insufficient free VRAM for generator qualification")

    details = {
        "protocol": config["protocol"],
        "locked_test_evaluated": False,
        "mask_qualified_images": len(masks),
        "mask_qualified_unique_groups": (
            int(masks["group_id"].nunique()) if "group_id" in masks else 0
        ),
        "candidate_class_unique_lesions": class_counts,
        "hardware": hardware,
        "enabled_generators": [
            name
            for name, value in config["generators"].items()
            if bool(value.get("enabled"))
        ],
    }
    return details, errors


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    details, errors = validate(config, repository_root, not args.skip_hardware)
    report = {
        "gate_open": not errors,
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
