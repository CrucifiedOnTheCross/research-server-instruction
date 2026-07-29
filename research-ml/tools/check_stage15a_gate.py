from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from tools.check_stage11b_gate import compare_recipe
from tools.prepare_stage15a_variants import (
    EXPECTED_PER_CLASS,
    SD15_REVISION,
    TARGET_CLASSES,
)


CONFIGS = {
    "original_replay": "configs/ham10000_stage15a_original_replay_convnext_small_384.yaml",
    "offline_crop": "configs/ham10000_stage15a_offline_crop_convnext_small_384.yaml",
    "vae_roundtrip": "configs/ham10000_stage15a_vae_roundtrip_convnext_small_384.yaml",
    "img2img_strength05": "configs/ham10000_stage15a_img2img_strength05_convnext_small_384.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 15A causal arms.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/ham10000"
    )
    parser.add_argument(
        "--report", default="outputs/reports/stage15a_gate.json"
    )
    return parser.parse_args()


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def validate_additions(
    frame: pd.DataFrame, name: str, data_root: Path
) -> list[str]:
    errors: list[str] = []
    expected = {label: EXPECTED_PER_CLASS for label in TARGET_CLASSES}
    if len(frame) != 90 or counts(frame) != expected:
        errors.append(f"{name}: expected balanced 90 rows, found {len(frame)} {counts(frame)}")
    if frame["source_image_id"].astype(str).nunique() != 90:
        errors.append(f"{name}: source image IDs are not unique")
    if frame["source_group_id"].astype(str).nunique() != 90:
        errors.append(f"{name}: source lesion groups are not unique")
    missing_images = [
        path
        for path in frame["image_path"].astype(str)
        if not (data_root / path).is_file()
    ]
    if missing_images:
        errors.append(f"{name}: {len(missing_images)} image files are missing")
    return errors


def validate_protocol(
    project_root: Path, data_root: Path, configs: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = data_root / "splits/stage15a/preparation_manifest.json"
    selected_path = (
        data_root
        / "splits/stage13_bkl_expansion/selected_synthetic_coverage_targeted.csv"
    )
    replay_rows_path = (
        data_root
        / "splits/stage13_bkl_expansion/source_replay_rows_coverage_targeted.csv"
    )
    required = [manifest_path, selected_path, replay_rows_path]
    for name in ("offline_crop", "vae_roundtrip", "img2img_strength05"):
        required.extend(
            [
                data_root / f"splits/stage15a/{name}_rows.csv",
                data_root / f"splits/stage15a/train_{name}.csv",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {}, [f"Missing Stage 15A artifact: {path}" for path in missing]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "stage15a_causal_generator_decomposition":
        errors.append("Unexpected preparation protocol")
    if manifest.get("locked_test_used") is not False:
        errors.append("Preparation manifest does not keep locked test closed")
    if manifest.get("sd15_revision") != SD15_REVISION:
        errors.append("SD1.5 revision is not the predeclared pinned revision")

    selected = pd.read_csv(selected_path)
    replay = pd.read_csv(replay_rows_path)
    additions = {
        name: pd.read_csv(data_root / f"splits/stage15a/{name}_rows.csv")
        for name in ("offline_crop", "vae_roundtrip", "img2img_strength05")
    }
    for name, frame in additions.items():
        errors.extend(validate_additions(frame, name, data_root))

    expected_sources = set(selected["source_image_id"].astype(str))
    expected_groups = set(selected["source_group_id"].astype(str))
    for name, frame in additions.items():
        if set(frame["source_image_id"].astype(str)) != expected_sources:
            errors.append(f"{name}: source image set differs from selected Stage 13B set")
        if set(frame["source_group_id"].astype(str)) != expected_groups:
            errors.append(f"{name}: source lesion set differs from selected Stage 13B set")
    if set(replay["replay_source_image_id"].astype(str)) != expected_sources:
        errors.append("Original replay sources differ from the Stage 15A source set")
    if not (
        additions["img2img_strength05"]["strength"].astype(float) == 0.05
    ).all():
        errors.append("Img2img arm contains a strength other than 0.05")

    base_multisets: dict[str, Counter[str]] = {}
    total_counts: dict[str, dict[str, int]] = {}
    for name, config in configs.items():
        train_path = data_root / config["data"]["train_csv"]
        if not train_path.is_file():
            errors.append(f"{name}: missing train CSV {train_path}")
            continue
        train = pd.read_csv(train_path)
        if name == "original_replay":
            base = train[train["is_replay"].astype(int) == 0]
            added = train[train["is_replay"].astype(int) == 1]
            if not (added["sample_weight"].astype(float) == 0.5).all():
                errors.append("Original replay weights are not 0.5")
            if config["training"].get("use_sample_weights") is not True:
                errors.append("Original replay does not enable CSV sample weights")
        else:
            base = train[train["is_synthetic"].astype(int) == 0]
            added = train[train["is_synthetic"].astype(int) == 1]
            if float(config["training"].get("synthetic_weight", -1)) != 0.5:
                errors.append(f"{name}: synthetic weight is not 0.5")
        if len(added) != 90:
            errors.append(f"{name}: expected 90 additions, found {len(added)}")
        base_multisets[name] = Counter(base["image_id"].astype(str))
        total_counts[name] = counts(train)

    if base_multisets and len({tuple(sorted(value.items())) for value in base_multisets.values()}) != 1:
        errors.append("Base real-image multisets differ across Stage 15A arms")
    if total_counts and len({tuple(sorted(value.items())) for value in total_counts.values()}) != 1:
        errors.append("Class counts differ across Stage 15A arms")

    reference = configs["original_replay"]
    for name, config in configs.items():
        errors.extend(compare_recipe(reference, config, name))
        if config["training"]["monitor"] != "val/auprc_ovr_macro":
            errors.append(f"{name}: checkpoint monitor is not val/auprc_ovr_macro")
        if config["evaluation"].get("run_test") is not False:
            errors.append(f"{name}: locked test evaluation is enabled")

    val_groups = set(
        pd.read_csv(data_root / reference["data"]["val_csv"])["group_id"].astype(str)
    )
    test_groups = set(
        pd.read_csv(data_root / reference["data"]["test_csv"])["group_id"].astype(str)
    )
    if expected_groups & (val_groups | test_groups):
        errors.append("Stage 15A source lesions overlap validation or locked test")

    return {
        "source_rows": int(len(selected)),
        "unique_source_images": int(selected["source_image_id"].nunique()),
        "unique_source_groups": int(selected["source_group_id"].nunique()),
        "class_counts": counts(selected),
        "total_counts_by_arm": total_counts,
        "sd15_revision": manifest.get("sd15_revision"),
    }, errors


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    configs = {
        name: load_config(project_root / path) for name, path in CONFIGS.items()
    }
    protocol, errors = validate_protocol(project_root, data_root, configs)
    report = {
        "protocol": "stage15a_causal_generator_decomposition_gate",
        "gate_open": not errors,
        "locked_test_used": False,
        "configs": CONFIGS,
        "details": protocol,
        "errors": errors,
    }
    report_path = project_root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
