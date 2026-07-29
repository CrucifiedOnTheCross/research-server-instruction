from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from tools.check_stage11b_gate import compare_recipe


TARGET_CLASSES = ("mel", "akiec", "bkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the locked Stage 13B paired protocol.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument(
        "--manifest",
        default="outputs/reports/stage13_bkl_expansion_selection/stage13_selection_manifest.json",
    )
    parser.add_argument(
        "--synthetic-config",
        default="configs/ham10000_stage13b_synthetic_coverage_convnext_small_384.yaml",
    )
    parser.add_argument(
        "--replay-config",
        default="configs/ham10000_stage13b_replay_coverage_convnext_small_384.yaml",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/ham10000_stage11_convnext_small_regularized_384.yaml",
    )
    parser.add_argument("--report", default="outputs/reports/stage13b_gate.json")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("protocol") != "stage13_multiencoder_coverage_targeted":
        errors.append("Unexpected Stage 13 selection protocol")
    if manifest.get("locked_test_used") is not False:
        errors.append("Stage 13 selection used the locked test")
    gate = manifest.get("gate", {})
    if gate.get("gate_open") is not True:
        errors.append("Stage 13 geometry gate is not open")
    checks = gate.get("checks", {})
    failed = [name for name, value in checks.items() if value is not True and name != "locked_test_used"]
    if failed:
        errors.append(f"Stage 13 gate contains failed checks: {failed}")
    return errors


def validate_pair(
    data_root: Path,
    synthetic_config: dict[str, Any],
    replay_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    split_dir = data_root / "splits/stage13_bkl_expansion"
    selected_path = split_dir / "selected_synthetic_coverage_targeted.csv"
    replay_rows_path = split_dir / "source_replay_rows_coverage_targeted.csv"
    synthetic_path = data_root / synthetic_config["data"]["train_csv"]
    replay_path = data_root / replay_config["data"]["train_csv"]
    val_path = data_root / synthetic_config["data"]["val_csv"]
    test_path = data_root / synthetic_config["data"]["test_csv"]
    paths = (selected_path, replay_rows_path, synthetic_path, replay_path, val_path, test_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {}, [f"Missing Stage 13B input: {path}" for path in missing]

    selected = pd.read_csv(selected_path)
    replay_rows = pd.read_csv(replay_rows_path)
    synthetic_train = pd.read_csv(synthetic_path)
    replay_train = pd.read_csv(replay_path)
    validation = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    synthetic_added = synthetic_train[synthetic_train["is_synthetic"].astype(int) == 1]
    synthetic_base = synthetic_train[synthetic_train["is_synthetic"].astype(int) == 0]
    replay_added = replay_train[replay_train["is_replay"].astype(int) == 1]
    replay_base = replay_train[replay_train["is_replay"].astype(int) == 0]
    expected = {label: 30 for label in TARGET_CLASSES}

    if class_counts(selected) != expected:
        errors.append(f"Selected dose mismatch: {class_counts(selected)}")
    if len(selected) != 90 or selected["source_image_id"].astype(str).nunique() != 90:
        errors.append("Stage 13B requires 90 selected rows from 90 unique sources")
    if not (selected["stage13_selected"].astype(int) == 1).all():
        errors.append("Selected CSV contains a row not marked stage13_selected")
    if set(synthetic_added["image_id"].astype(str)) != set(selected["image_id"].astype(str)):
        errors.append("Synthetic additions do not exactly match Stage 13 selection")
    if len(replay_added) != 90 or class_counts(replay_added) != expected:
        errors.append("Replay arm does not contain the matched 30-per-class dose")
    if Counter(synthetic_base["image_id"].astype(str)) != Counter(replay_base["image_id"].astype(str)):
        errors.append("Base real-image multisets differ between paired arms")
    if class_counts(synthetic_train) != class_counts(replay_train):
        errors.append("Paired arms have different class counts")
    if not (replay_added["sample_weight"].astype(float) == 0.5).all():
        errors.append("Replay rows do not all use sample_weight=0.5")

    source_by_synthetic = selected.set_index(selected["image_id"].astype(str))[
        "source_image_id"
    ].astype(str)
    replay_by_synthetic = replay_rows.set_index(
        replay_rows["replay_for_synthetic_image_id"].astype(str)
    )["replay_source_image_id"].astype(str)
    if source_by_synthetic.to_dict() != replay_by_synthetic.to_dict():
        errors.append("Replay source mapping differs from Stage 13 selection")

    eval_ids = set(validation["image_id"].astype(str)) | set(test["image_id"].astype(str))
    eval_groups = set(validation["group_id"].astype(str)) | set(test["group_id"].astype(str))
    if set(synthetic_base["image_id"].astype(str)) & eval_ids:
        errors.append("Real training rows overlap validation/test image IDs")
    if set(selected["source_group_id"].astype(str)) & eval_groups:
        errors.append("Synthetic source lesions overlap validation/test groups")
    if set(replay_added["replay_source_image_id"].astype(str)) & eval_ids:
        errors.append("Replay source IDs overlap validation/test image IDs")

    if float(synthetic_config["training"]["synthetic_weight"]) != 0.5:
        errors.append("Synthetic arm weight is not 0.5")
    if replay_config["training"].get("use_sample_weights") is not True:
        errors.append("Replay arm does not enable CSV sample weights")

    return {
        "selected_rows": int(len(selected)),
        "selected_by_class": class_counts(selected),
        "unique_sources": int(selected["source_image_id"].astype(str).nunique()),
        "total_rows_per_arm": int(len(synthetic_train)),
    }, errors


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    manifest = read_json(project_root / args.manifest)
    baseline = load_config(project_root / args.baseline_config)
    synthetic = load_config(project_root / args.synthetic_config)
    replay = load_config(project_root / args.replay_config)
    errors = validate_manifest(manifest)
    errors.extend(compare_recipe(baseline, synthetic, "synthetic"))
    errors.extend(compare_recipe(baseline, replay, "replay"))
    pair, pair_errors = validate_pair(data_root, synthetic, replay)
    errors.extend(pair_errors)
    report = {
        "protocol": "stage13b_confirmatory_gate",
        "gate_open": not errors,
        "locked_test_used": False,
        "configs": [args.replay_config, args.synthetic_config],
        "pair": pair,
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
