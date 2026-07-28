from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


REQUIRED_RUN_ARTIFACTS = (
    "summary.json",
    "config.resolved.yaml",
    "sampling_plan.json",
    "model_initialization.json",
    "class_counts.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "best.pt",
)
TARGET_CLASSES = ("mel", "akiec", "bkl")
RECIPE_PATHS = (
    ("runtime", "deterministic"),
    ("data", "val_csv"),
    ("data", "test_csv"),
    ("data", "image_size"),
    ("data", "val_size"),
    ("model", "name"),
    ("model", "drop_path_rate"),
    ("augmentation",),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "accumulation_steps"),
    ("training", "lr"),
    ("training", "layer_decay"),
    ("training", "warmup_epochs"),
    ("training", "early_stopping_patience"),
    ("training", "label_smoothing"),
    ("training", "model_ema"),
    ("training", "model_ema_decay"),
    ("training", "model_ema_warmup"),
    ("imbalance", "sampler"),
    ("imbalance", "loss"),
    ("imbalance", "class_weights"),
    ("evaluation", "run_test"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the locked Stage 11B paired protocol.")
    parser.add_argument("--decision", default="configs/stage11_decision.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--report", default="outputs/reports/stage11b_gate.json")
    parser.add_argument("--print-configs", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def validate_stage11_qualification(
    outputs: Path, experiment: str, seeds: list[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    for seed in seeds:
        candidates = sorted((outputs / experiment).glob(f"*_{seed}/summary.json"))
        if len(candidates) != 1:
            errors.append(
                f"{experiment} seed={seed}: expected one completed run, found {len(candidates)}"
            )
            continue
        run_dir = candidates[0].parent
        missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{experiment} seed={seed}: missing {', '.join(missing)}")
            continue
        summary = read_json(run_dir / "summary.json")
        if summary.get("test_evaluated") is not False:
            errors.append(f"{experiment} seed={seed}: locked test was evaluated")
            continue
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": summary.get("best_epoch"),
                "best_metric": summary.get("best_metric"),
            }
        )
    return runs, errors


def compare_recipe(
    baseline: dict[str, Any], arm: dict[str, Any], arm_name: str
) -> list[str]:
    errors: list[str] = []
    for path in RECIPE_PATHS:
        if nested(baseline, path) != nested(arm, path):
            dotted = ".".join(path)
            errors.append(
                f"{arm_name}: {dotted} differs from qualified baseline "
                f"({nested(arm, path)!r} != {nested(baseline, path)!r})"
            )
    return errors


def validate_pair(
    data_root: Path,
    synthetic_config: dict[str, Any],
    replay_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    synthetic_path = data_root / synthetic_config["data"]["train_csv"]
    replay_path = data_root / replay_config["data"]["train_csv"]
    selected_path = data_root / "splits/stage10/selected_synthetic_strict_id.csv"
    replay_rows_path = data_root / "splits/stage10/source_replay_rows_strict_id.csv"
    manifest_path = data_root / "splits/stage10/stage10_strata_manifest.json"
    val_path = data_root / synthetic_config["data"]["val_csv"]
    test_path = data_root / synthetic_config["data"]["test_csv"]
    paths = (synthetic_path, replay_path, selected_path, replay_rows_path, manifest_path, val_path, test_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {}, [f"Missing Stage 11B input: {path}" for path in missing]

    synthetic_train = pd.read_csv(synthetic_path)
    replay_train = pd.read_csv(replay_path)
    selected = pd.read_csv(selected_path)
    replay_rows = pd.read_csv(replay_rows_path)
    validation = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    manifest = read_json(manifest_path)

    synthetic_added = synthetic_train[synthetic_train["is_synthetic"].astype(int) == 1].copy()
    synthetic_base = synthetic_train[synthetic_train["is_synthetic"].astype(int) == 0].copy()
    replay_added = replay_train[replay_train["is_replay"].astype(int) == 1].copy()
    replay_base = replay_train[replay_train["is_replay"].astype(int) == 0].copy()

    if len(synthetic_train) != len(replay_train):
        errors.append("Synthetic and replay arms have different row counts")
    if len(synthetic_added) != 90 or len(replay_added) != 90:
        errors.append(
            f"Expected 90 additions per arm, found synthetic={len(synthetic_added)}, "
            f"replay={len(replay_added)}"
        )
    if Counter(synthetic_base["image_id"].astype(str)) != Counter(replay_base["image_id"].astype(str)):
        errors.append("Base real-image multisets differ between paired arms")
    if normalized_counts(synthetic_train["label"]) != normalized_counts(replay_train["label"]):
        errors.append("Class counts differ between paired arms")
    expected_dose = {label: 30 for label in TARGET_CLASSES}
    if normalized_counts(synthetic_added["label"]) != expected_dose:
        errors.append(f"Synthetic strict-ID dose mismatch: {normalized_counts(synthetic_added['label'])}")
    if normalized_counts(replay_added["label"]) != expected_dose:
        errors.append(f"Replay dose mismatch: {normalized_counts(replay_added['label'])}")
    if set(synthetic_added["image_id"].astype(str)) != set(selected["image_id"].astype(str)):
        errors.append("Synthetic train additions do not exactly match selected strict-ID rows")
    if not (selected["passes_geometry_filter"].astype(int) == 1).all():
        errors.append("Selected strict-ID CSV contains rows that failed the geometry filter")
    if not (selected["stage10_stratum"].astype(str) == "strict_id").all():
        errors.append("Selected synthetic CSV contains a non-strict_id stratum")
    if not (replay_added["sample_weight"].astype(float) == 0.5).all():
        errors.append("Replay additions do not all have sample_weight=0.5")
    if float(synthetic_config["training"]["synthetic_weight"]) != 0.5:
        errors.append("Synthetic arm does not use synthetic_weight=0.5")
    per_class_weight = synthetic_config["training"].get("synthetic_weight_per_class", {})
    if any(float(per_class_weight.get(label, -1)) != 0.5 for label in TARGET_CLASSES):
        errors.append("Synthetic target-class weights are not all 0.5")
    if replay_config["training"].get("use_sample_weights") is not True:
        errors.append("Replay arm does not enable CSV sample weights")

    source_by_synthetic = selected.set_index(selected["image_id"].astype(str))[
        "source_image_id"
    ].astype(str)
    replay_by_synthetic = replay_rows.set_index(
        replay_rows["replay_for_synthetic_image_id"].astype(str)
    )["replay_source_image_id"].astype(str)
    if source_by_synthetic.to_dict() != replay_by_synthetic.to_dict():
        errors.append("Replay source mapping is not one-to-one with selected synthetic sources")

    eval_image_ids = set(validation["image_id"].astype(str)) | set(test["image_id"].astype(str))
    eval_groups = set(validation["group_id"].astype(str)) | set(test["group_id"].astype(str))
    if set(synthetic_base["image_id"].astype(str)) & eval_image_ids:
        errors.append("Real training rows overlap validation/test image IDs")
    if set(replay_added["image_id"].astype(str)) & eval_image_ids:
        errors.append("Replay rows overlap validation/test image IDs")
    if set(synthetic_base["group_id"].astype(str)) & eval_groups:
        errors.append("Real training rows overlap validation/test lesion groups")
    if set(selected["source_group_id"].astype(str)) & eval_groups:
        errors.append("Synthetic source lesion groups overlap validation/test")

    if manifest.get("protocol") != "stage10_equal_dose_geometry_strata":
        errors.append("Unexpected Stage 10 manifest protocol")
    if int(manifest.get("rows_per_stratum", 0)) != 90:
        errors.append("Stage 10 manifest does not declare 90 rows per stratum")
    if float(manifest.get("sample_weight", -1)) != 0.5:
        errors.append("Stage 10 manifest sample weight is not 0.5")

    report = {
        "row_counts": {
            "base_real": int(len(synthetic_base)),
            "synthetic_added": int(len(synthetic_added)),
            "replay_added": int(len(replay_added)),
            "total_per_arm": int(len(synthetic_train)),
        },
        "class_counts_per_arm": normalized_counts(synthetic_train["label"]),
        "unique_replay_sources": int(replay_added["replay_source_image_id"].nunique()),
        "max_source_reuse": int(replay_added["replay_source_image_id"].value_counts().max()),
        "hashes": {str(path): sha256(path) for path in paths},
    }
    return report, errors


def main() -> None:
    args = parse_args()
    decision = yaml.safe_load(Path(args.decision).read_text(encoding="utf-8"))
    stage11 = decision["stage11"]
    errors: list[str] = []

    if decision.get("status") != "approved" or decision.get("stage10_analysis_reviewed") is not True:
        errors.append("Stage 10/11 decision has not been approved")
    if stage11.get("stage11a_status") != "complete":
        errors.append("Stage 11A qualification is not complete")
    if stage11.get("stage11b_status") not in {"approved_not_started", "running"}:
        errors.append(f"Stage 11B status is not launchable: {stage11.get('stage11b_status')}")

    config_paths = [str(path) for path in stage11.get("stage11b_configs", [])]
    if len(config_paths) != 2 or any(not Path(path).is_file() for path in config_paths):
        errors.append("Stage 11B requires exactly two existing config files")

    baseline_path = str(stage11["selected_baseline"])
    baseline = load_config(baseline_path)
    configs = {Path(path).name: load_config(path) for path in config_paths if Path(path).is_file()}
    synthetic_items = [(name, cfg) for name, cfg in configs.items() if "synthetic" in name]
    replay_items = [(name, cfg) for name, cfg in configs.items() if "replay" in name]
    pair_report: dict[str, Any] = {}
    if len(synthetic_items) != 1 or len(replay_items) != 1:
        errors.append("Could not identify one synthetic and one replay Stage 11B config")
    else:
        for name, config in configs.items():
            errors.extend(compare_recipe(baseline, config, name))
            if config["evaluation"]["run_test"] is not False:
                errors.append(f"{name}: locked test evaluation must remain disabled")
        pair_report, pair_errors = validate_pair(
            Path(args.data_root), synthetic_items[0][1], replay_items[0][1]
        )
        errors.extend(pair_errors)

    baseline_experiment = baseline["experiment"]["name"]
    qualification_runs, qualification_errors = validate_stage11_qualification(
        Path(args.outputs), baseline_experiment, [int(seed) for seed in stage11["seeds"]]
    )
    errors.extend(qualification_errors)

    report = {
        "gate_open": not errors,
        "comparison": stage11.get("stage11b_comparison"),
        "selected_baseline": baseline_path,
        "configs": config_paths,
        "seeds": [int(seed) for seed in stage11["stage11b_seeds"]],
        "qualified_baseline_runs": qualification_runs,
        "pair_audit": pair_report,
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        raise SystemExit("Stage 11B gate closed: " + "; ".join(errors))
    if args.print_configs:
        print("\n".join(config_paths))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
