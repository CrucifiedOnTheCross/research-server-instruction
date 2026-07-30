from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply preregistered train-only morphology gates to pseudo-masks."
    )
    parser.add_argument(
        "--config", default="configs/stage16g_generator_qualification.yaml"
    )
    parser.add_argument(
        "--audit-root", default="outputs/reports/stage16g_segmentation_audit"
    )
    return parser.parse_args()


def select_masks(
    pseudo: pd.DataFrame,
    qualification: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    maximum_border_fraction: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    lower_area = float(qualification["gt_area_fraction"].quantile(lower_quantile))
    upper_area = float(qualification["gt_area_fraction"].quantile(upper_quantile))
    selected = pseudo[
        (pseudo["pseudo_mask_passed"].astype(int) == 1)
        & pseudo["area_fraction"].between(lower_area, upper_area, inclusive="both")
        & (
            pseudo["border_foreground_fraction"].astype(float)
            <= maximum_border_fraction
        )
    ].copy()
    selected["generator_mask_eligible"] = 1
    summary = {
        "locked_test_evaluated": False,
        "qualification_area_lower": lower_area,
        "qualification_area_upper": upper_area,
        "maximum_border_fraction": maximum_border_fraction,
        "input_candidates": len(pseudo),
        "segmentation_gate_passed": int(
            (pseudo["pseudo_mask_passed"].astype(int) == 1).sum()
        ),
        "generator_mask_eligible": len(selected),
        "unique_groups": int(selected["group_id"].nunique()),
    }
    return selected, summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.audit_root)
    pseudo = pd.read_csv(root / "pseudo_mask_manifest.csv")
    qualification = pd.read_csv(root / "qualification_per_image.csv")
    settings = config["segmentation_bootstrap"]
    lower, upper = settings["pseudo_mask_area_reference_quantiles"]
    selected, summary = select_masks(
        pseudo,
        qualification,
        float(lower),
        float(upper),
        float(settings["pseudo_mask_maximum_border_fraction"]),
    )
    selected.to_csv(root / "generator_mask_eligible.csv", index=False)
    class_counts = (
        selected.groupby("label")
        .agg(images=("image_id", "size"), groups=("group_id", "nunique"))
        .reset_index()
    )
    class_counts.to_csv(root / "generator_mask_eligible_by_class.csv", index=False)
    summary["class_counts"] = {
        str(row["label"]): int(row["images"])
        for _, row in class_counts.iterrows()
    }
    (root / "generator_mask_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
