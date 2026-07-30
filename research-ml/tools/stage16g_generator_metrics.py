from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_arm(frame: pd.DataFrame, gates: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "images": len(frame),
        "hash_failures": int((~frame["output_hash_valid"]).sum()),
        "label_agreement": float(frame["label_agreement"].mean()),
        "mean_true_label_probability": float(frame["true_label_probability"].mean()),
        "mean_source_cosine_similarity": float(frame["source_cosine_similarity"].mean()),
        "near_duplicate_fraction": float(
            (
                frame["source_cosine_similarity"]
                > float(gates["maximum_source_cosine_similarity"])
            ).mean()
        ),
        "mean_mask_iou": float(frame["regenerated_mask_iou"].mean()),
        "mean_mask_dice": float(frame["regenerated_mask_dice"].mean()),
        "mean_outside_mask_mae": float(frame["outside_mask_mae"].mean()),
        "mean_inside_mask_mae": float(frame["inside_mask_mae"].mean()),
    }
    summary["technical_gate_passed"] = bool(
        summary["images"] == int(gates["expected_images_per_arm"])
        and summary["hash_failures"] <= int(gates["maximum_failed_images"])
        and summary["label_agreement"]
        >= float(gates["minimum_independent_label_agreement"])
        and summary["mean_mask_iou"] >= float(gates["minimum_regenerated_mask_iou"])
        and summary["mean_outside_mask_mae"]
        <= float(gates["maximum_outside_mask_mae"])
        and summary["near_duplicate_fraction"] == 0.0
    )
    return summary

