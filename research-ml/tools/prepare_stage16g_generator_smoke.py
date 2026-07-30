from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leak-safe Stage 16G generator smoke anchors.")
    parser.add_argument("--config", default="configs/stage16g_generator_smoke.yaml")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_morphology_anchors(
    rows: pd.DataFrame,
    target_classes: list[str],
    anchors_per_class: int,
    morphology_field: str,
) -> pd.DataFrame:
    selected: list[pd.Series] = []
    quantiles = [(index + 0.5) / anchors_per_class for index in range(anchors_per_class)]
    for label in target_classes:
        pool = rows[rows["label"] == label].copy()
        pool = pool.sort_values([morphology_field, "group_id", "image_id"])
        pool = pool.drop_duplicates("group_id", keep="first")
        if len(pool) < anchors_per_class:
            raise RuntimeError(f"{label}: only {len(pool)} unique eligible groups")
        values = pool[morphology_field].astype(float)
        used: set[str] = set()
        for quantile in quantiles:
            target = float(values.quantile(quantile))
            candidates = pool.assign(distance=(values - target).abs()).sort_values(
                ["distance", morphology_field, "group_id", "image_id"]
            )
            row = next(
                candidate
                for _, candidate in candidates.iterrows()
                if str(candidate["group_id"]) not in used
            )
            used.add(str(row["group_id"]))
            selected.append(row.drop(labels=["distance"], errors="ignore"))
    output = pd.DataFrame(selected).reset_index(drop=True)
    output["anchor_index"] = output.groupby("label").cumcount()
    return output


def forbidden_ids_and_hashes(data_root: Path, config: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    for key in ("validation_split", "locked_test_split"):
        split = pd.read_csv(data_root / config["data"][key])
        ids.update(split["image_id"].astype(str))
        if "sha256" in split:
            hashes.update(split["sha256"].dropna().astype(str))
    return ids, hashes


def prepare(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_root = Path(config["data"]["root"])
    rows = pd.read_csv(config["data"]["eligible_manifest"])
    target_classes = [str(value) for value in config["selection"]["target_classes"]]
    rows = rows[
        rows["label"].isin(target_classes)
        & (rows["split"].astype(str) == "train")
        & (rows["is_synthetic"].astype(int) == 0)
        & (rows["generator_mask_eligible"].astype(int) == 1)
    ].copy()
    forbidden_ids, forbidden_hashes = forbidden_ids_and_hashes(data_root, config)
    overlap_ids = set(rows["image_id"].astype(str)) & forbidden_ids
    overlap_hashes = set(rows["sha256"].dropna().astype(str)) & forbidden_hashes
    if overlap_ids or overlap_hashes:
        raise RuntimeError(
            f"Evaluation leakage detected: ids={len(overlap_ids)}, hashes={len(overlap_hashes)}"
        )
    selected = select_morphology_anchors(
        rows,
        target_classes,
        int(config["selection"]["anchors_per_class"]),
        str(config["selection"]["morphology_field"]),
    )
    selected["source_sha256_verified"] = [
        sha256(data_root / path) for path in selected["image_path"].astype(str)
    ]
    selected["mask_sha256"] = [
        sha256(data_root / path) for path in selected["pseudo_mask_path"].astype(str)
    ]
    if not (
        selected["source_sha256_verified"].astype(str)
        == selected["sha256"].astype(str)
    ).all():
        raise RuntimeError("Source image hash mismatch")
    summary = {
        "protocol": config["protocol"],
        "test_evaluated": False,
        "selection_strategy": config["selection"]["strategy"],
        "anchors": len(selected),
        "unique_groups": int(selected["group_id"].nunique()),
        "anchors_per_class": selected["label"].value_counts().sort_index().to_dict(),
        "planned_per_arm": len(selected)
        * int(config["selection"]["candidates_per_anchor"]),
        "evaluation_overlap_ids": 0,
        "evaluation_overlap_hashes": 0,
    }
    return selected, summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    anchors, summary = prepare(config)
    output_root = Path(config["data"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(output_root / "anchors.csv", index=False)
    (output_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

