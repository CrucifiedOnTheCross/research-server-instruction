from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leak-safe Stage 16G-P1 LoRA data.")
    parser.add_argument("--config", default="configs/stage16g_p1_domain_lora.yaml")
    return parser.parse_args()


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def prepare_domain_rows(
    train: pd.DataFrame,
    anchors: pd.DataFrame,
    target_classes: list[str],
    prompts: dict[str, str],
    maximum_groups_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_ids = set(anchors["image_id"].astype(str))
    anchor_groups = set(anchors["group_id"].astype(str))
    pool = train[
        train["label"].isin(target_classes)
        & (train["is_synthetic"].astype(int) == 0)
        & (train["split"].astype(str) == "train")
    ].copy()
    pool = pool[
        ~pool["image_id"].astype(str).isin(anchor_ids)
        & ~pool["group_id"].astype(str).isin(anchor_groups)
    ].copy()
    pool["_order"] = [
        stable_hash(str(group), seed) for group in pool["group_id"].astype(str)
    ]
    pool = pool.sort_values(["label", "_order", "image_id"])
    pool = pool.drop_duplicates("group_id", keep="first")
    selected = (
        pool.groupby("label", group_keys=False)
        .head(maximum_groups_per_class)
        .copy()
    )
    selected["caption"] = selected["label"].map(prompts)
    if selected["caption"].isna().any():
        raise RuntimeError("A selected class does not have a prompt")
    selected = selected.drop(columns=["_order"]).sort_values(["label", "group_id"])
    if set(selected["group_id"].astype(str)) & anchor_groups:
        raise RuntimeError("Anchor group leakage into LoRA training data")
    summary = {
        "test_evaluated": False,
        "rows": len(selected),
        "unique_groups": int(selected["group_id"].nunique()),
        "class_counts": selected["label"].value_counts().sort_index().to_dict(),
        "excluded_anchor_ids": len(anchor_ids),
        "excluded_anchor_groups": len(anchor_groups),
        "anchor_group_overlap": 0,
    }
    return selected, summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(config["data"]["root"])
    train = pd.read_csv(data_root / config["data"]["train_split"])
    anchors = pd.read_csv(config["data"]["excluded_anchor_manifest"])
    selected, summary = prepare_domain_rows(
        train,
        anchors,
        [str(value) for value in config["data"]["target_classes"]],
        {str(key): str(value) for key, value in config["prompts"].items()},
        int(config["data"]["maximum_groups_per_class"]),
        int(config["training"]["seed"]),
    )
    output_root = Path(config["data"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(config["data"]["training_manifest"], index=False)
    (output_root / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

