from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    required = {"experiment", "data", "model", "training", "checkpoint", "evaluation"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if payload["evaluation"].get("run_test", False):
        raise ValueError("Stage 1 test evaluation is locked")
    if payload["checkpoint"].get("primary_policy") != "last":
        raise ValueError("Confirmatory training requires last checkpoint as primary")
    if payload["checkpoint"].get("secondary_policy") != "best_validation":
        raise ValueError("Secondary checkpoint policy must be best_validation")
    if payload["checkpoint"].get("monitor") != payload["training"].get("monitor"):
        raise ValueError("Checkpoint and training monitor must match")
    return payload


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
