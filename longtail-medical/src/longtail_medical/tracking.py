"""Stable MLflow metadata for grouping scientific runs."""

from __future__ import annotations

from collections.abc import Mapping


def experiment_arm(name: str) -> str:
    for prefix in ("stage2_", "stage1_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def contamination_policy(name: str) -> str:
    if "exposure_matched" in name:
        return "decontaminated_exposure_matched"
    if "decontaminated_unmatched" in name:
        return "decontaminated_unmatched"
    if "lesion_disjoint" in name:
        return "lesion_disjoint"
    if "monica_original" in name:
        return "original_contaminated"
    return "not_applicable"


def build_mlflow_tags(
    config: Mapping,
    *,
    git_commit: str | None = None,
    run_signature: str | None = None,
) -> dict[str, str]:
    """Build low-cardinality grouping tags plus immutable provenance tags."""
    name = str(config["experiment"]["name"])
    configured = config.get("tracking", {}).get("tags", {})
    tags = {
        "project": "longtail-medical",
        "stage": "unspecified",
        "dataset": "ISIC2019",
        "task": "multiclass_skin_lesion_classification",
        **{str(key): str(value) for key, value in configured.items()},
        "experiment_arm": experiment_arm(name),
        "contamination_policy": contamination_policy(name),
        "protocol_version": str(config["data"]["protocol_version"]),
        "seed": str(config["experiment"]["seed"]),
        "model": str(config["model"]["name"]),
        "checkpoint_policy": str(config["checkpoint"]["primary_policy"]),
        "test_evaluated": "false",
    }
    if git_commit:
        tags["git_commit"] = git_commit
    if run_signature:
        tags["run_signature"] = run_signature
    return tags


def namespace_epoch_metrics(row: Mapping) -> dict[str, float]:
    """Add readable MLflow namespaces while preserving legacy metric keys."""
    metrics = {
        str(key): float(value) for key, value in row.items() if key != "epoch"
    }
    metrics["train/loss"] = float(row["train_loss"])
    for key, value in row.items():
        if key.startswith("val_"):
            metrics[f"val/{key.removeprefix('val_')}"] = float(value)
    if "epoch_seconds" in row:
        metrics["system/epoch_seconds"] = float(row["epoch_seconds"])
    if "gpu_max_memory_gib" in row:
        metrics["system/gpu_max_memory_gib"] = float(row["gpu_max_memory_gib"])
    return metrics
