#!/usr/bin/env python3
"""Idempotently attach Stage 3B locked-test metrics without uploading checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


EXPERIMENT = "ISIC2019-LT Stage3B Split Confirmation"


def scalar_metrics(payload: dict, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{key}": float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def exactly_one(client, experiment_id: str, filter_string: str):
    runs = client.search_runs([experiment_id], filter_string=filter_string, max_results=2)
    if len(runs) != 1:
        raise RuntimeError(f"Expected one MLflow run for {filter_string!r}, found {len(runs)}")
    return runs[0]


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    project = Path(__file__).resolve().parents[1]
    locked_path = project / "outputs/stage3b_locked_test/summary.json"
    analysis_path = project / "outputs/stage3b_analysis/stage3b_analysis.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not locked.get("one_shot") or not locked.get("test_evaluated"):
        raise RuntimeError("Stage 3B locked result is incomplete")
    mlflow.set_tracking_uri("http://10.200.1.180:5000")
    experiment = mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    touched = set()
    for record in locked["records"]:
        source = Path(record["source_run"])
        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        signature = source_summary["run_signature"]
        variant = record["variant"]
        if variant == "raw":
            run = exactly_one(client, experiment.experiment_id, f"tags.run_signature = '{signature}'")
        else:
            derivation = "logit_adjustment" if variant == "logit_adjustment_tau1" else "temperature_scaling"
            run = exactly_one(
                client, experiment.experiment_id,
                f"tags.source_run_signature = '{signature}' and tags.derivation = '{derivation}'",
            )
        prefix = f"locked_test/{record['checkpoint']}"
        metrics = {
            **scalar_metrics(record["image"], f"{prefix}/image"),
            **scalar_metrics(record["lesion"], f"{prefix}/lesion"),
        }
        for key, value in metrics.items():
            client.log_metric(run.info.run_id, key, value)
        client.set_tag(run.info.run_id, "locked_test_available", "true")
        client.set_tag(run.info.run_id, "locked_test_one_shot", "true")
        touched.add(run.info.run_id)

    analysis_id = "stage3b_locked_split_model_lesion_v1"
    existing = client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.analysis_id = '{analysis_id}'", max_results=2
    )
    if len(existing) > 1:
        raise RuntimeError("Duplicate Stage 3B analysis runs in MLflow")
    if existing:
        analysis_run_id = existing[0].info.run_id
    else:
        analysis_run_id = client.create_run(
            experiment.experiment_id,
            tags={
                "mlflow.runName": "stage3b_locked_hierarchical_analysis",
                "project": "longtail-medical", "stage": "stage3b_analysis",
                "analysis_id": analysis_id, "test_evaluated": "true",
                "git_commit_training": locked["git_commit"],
            },
        ).info.run_id
    for metric, values in analysis["hierarchical_bootstrap"].items():
        for key in ("mean", "ci95_low", "ci95_high", "probability_positive"):
            client.log_metric(analysis_run_id, f"primary/{metric}/{key}", float(values[key]))
    comparison = analysis["paired_comparisons_last_lesion"]["ldam_drw_minus_ce"]["overall"]
    for metric, values in comparison.items():
        client.log_metric(analysis_run_id, f"paired/{metric}/mean", float(values["mean"]))
        client.log_metric(analysis_run_id, f"paired/{metric}/positive_pairs", float(values["positive_pairs"]))
    client.log_artifact(analysis_run_id, str(analysis_path))
    client.log_artifact(
        analysis_run_id,
        str(project / "docs/stage3b_split_generalization_protocol_2026-08-02.md"),
    )
    client.set_terminated(analysis_run_id, status="FINISHED")
    print(json.dumps({
        "status": "completed", "updated_runs": len(touched),
        "analysis_run_id": analysis_run_id, "checkpoints_uploaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
