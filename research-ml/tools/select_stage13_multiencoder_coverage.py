from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from timm.data import create_transform, resolve_model_data_config

from src.config import load_config
from src.datasets import build_transforms
from src.models import create_model
from tools.analyze_stage12_failure_modes import (
    TARGET_CLASSES,
    collect_frequency,
    kth_radius,
    load_or_extract,
    prdc,
    resolve,
    row_id,
    source_id,
    vendi_score,
)
from tools.make_source_matched_replay import build_source_replay


SEEDS = (42, 43, 44)
ENCODER_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 13 multi-encoder coverage selection.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--real-csv", default="splits/stage8/train_real.csv")
    parser.add_argument(
        "--pool-csv",
        default="outputs/reports/stage8_dino_geometry/synthetic_dino_scores.csv",
    )
    parser.add_argument("--strict-csv", default="splits/stage10/selected_synthetic_strict_id.csv")
    parser.add_argument("--out-dir", default="outputs/reports/stage13_selection")
    parser.add_argument("--split-out-dir", default="splits/stage13")
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--dose-per-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prdc-k", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    return median, np.maximum(1.4826 * mad, 1e-6)


def score_encoder(
    encoder: str,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    pool: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = np.asarray([str(row["label"]) for row in rows])
    kinds = np.asarray([str(row["_kind"]) for row in rows])
    ids = np.asarray([row_id(row) for row in rows])
    real_mask = kinds == "real"
    synth_indices = np.flatnonzero(kinds == "synthetic")
    synth_features = features[synth_indices]
    synth_rows = [rows[index] for index in synth_indices]
    if [row_id(row) for row in synth_rows] != pool["image_id"].astype(str).tolist():
        raise ValueError(f"{encoder}: synthetic feature order differs from pool")
    real_id_to_index = {
        ids[index]: index for index in np.flatnonzero(real_mask)
    }
    records: list[dict[str, Any]] = []
    distances_by_class: dict[str, np.ndarray] = {}
    radii_by_class: dict[str, np.ndarray] = {}

    for label in TARGET_CLASSES:
        class_real_indices = np.flatnonzero(real_mask & (labels == label))
        class_synth_positions = np.flatnonzero(
            pool["label"].astype(str).to_numpy() == label
        )
        real_x = features[class_real_indices]
        synth_x = synth_features[class_synth_positions]
        radii = kth_radius(real_x, k)
        distances = 1.0 - synth_x @ real_x.T
        distances_by_class[label] = distances
        radii_by_class[label] = radii
        other_x = features[np.flatnonzero(real_mask & (labels != label))]
        other_distance = (1.0 - synth_x @ other_x.T).min(axis=1)

        for local_position, pool_position in enumerate(class_synth_positions):
            row = synth_rows[pool_position]
            source_key = source_id(row)
            if source_key not in real_id_to_index:
                raise ValueError(f"{encoder}: missing source {source_key}")
            source_global_index = real_id_to_index[source_key]
            source_class_position = int(
                np.flatnonzero(class_real_indices == source_global_index)[0]
            )
            distance_row = distances[local_position].copy()
            distance_row[source_class_position] = np.inf
            nearest_non_source = float(distance_row.min())
            own_source_distance = float(
                1.0 - np.dot(synth_x[local_position], features[source_global_index])
            )
            source_radius = float(radii[source_class_position])
            nearest_same = float(distances[local_position].min())
            records.append(
                {
                    "image_id": row_id(row),
                    f"{encoder}_inside": int(
                        (distances[local_position] <= radii).any()
                    ),
                    f"{encoder}_nearest_same": nearest_same,
                    f"{encoder}_nearest_non_source": nearest_non_source,
                    f"{encoder}_nearest_other": float(other_distance[local_position]),
                    f"{encoder}_class_margin": float(
                        other_distance[local_position] - nearest_same
                    ),
                    f"{encoder}_own_source_distance": own_source_distance,
                    f"{encoder}_source_radius": source_radius,
                    f"{encoder}_local_novelty_ratio": own_source_distance
                    / max(source_radius, 1e-12),
                }
            )
    return pd.DataFrame(records), distances_by_class, radii_by_class


def add_frequency_scores(
    pool: pd.DataFrame,
    real: pd.DataFrame,
    rows: list[dict[str, Any]],
    data_root: Path,
    out_dir: Path,
) -> pd.DataFrame:
    cache = out_dir / "frequency_all_candidates.csv"
    if cache.exists():
        frequency = pd.read_csv(cache)
    else:
        frequency = collect_frequency(rows, data_root)
        frequency.to_csv(cache, index=False)
    metrics = [
        "fft_low_fraction",
        "fft_mid_fraction",
        "fft_high_fraction",
        "spectral_slope",
        "gradient_rms",
    ]
    real_frequency = frequency[frequency["kind"] == "real"].set_index("image_id")
    synth_frequency = frequency[frequency["kind"] == "synthetic"].set_index("image_id")
    records: list[dict[str, Any]] = []
    for label in TARGET_CLASSES:
        real_ids = real[real["label"].astype(str) == label]["image_id"].astype(str)
        real_values = real_frequency.loc[real_ids, metrics].to_numpy(float)
        median, scale = robust_scale(real_values)
        class_pool = pool[pool["label"].astype(str) == label]
        source_distances: list[float] = []
        staged: list[dict[str, Any]] = []
        for _, row in class_pool.iterrows():
            values = synth_frequency.loc[str(row["image_id"]), metrics].to_numpy(float)
            source_values = real_frequency.loc[str(row["source_image_id"]), metrics].to_numpy(float)
            z = np.abs((values - median) / scale)
            source_distance = float(np.sqrt(np.mean(((values - source_values) / scale) ** 2)))
            source_distances.append(source_distance)
            staged.append(
                {
                    "image_id": str(row["image_id"]),
                    "frequency_max_abs_z": float(z.max()),
                    "frequency_source_distance": source_distance,
                }
            )
        q50 = float(np.quantile(source_distances, 0.50))
        q75 = float(np.quantile(source_distances, 0.75))
        for record in staged:
            record["frequency_source_q50"] = q50
            record["frequency_source_q75"] = q75
            records.append(record)
    return pd.DataFrame(records)


def greedy_facility_select(
    candidates: pd.DataFrame,
    similarity: np.ndarray,
    weights: np.ndarray,
    dose: int,
) -> list[int]:
    if len(candidates) < dose:
        raise ValueError(f"Only {len(candidates)} candidates for dose {dose}")
    current = np.zeros(similarity.shape[1], dtype=np.float32)
    selected: list[int] = []
    used_sources: set[str] = set()
    available = list(range(len(candidates)))
    quality = candidates["stage13_quality_score"].to_numpy(float)
    image_ids = candidates["image_id"].astype(str).to_numpy()
    sources = candidates["source_image_id"].astype(str).to_numpy()
    for _ in range(dose):
        best: tuple[float, float, str, int] | None = None
        for position in available:
            if sources[position] in used_sources:
                continue
            gain = float(
                np.sum(weights * np.maximum(similarity[position] - current, 0.0))
            )
            key = (gain, quality[position], image_ids[position], position)
            if best is None or key[:2] > best[:2] or (
                key[:2] == best[:2] and key[2] < best[2]
            ):
                best = key
        if best is None:
            raise ValueError("Source uniqueness prevents completing facility selection")
        position = best[3]
        selected.append(position)
        used_sources.add(sources[position])
        current = np.maximum(current, similarity[position])
        available.remove(position)
    return selected


def selection_gate(
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    frequency_wins: int,
) -> dict[str, Any]:
    coverage_wins = int((comparison["coverage_delta_new_minus_strict"] > 0).sum())
    mean_coverage = float(comparison["coverage_delta_new_minus_strict"].mean())
    mean_precision = float(comparison["precision_delta_new_minus_strict"].mean())
    mean_density = float(comparison["density_delta_new_minus_strict"].mean())
    tier_b_fraction = float((selected["stage13_tier"] == "B").mean())
    checks = {
        "balanced_dose": selected["label"].value_counts().to_dict()
        == {label: 30 for label in TARGET_CLASSES},
        "unique_sources": int(selected["source_image_id"].nunique()) == 90,
        "coverage_wins_at_least_8_of_12": coverage_wins >= 8,
        "mean_coverage_positive": mean_coverage > 0,
        "mean_precision_not_below_margin": mean_precision >= -0.05,
        "mean_density_not_below_margin": mean_density >= -0.05,
        "frequency_wins_at_least_2_of_3": frequency_wins >= 2,
        "tier_b_fraction_at_most_half": tier_b_fraction <= 0.5,
        "locked_test_used": False,
    }
    return {
        "gate_open": all(
            value for key, value in checks.items() if key != "locked_test_used"
        )
        and checks["locked_test_used"] is False,
        "checks": checks,
        "coverage_wins": coverage_wins,
        "mean_coverage_delta": mean_coverage,
        "mean_precision_delta": mean_precision,
        "mean_density_delta": mean_density,
        "frequency_class_wins": frequency_wins,
        "tier_b_fraction": tier_b_fraction,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = resolve(project_root, args.out_dir)
    split_out_dir = resolve(data_root, args.split_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = resolve(project_root, args.pool_csv)
    if not pool_path.exists():
        pool_path = resolve(data_root, args.pool_csv)
    strict_path = resolve(data_root, args.strict_csv)
    real_path = resolve(data_root, args.real_csv)
    pool = pd.read_csv(pool_path)
    pool = pool[pool["label"].astype(str).isin(TARGET_CLASSES)].reset_index(drop=True)
    strict = pd.read_csv(strict_path)
    real = pd.read_csv(real_path)
    real = real[real["is_synthetic"].astype(int) == 0].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [
        {**row, "_kind": "real"}
        for row in real.to_dict(orient="records")
    ] + [
        {**row, "_kind": "synthetic"}
        for row in pool.to_dict(orient="records")
    ]
    feature_tables: list[pd.DataFrame] = []
    distances: dict[str, dict[str, np.ndarray]] = {}
    radii: dict[str, dict[str, np.ndarray]] = {}

    dino = timm.create_model(args.dino_model, pretrained=True, num_classes=0).to(device)
    dino_transform = create_transform(
        **resolve_model_data_config(dino), is_training=False
    )
    dino_features = load_or_extract(
        out_dir,
        "dino_all_candidates",
        f"timm:{args.dino_model}",
        dino,
        rows,
        data_root,
        dino_transform,
        device,
        args.batch_size,
        args.num_workers,
    )
    table, distances["dino"], radii["dino"] = score_encoder(
        "dino", rows, dino_features, pool, args.prdc_k
    )
    feature_tables.append(table)
    del dino
    torch.cuda.empty_cache()

    for seed in SEEDS:
        candidates = [
            path
            for path in project_root.glob(
                f"outputs/stage11_real_convnext_small_regularized_384/*_{seed}"
            )
            if "invalid" not in path.name and (path / "best.pt").exists()
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one real-only ConvNeXt-S run for seed {seed}")
        run_dir = candidates[0]
        config = load_config(run_dir / "config.resolved.yaml")
        model = create_model(config, num_classes=7).to(device)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        features = load_or_extract(
            out_dir,
            f"convnext_seed{seed}_all_candidates",
            f"{run_dir}:best.pt",
            model,
            rows,
            data_root,
            build_transforms(config, train=False),
            device,
            args.batch_size,
            args.num_workers,
        )
        name = f"convnext_seed{seed}"
        table, distances[name], radii[name] = score_encoder(
            name, rows, features, pool, args.prdc_k
        )
        feature_tables.append(table)
        del model
        torch.cuda.empty_cache()

    scores = pool.copy()
    for table in feature_tables:
        scores = scores.merge(table, on="image_id", how="left", validate="one_to_one")
    target_real_rows = [
        row for row in rows if row["_kind"] == "real" and row["label"] in TARGET_CLASSES
    ]
    frequency_rows = target_real_rows + [
        row for row in rows if row["_kind"] == "synthetic"
    ]
    frequency = add_frequency_scores(pool, real, frequency_rows, data_root, out_dir)
    scores = scores.merge(frequency, on="image_id", validate="one_to_one")

    encoder_names = ["dino", *[f"convnext_seed{seed}" for seed in SEEDS]]
    scores["inside_count"] = sum(scores[f"{name}_inside"] for name in encoder_names)
    scores["positive_margin_count"] = sum(
        scores[f"{name}_class_margin"] > 0 for name in encoder_names
    )
    scores["novelty_band_count"] = sum(
        scores[f"{name}_local_novelty_ratio"].between(0.25, 1.50)
        for name in encoder_names
    )
    scores["stage13_tier"] = ""
    tier_a = (
        (scores["inside_count"] >= 3)
        & (scores["positive_margin_count"] >= 3)
        & (scores["novelty_band_count"] >= 3)
        & (scores["frequency_max_abs_z"] <= 4)
        & (
            scores["frequency_source_distance"]
            <= scores["frequency_source_q50"]
        )
    )
    tier_b = (
        (scores["inside_count"] >= 2)
        & (scores["positive_margin_count"] >= 3)
        & (scores["novelty_band_count"] >= 2)
        & (scores["frequency_max_abs_z"] <= 5)
        & (
            scores["frequency_source_distance"]
            <= scores["frequency_source_q75"]
        )
    )
    scores.loc[tier_b, "stage13_tier"] = "B"
    scores.loc[tier_a, "stage13_tier"] = "A"
    scores["stage13_quality_score"] = (
        scores["inside_count"] / ENCODER_COUNT
        + scores["positive_margin_count"] / ENCODER_COUNT
        + scores["novelty_band_count"] / ENCODER_COUNT
        - 0.05 * scores["frequency_source_distance"]
    )

    selected_parts: list[pd.DataFrame] = []
    selection_metadata: dict[str, Any] = {}
    strict_ids = set(strict["image_id"].astype(str))
    for label in TARGET_CLASSES:
        class_scores = scores[scores["label"].astype(str) == label].copy()
        tier_a_sources = class_scores[class_scores["stage13_tier"] == "A"][
            "source_image_id"
        ].nunique()
        allowed_tiers = {"A"} if tier_a_sources >= args.dose_per_class else {"A", "B"}
        eligible = class_scores[
            class_scores["stage13_tier"].isin(allowed_tiers)
        ].reset_index()
        class_pool_positions = np.flatnonzero(
            pool["label"].astype(str).to_numpy() == label
        )
        eligible_local_positions = np.asarray(
            [
                int(np.flatnonzero(class_pool_positions == index)[0])
                for index in eligible["index"].to_numpy()
            ]
        )
        similarities: list[np.ndarray] = []
        strict_coverage_votes = np.zeros(
            distances["dino"][label].shape[1], dtype=int
        )
        strict_class_positions = [
            int(np.flatnonzero(class_pool_positions == index)[0])
            for index in class_scores[
                class_scores["image_id"].astype(str).isin(strict_ids)
            ].index
        ]
        for name in encoder_names:
            matrix = distances[name][label]
            scale = max(float(np.median(radii[name][label])), 1e-6)
            similarities.append(np.exp(-matrix[eligible_local_positions] / scale))
            if strict_class_positions:
                strict_coverage_votes += (
                    matrix[strict_class_positions]
                    <= radii[name][label][None, :]
                ).any(axis=0)
        combined_similarity = np.mean(similarities, axis=0)
        weights = 1.0 + (strict_coverage_votes < 2).astype(float)
        chosen_positions = greedy_facility_select(
            eligible, combined_similarity, weights, args.dose_per_class
        )
        chosen = eligible.iloc[chosen_positions].copy()
        chosen["stage13_rank"] = np.arange(1, len(chosen) + 1)
        selected_parts.append(chosen.drop(columns=["index"]))
        selection_metadata[label] = {
            "tier_a_rows": int((class_scores["stage13_tier"] == "A").sum()),
            "tier_a_unique_sources": int(tier_a_sources),
            "tier_b_rows": int((class_scores["stage13_tier"] == "B").sum()),
            "allowed_tiers": sorted(allowed_tiers),
            "selected_tiers": chosen["stage13_tier"].value_counts().to_dict(),
        }
    selected = pd.concat(selected_parts, ignore_index=True)

    comparison_rows: list[dict[str, Any]] = []
    union_sources = set(selected["source_image_id"].astype(str)) | set(
        strict["source_image_id"].astype(str)
    )
    real_ids = real["image_id"].astype(str).to_numpy()
    real_labels = real["label"].astype(str).to_numpy()
    synth_offset = len(real)
    for encoder_index, name in enumerate(encoder_names):
        cache = np.load(out_dir / f"embeddings_{'dino_all_candidates' if name == 'dino' else name + '_all_candidates'}.npz")
        features = cache["features"]
        for label in TARGET_CLASSES:
            reference_mask = (
                (real_labels == label)
                & ~np.isin(real_ids, list(union_sources))
            )
            reference = features[:synth_offset][reference_mask]
            pool_label_positions = np.flatnonzero(
                pool["label"].astype(str).to_numpy() == label
            )
            pool_label_ids = pool.iloc[pool_label_positions]["image_id"].astype(str).to_numpy()
            selected_local = np.flatnonzero(
                np.isin(pool_label_ids, selected[selected["label"] == label]["image_id"].astype(str))
            )
            strict_local = np.flatnonzero(
                np.isin(pool_label_ids, strict[strict["label"] == label]["image_id"].astype(str))
            )
            label_features = features[synth_offset:][pool_label_positions]
            new_metrics = prdc(reference, label_features[selected_local], args.prdc_k)
            old_metrics = prdc(reference, label_features[strict_local], args.prdc_k)
            comparison_rows.append(
                {
                    "encoder": name,
                    "label": label,
                    "new_precision": new_metrics["precision"],
                    "strict_precision": old_metrics["precision"],
                    "precision_delta_new_minus_strict": new_metrics["precision"] - old_metrics["precision"],
                    "new_density": new_metrics["density"],
                    "strict_density": old_metrics["density"],
                    "density_delta_new_minus_strict": new_metrics["density"] - old_metrics["density"],
                    "new_coverage": new_metrics["coverage"],
                    "strict_coverage": old_metrics["coverage"],
                    "coverage_delta_new_minus_strict": new_metrics["coverage"] - old_metrics["coverage"],
                    "new_vendi": vendi_score(label_features[selected_local]),
                    "strict_vendi": vendi_score(label_features[strict_local]),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    frequency_wins = 0
    for label in TARGET_CLASSES:
        new_mean = selected[selected["label"] == label]["frequency_source_distance"].mean()
        strict_mean = scores[scores["image_id"].astype(str).isin(
            strict[strict["label"] == label]["image_id"].astype(str)
        )]["frequency_source_distance"].mean()
        frequency_wins += int(new_mean < strict_mean)
    gate = selection_gate(comparison, selected, frequency_wins)

    selected["stage13_selected"] = 1
    selected.to_csv(split_out_dir / "selected_synthetic_coverage_targeted.csv", index=False)
    real_train = real.copy()
    synthetic_train = pd.concat([real_train, selected], ignore_index=True, sort=False)
    synthetic_train.to_csv(split_out_dir / "train_synthetic_coverage_targeted.csv", index=False)
    selected_for_replay = selected.copy()
    selected_for_replay["selected_by_stage6"] = 1
    replay_train, replay_rows, replay_info = build_source_replay(
        real_train, selected_for_replay, sample_weight=0.5, expected_selected=90
    )
    replay_rows.to_csv(split_out_dir / "source_replay_rows_coverage_targeted.csv", index=False)
    replay_train.to_csv(split_out_dir / "train_source_replay_coverage_targeted.csv", index=False)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    comparison.to_csv(out_dir / "selection_comparison.csv", index=False)
    manifest = {
        "protocol": "stage13_multiencoder_coverage_targeted",
        "locked_test_used": False,
        "target_classes": list(TARGET_CLASSES),
        "dose_per_class": args.dose_per_class,
        "encoders": encoder_names,
        "selection": selection_metadata,
        "gate": gate,
        "replay": replay_info,
        "inputs": {
            "real_csv": str(real_path),
            "real_csv_sha256": sha256(real_path),
            "pool_csv": str(pool_path),
            "pool_csv_sha256": sha256(pool_path),
            "strict_csv": str(strict_path),
            "strict_csv_sha256": sha256(strict_path),
        },
    }
    (out_dir / "stage13_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

