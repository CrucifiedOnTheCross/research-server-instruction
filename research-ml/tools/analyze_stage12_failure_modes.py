from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.datasets import build_transforms
from src.models import create_model, extract_features


TARGET_CLASSES = ("mel", "akiec", "bkl")
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 12 synthetic failure-mode diagnostics.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--real-csv", default="splits/stage8/train_real.csv")
    parser.add_argument("--synthetic-csv", default="splits/stage10/selected_synthetic_strict_id.csv")
    parser.add_argument("--out-dir", default="outputs/reports/stage12_failure_mode_diagnostics")
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prdc-k", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--skip-convnext", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_id(row: dict[str, Any]) -> str:
    return Path(str(row.get("source_image_id") or row.get("source_image_path") or "")).stem


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id") or Path(str(row["image_path"])).stem)


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path, transform: Any) -> None:
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = resolve(self.root, str(row["image_path"]))
        image = self.transform(Image.open(path).convert("RGB"))
        return {"image": image, "index": index}


@torch.inference_mode()
def collect_embeddings(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> np.ndarray:
    loader = DataLoader(
        RowDataset(rows, root, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True)
        features = extract_features(model, images)
        chunks.append(features.detach().float().cpu().numpy())
    result = np.concatenate(chunks)
    return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)


def cache_key(rows: list[dict[str, Any]], encoder: str) -> str:
    digest = hashlib.sha256()
    digest.update(encoder.encode())
    for row in rows:
        digest.update(str(row["image_path"]).encode())
        digest.update(str(row["label"]).encode())
    return digest.hexdigest()


def load_or_extract(
    out_dir: Path,
    cache_name: str,
    encoder_description: str,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    root: Path,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    npz_path = out_dir / f"embeddings_{cache_name}.npz"
    index_path = out_dir / f"embeddings_{cache_name}_index.csv"
    expected_key = cache_key(rows, encoder_description)
    if npz_path.exists() and index_path.exists():
        cache = np.load(npz_path)
        if str(cache["cache_key"].item()) == expected_key:
            return cache["features"]
    features = collect_embeddings(
        model, rows, root, transform, device, batch_size, num_workers, cache_name
    )
    np.savez_compressed(npz_path, features=features, cache_key=expected_key)
    pd.DataFrame(
        {
            "row_index": np.arange(len(rows)),
            "image_id": [row_id(row) for row in rows],
            "image_path": [row["image_path"] for row in rows],
            "label": [row["label"] for row in rows],
            "kind": [row["_kind"] for row in rows],
        }
    ).to_csv(index_path, index=False)
    return features


def kth_radius(x: np.ndarray, k: int) -> np.ndarray:
    neighbors = min(k + 1, len(x))
    if neighbors <= 1:
        return np.zeros(len(x))
    distances = NearestNeighbors(n_neighbors=neighbors, metric="cosine").fit(x).kneighbors(x)[0]
    return distances[:, -1]


def prdc(real: np.ndarray, generated: np.ndarray, k: int) -> dict[str, float]:
    if len(real) <= k or len(generated) <= k:
        raise ValueError(f"PRDC needs more than k={k} samples in each set")
    real_radii = kth_radius(real, k)
    generated_radii = kth_radius(generated, k)
    distances = 1.0 - generated @ real.T
    precision = float((distances <= real_radii[None, :]).any(axis=1).mean())
    recall = float((distances <= generated_radii[:, None]).any(axis=0).mean())
    density = float((distances <= real_radii[None, :]).sum(axis=1).mean() / k)
    coverage = float((distances.min(axis=0) <= real_radii).mean())
    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


def vendi_score(x: np.ndarray) -> float:
    if len(x) == 0:
        return math.nan
    kernel = np.clip(x @ x.T, -1.0, 1.0)
    eigenvalues = np.linalg.eigvalsh(kernel / len(x))
    probabilities = np.clip(eigenvalues, 0.0, None)
    probabilities /= probabilities.sum().clip(min=1e-12)
    nonzero = probabilities > 1e-12
    return float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    probabilities = singular**2
    probabilities /= probabilities.sum().clip(min=1e-12)
    nonzero = probabilities > 1e-12
    return float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))


def feature_diagnostics(
    encoder: str,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    novelty: list[dict[str, Any]] = []
    kinds = np.asarray([row["_kind"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    source_ids = {source_id(row) for row in rows if row["_kind"] == "synthetic"}

    for label in TARGET_CLASSES:
        label_mask = labels == label
        reference_indices = np.flatnonzero(
            label_mask
            & (kinds == "real")
            & np.asarray([row_id(row) not in source_ids for row in rows])
        )
        reference = features[reference_indices]
        for arm in ("synthetic", "source"):
            arm_indices = np.flatnonzero(label_mask & (kinds == arm))
            arm_features = features[arm_indices]
            values = prdc(reference, arm_features, k)
            nearest = (1.0 - arm_features @ reference.T).min(axis=1)
            metrics.append(
                {
                    "encoder": encoder,
                    "label": label,
                    "arm": arm,
                    "n_reference_non_source": len(reference),
                    "n_arm": len(arm_features),
                    **{f"prdc_{name}": value for name, value in values.items()},
                    "vendi_score": vendi_score(arm_features),
                    "effective_rank": effective_rank(arm_features),
                    "nearest_reference_distance_mean": float(nearest.mean()),
                }
            )

        source_lookup = {
            row_id(rows[index]): features[index]
            for index in np.flatnonzero(label_mask & (kinds == "source"))
        }
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(reference)
        for index in np.flatnonzero(label_mask & (kinds == "synthetic")):
            row = rows[index]
            own = source_lookup[source_id(row)]
            own_distance = float(1.0 - np.dot(features[index], own))
            non_source_distance = float(nn.kneighbors(features[index][None, :])[0][0, 0])
            novelty.append(
                {
                    "encoder": encoder,
                    "label": label,
                    "synthetic_image_id": row_id(row),
                    "source_image_id": source_id(row),
                    "source_group_id": row.get("source_group_id", ""),
                    "own_source_distance": own_distance,
                    "nearest_non_source_distance": non_source_distance,
                    "novelty_ratio": own_distance / max(non_source_distance, 1e-12),
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(novelty)


def center_crop_resize(image: Image.Image, size: int = 384) -> np.ndarray:
    image = image.convert("L")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def frequency_features(image: np.ndarray) -> dict[str, float]:
    centered = image - float(image.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    height, width = spectrum.shape
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2)
    radius /= min(height, width) / 2
    valid = radius <= 1.0
    total = float(spectrum[valid].sum()) + 1e-12
    low = float(spectrum[valid & (radius < 0.20)].sum() / total)
    middle = float(spectrum[valid & (radius >= 0.20) & (radius < 0.50)].sum() / total)
    high = float(spectrum[valid & (radius >= 0.50)].sum() / total)

    bins = np.linspace(0.05, 1.0, 40)
    centers = (bins[:-1] + bins[1:]) / 2
    radial_power = np.asarray(
        [spectrum[(radius >= lo) & (radius < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    )
    slope = float(np.polyfit(np.log(centers), np.log(radial_power + 1e-12), 1)[0])
    gradient_y, gradient_x = np.gradient(image)
    gradient_rms = float(np.sqrt(np.mean(gradient_x**2 + gradient_y**2)))
    return {
        "fft_low_fraction": low,
        "fft_mid_fraction": middle,
        "fft_high_fraction": high,
        "spectral_slope": slope,
        "gradient_rms": gradient_rms,
    }


def collect_frequency(rows: list[dict[str, Any]], root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="frequency features"):
        path = resolve(root, str(row["image_path"]))
        values = frequency_features(center_crop_resize(Image.open(path)))
        records.append(
            {
                "image_id": row_id(row),
                "image_path": row["image_path"],
                "label": row["label"],
                "kind": row["_kind"],
                "source_image_id": source_id(row) if row["_kind"] == "synthetic" else "",
                "source_group_id": row.get("source_group_id") or row.get("group_id", ""),
                "pair_id": row.get("_pair_id", ""),
                **values,
            }
        )
    return pd.DataFrame(records)


def grouped_bootstrap_mean(
    values: np.ndarray, groups: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.mean(
            np.concatenate([values[groups == group] for group in sampled])
        )
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def frequency_bootstrap(
    frame: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    synthetic = frame[frame["kind"] == "synthetic"].copy()
    source = frame[frame["kind"] == "source"].copy()
    source = source.set_index("pair_id")
    records: list[dict[str, Any]] = []
    metrics = ("fft_low_fraction", "fft_mid_fraction", "fft_high_fraction", "spectral_slope", "gradient_rms")
    for label in TARGET_CLASSES:
        subset = synthetic[synthetic["label"] == label]
        source_rows = source.loc[subset["pair_id"]]
        groups = subset["source_group_id"].astype(str).to_numpy()
        for metric in metrics:
            delta = subset[metric].to_numpy(float) - source_rows[metric].to_numpy(float)
            mean, low, high = grouped_bootstrap_mean(delta, groups, replicates, seed)
            records.append(
                {
                    "label": label,
                    "metric": metric,
                    "mean_synthetic_minus_source": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "source_groups": len(np.unique(groups)),
                }
            )
    return pd.DataFrame(records)


def infer_seed(path: Path) -> int:
    match = re.search(r"_(42|43|44)$", path.parent.name)
    if not match:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group(1))


def ranking_tail_diagnostics(
    replay: pd.DataFrame, synthetic: pd.DataFrame, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []
    confusers: list[dict[str, Any]] = []
    for label in TARGET_CLASSES:
        positive = replay["target"].to_numpy() == label
        n_positive = int(positive.sum())
        if n_positive == 0:
            continue
        replay_scores = replay[f"prob_{label}"].to_numpy(float)
        synthetic_scores = synthetic[f"prob_{label}"].to_numpy(float)

        def top_k_stats(scores: np.ndarray) -> tuple[float, float]:
            top_indices = np.argsort(scores)[-n_positive:]
            true_positives = int(positive[top_indices].sum())
            value = true_positives / n_positive
            return value, value

        replay_precision, replay_recall = top_k_stats(replay_scores)
        synthetic_precision, synthetic_recall = top_k_stats(synthetic_scores)
        replay_ap = float(average_precision_score(positive, replay_scores))
        synthetic_ap = float(average_precision_score(positive, synthetic_scores))
        ranking.append(
            {
                "seed": seed,
                "label": label,
                "positive_count": n_positive,
                "auprc_replay": replay_ap,
                "auprc_synthetic": synthetic_ap,
                "auprc_delta": synthetic_ap - replay_ap,
                "top_k_precision_delta": synthetic_precision - replay_precision,
                "top_k_recall_delta": synthetic_recall - replay_recall,
                "positive_q10_delta": float(
                    np.quantile(synthetic_scores[positive], 0.10)
                    - np.quantile(replay_scores[positive], 0.10)
                ),
                "positive_q50_delta": float(
                    np.quantile(synthetic_scores[positive], 0.50)
                    - np.quantile(replay_scores[positive], 0.50)
                ),
                "negative_q90_delta": float(
                    np.quantile(synthetic_scores[~positive], 0.90)
                    - np.quantile(replay_scores[~positive], 0.90)
                ),
                "negative_q95_delta": float(
                    np.quantile(synthetic_scores[~positive], 0.95)
                    - np.quantile(replay_scores[~positive], 0.95)
                ),
                "negative_q99_delta": float(
                    np.quantile(synthetic_scores[~positive], 0.99)
                    - np.quantile(replay_scores[~positive], 0.99)
                ),
            }
        )
        for true_class in sorted(set(replay["target"]) - {label}):
            mask = replay["target"].to_numpy() == true_class
            confusers.append(
                {
                    "seed": seed,
                    "scored_class": label,
                    "true_negative_class": true_class,
                    "count": int(mask.sum()),
                    "mean_probability_delta": float(
                        (synthetic_scores[mask] - replay_scores[mask]).mean()
                    ),
                    "q95_probability_delta": float(
                        np.quantile(synthetic_scores[mask], 0.95)
                        - np.quantile(replay_scores[mask], 0.95)
                    ),
                }
            )
    return ranking, confusers


def prediction_diagnostics(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    confuser_records: list[dict[str, Any]] = []
    for replay_path in sorted(project_root.glob("outputs/stage11b_replay_strict_id_convnext_small_384/*/val_predictions_best.csv")):
        seed = infer_seed(replay_path)
        synthetic_paths = list(
            project_root.glob(f"outputs/stage11b_synthetic_strict_id_convnext_small_384/*_{seed}/val_predictions_best.csv")
        )
        if len(synthetic_paths) != 1:
            raise ValueError(f"Expected one synthetic prediction file for seed {seed}")
        replay = pd.read_csv(replay_path).sort_values("path").reset_index(drop=True)
        synthetic = pd.read_csv(synthetic_paths[0]).sort_values("path").reset_index(drop=True)
        if not replay[["path", "target"]].equals(synthetic[["path", "target"]]):
            raise ValueError(f"Prediction alignment failed for seed {seed}")
        ranking, confusers = ranking_tail_diagnostics(replay, synthetic, seed)
        ranking_records.extend(ranking)
        confuser_records.extend(confusers)

        correct_replay = replay["prediction"] == replay["target"]
        correct_synthetic = synthetic["prediction"] == synthetic["target"]
        transitions.extend(
            [
                {"seed": seed, "transition": "gain", "count": int((~correct_replay & correct_synthetic).sum())},
                {"seed": seed, "transition": "loss", "count": int((correct_replay & ~correct_synthetic).sum())},
                {"seed": seed, "transition": "unchanged_correct", "count": int((correct_replay & correct_synthetic).sum())},
                {"seed": seed, "transition": "unchanged_wrong", "count": int((~correct_replay & ~correct_synthetic).sum())},
            ]
        )
        for target_class in replay["target"].unique():
            for predicted_class in replay["prediction"].unique():
                mask = (replay["prediction"] == target_class) & (synthetic["prediction"] == predicted_class)
                if mask.any():
                    transitions.append(
                        {
                            "seed": seed,
                            "transition": f"argmax_{target_class}_to_{predicted_class}",
                            "count": int(mask.sum()),
                        }
                    )

        for label in TARGET_CLASSES:
            delta = synthetic[f"prob_{label}"].to_numpy(float) - replay[f"prob_{label}"].to_numpy(float)
            positive = replay["target"].to_numpy() == label
            pos_delta = float(delta[positive].mean())
            neg_delta = float(delta[~positive].mean())
            records.append(
                {
                    "seed": seed,
                    "label": label,
                    "positive_probability_delta": pos_delta,
                    "negative_probability_delta": neg_delta,
                    "separation_delta": pos_delta - neg_delta,
                    "positive_count": int(positive.sum()),
                    "negative_count": int((~positive).sum()),
                }
            )
    return (
        pd.DataFrame(records),
        pd.DataFrame(transitions),
        pd.DataFrame(ranking_records),
        pd.DataFrame(confuser_records),
    )


def plot_distribution_metrics(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame["label"] == "mel"]
    encoders = list(dict.fromkeys(subset["encoder"]))
    figure, axes = plt.subplots(len(encoders), 1, figsize=(9, 3.2 * len(encoders)), squeeze=False)
    for axis, encoder in zip(axes[:, 0], encoders):
        data = subset[subset["encoder"] == encoder].set_index("arm")
        x = np.arange(4)
        width = 0.36
        metrics = ["prdc_precision", "prdc_recall", "prdc_density", "prdc_coverage"]
        axis.bar(x - width / 2, data.loc["source", metrics], width, label="source replay")
        axis.bar(x + width / 2, data.loc["synthetic", metrics], width, label="synthetic")
        axis.set_xticks(x, [item.replace("prdc_", "") for item in metrics])
        axis.set_title(f"Melanoma distribution diagnostics: {encoder}")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_probability_shifts(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = list(TARGET_CLASSES)
    positions = np.arange(len(labels))
    means = frame.groupby("label")["separation_delta"].mean().reindex(labels)
    errors = frame.groupby("label")["separation_delta"].std().reindex(labels).fillna(0)
    axis.bar(positions, means, yerr=errors, capsize=4, color=["#a33d3d", "#4776a8", "#6b8e4e"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Synthetic minus replay separation")
    axis.set_title("Stage 11B class probability separation shift")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = resolve(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_rows_all = [row for row in read_rows(resolve(data_root, args.real_csv)) if row["label"] in TARGET_CLASSES]
    synthetic_rows = read_rows(resolve(data_root, args.synthetic_csv))
    real_by_id = {row_id(row): row for row in real_rows_all}
    source_rows: list[dict[str, Any]] = []
    for row in synthetic_rows:
        key = source_id(row)
        if key not in real_by_id:
            raise ValueError(f"Synthetic source {key} is absent from real train")
        source_rows.append({**real_by_id[key], "_pair_id": row_id(row)})
        row["_pair_id"] = row_id(row)

    rows: list[dict[str, Any]] = []
    for row in real_rows_all:
        rows.append({**row, "_kind": "real"})
    for row in source_rows:
        rows.append({**row, "_kind": "source"})
    for row in synthetic_rows:
        rows.append({**row, "_kind": "synthetic"})

    feature_frames: list[pd.DataFrame] = []
    novelty_frames: list[pd.DataFrame] = []
    if not args.skip_dino:
        model = timm.create_model(args.dino_model, pretrained=True, num_classes=0).to(device)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        features = load_or_extract(
            out_dir, "dino", f"timm:{args.dino_model}", model, rows, data_root,
            transform, device, args.batch_size, args.num_workers,
        )
        metrics, novelty = feature_diagnostics("dino", rows, features, args.prdc_k)
        feature_frames.append(metrics)
        novelty_frames.append(novelty)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_convnext:
        for seed in SEEDS:
            candidates = [
                path for path in project_root.glob(
                    f"outputs/stage11_real_convnext_small_regularized_384/*_{seed}"
                )
                if "invalid" not in path.name and (path / "best.pt").exists()
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one qualified ConvNeXt-S run for seed {seed}, found {candidates}")
            run_dir = candidates[0]
            config = load_config(run_dir / "config.resolved.yaml")
            model = create_model(config, num_classes=7).to(device)
            checkpoint = torch.load(run_dir / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model"])
            transform = build_transforms(config, train=False)
            features = load_or_extract(
                out_dir, f"convnext_seed{seed}", f"{run_dir}:best.pt", model, rows,
                data_root, transform, device, args.batch_size, args.num_workers,
            )
            metrics, novelty = feature_diagnostics(
                f"convnext_real_seed{seed}", rows, features, args.prdc_k
            )
            feature_frames.append(metrics)
            novelty_frames.append(novelty)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    feature_frame = pd.concat(feature_frames, ignore_index=True)
    novelty_frame = pd.concat(novelty_frames, ignore_index=True)
    feature_frame.to_csv(out_dir / "feature_distribution_metrics.csv", index=False)
    novelty_frame.to_csv(out_dir / "source_conditional_novelty.csv", index=False)

    frequency_rows = [
        {**row, "_kind": "source"} for row in source_rows
    ] + [{**row, "_kind": "synthetic"} for row in synthetic_rows]
    frequency = collect_frequency(frequency_rows, data_root)
    frequency.to_csv(out_dir / "frequency_features.csv", index=False)
    frequency_ci = frequency_bootstrap(
        frequency, args.bootstrap_replicates, args.bootstrap_seed
    )
    frequency_ci.to_csv(out_dir / "frequency_paired_bootstrap.csv", index=False)

    probability, transitions, ranking, confusers = prediction_diagnostics(project_root)
    probability.to_csv(out_dir / "prediction_probability_shifts.csv", index=False)
    transitions.to_csv(out_dir / "argmax_transition_counts.csv", index=False)
    ranking.to_csv(out_dir / "ranking_tail_diagnostics.csv", index=False)
    confusers.to_csv(out_dir / "confuser_probability_shifts.csv", index=False)
    plot_distribution_metrics(feature_frame, out_dir / "melanoma_prdc_by_encoder.png")
    plot_probability_shifts(probability, out_dir / "probability_separation_shift.png")

    summary = {
        "status": "complete",
        "stage": "stage12",
        "locked_test_evaluated": False,
        "synthetic_rows": len(synthetic_rows),
        "replay_presentations": len(source_rows),
        "unique_sources": len({row_id(row) for row in source_rows}),
        "target_classes": list(TARGET_CLASSES),
        "encoders": feature_frame["encoder"].unique().tolist(),
        "prdc_k": args.prdc_k,
        "bootstrap_replicates": args.bootstrap_replicates,
        "reference_excludes_all_synthetic_sources": True,
        "prediction_seeds": sorted(probability["seed"].unique().tolist()),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
