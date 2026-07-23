from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.datasets import build_transforms, make_dataloaders
from src.models import create_model, extract_features


DEFAULT_CONFUSING = {
    "mel": ["nv", "bkl", "akiec"],
    "akiec": ["bkl", "bcc", "mel"],
    "bkl": ["mel", "nv", "akiec", "bcc"],
    "bcc": ["akiec", "bkl", "mel"],
    "df": ["akiec", "nv"],
    "vasc": ["nv", "mel"],
}


class RowImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path, class_to_idx: dict[str, int], transform: Any) -> None:
        self.rows = rows
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(self.class_to_idx[str(row["label"])], dtype=torch.long),
            "row_index": index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 feature-geometry diagnostics and geometry-filtered split builder.")
    parser.add_argument("--encoder-run-dir", default=None, help="Run directory with best.pt and config.resolved.yaml.")
    parser.add_argument("--data-config", default=None, help="Config used only to resolve data paths and class labels.")
    parser.add_argument("--encoder-model", default=None, help="Independent pretrained timm encoder without task fine-tuning.")
    parser.add_argument("--synthetic-csv", required=True, help="Synthetic pool CSV relative to data root or absolute.")
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/reports/stage6_feature_geometry")
    parser.add_argument("--split-out-dir", default="/srv/research/projects/default/ham10000/splits/stage6")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--select-classes", default="mel,akiec")
    parser.add_argument("--top-k-per-class", type=int, default=80)
    parser.add_argument("--real-k", type=int, default=5)
    parser.add_argument("--min-feature-margin", type=float, default=0.02)
    parser.add_argument("--max-real-distance-quantile", type=float, default=0.80)
    parser.add_argument("--diversity-min-distance", type=float, default=0.035)
    parser.add_argument("--train-name", default="train_stage6_geometry_mel_akiec.csv")
    parser.add_argument("--selected-name", default="selected_synthetic_geometry.csv")
    parser.add_argument("--sample-name", default="synthetic_geometry_samples.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames and not key.startswith("_"):
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@torch.no_grad()
def collect_features(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    root: Path,
    class_to_idx: dict[str, int],
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    loader = DataLoader(
        RowImageDataset(rows, root, class_to_idx, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ordered_rows: list[dict[str, Any]] = []
    model.eval()
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        features.append(feats)
        targets.append(batch["target"].numpy())
        ordered_rows.extend(rows[int(i)] for i in batch["row_index"])
    x = np.concatenate(features, axis=0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x, np.concatenate(targets), ordered_rows


def source_key(row: dict[str, Any]) -> str:
    for key in ("source_image_id", "source_image_path", "image_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return Path(value).stem
    return Path(str(row.get("image_path", ""))).stem


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def effective_rank(x: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    power = singular_values**2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    probs = power / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def kth_real_radii(x_real: np.ndarray, k: int) -> np.ndarray:
    if len(x_real) <= 1:
        return np.zeros(len(x_real), dtype=float)
    n_neighbors = min(k + 1, len(x_real))
    dists = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x_real).kneighbors(x_real, return_distance=True)[0]
    return dists[:, -1]


def class_geometry_metrics(
    label: str,
    real_x: np.ndarray,
    synth_x: np.ndarray,
    synth_rows: list[dict[str, Any]],
    k: int,
    duplicate_threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = kth_real_radii(real_x, k)
    distance_rs = 1.0 - synth_x @ real_x.T
    nearest_real_distance = distance_rs.min(axis=1)
    nearest_real_index = distance_rs.argmin(axis=1)
    inside_matrix = distance_rs <= radii[None, :]
    precision = float(inside_matrix.any(axis=1).mean()) if len(synth_x) else 0.0
    density = float(inside_matrix.sum(axis=1).mean() / max(k, 1)) if len(synth_x) else 0.0
    coverage = float((distance_rs.min(axis=0) <= radii).mean()) if len(real_x) else 0.0
    center_shift = float(1.0 - np.dot(real_x.mean(axis=0), synth_x.mean(axis=0)) / (
        np.linalg.norm(real_x.mean(axis=0)) * np.linalg.norm(synth_x.mean(axis=0)) + 1e-12
    ))

    if len(synth_x) > 1:
        synth_nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(synth_x).kneighbors(synth_x, return_distance=True)[0][:, 1]
        duplicate_rate = float((synth_nn < duplicate_threshold).mean())
        synth_nn_mean = float(np.mean(synth_nn))
    else:
        synth_nn = np.zeros(len(synth_x), dtype=float)
        duplicate_rate = 0.0
        synth_nn_mean = None

    source_counts = Counter(source_key(row) for row in synth_rows)
    max_source_share = float(max(source_counts.values()) / max(len(synth_rows), 1)) if source_counts else 0.0
    metrics = {
        "label": label,
        "n_real": int(len(real_x)),
        "n_synthetic": int(len(synth_x)),
        "prdc_precision": precision,
        "prdc_density": density,
        "prdc_coverage": coverage,
        "nearest_real_distance_mean": float(np.mean(nearest_real_distance)) if len(nearest_real_distance) else None,
        "nearest_real_distance_p50": float(np.quantile(nearest_real_distance, 0.50)) if len(nearest_real_distance) else None,
        "nearest_real_distance_p80": float(np.quantile(nearest_real_distance, 0.80)) if len(nearest_real_distance) else None,
        "nearest_real_distance_p95": float(np.quantile(nearest_real_distance, 0.95)) if len(nearest_real_distance) else None,
        "real_radius_p80": float(np.quantile(radii, 0.80)) if len(radii) else None,
        "center_shift_cosine_distance": center_shift,
        "real_effective_rank": effective_rank(real_x),
        "synthetic_effective_rank": effective_rank(synth_x),
        "synthetic_nn_distance_mean": synth_nn_mean,
        "duplicate_rate": duplicate_rate,
        "unique_source_count": int(len(source_counts)),
        "max_source_share": max_source_share,
    }
    return metrics, nearest_real_distance, nearest_real_index, inside_matrix.any(axis=1), synth_nn


def greedy_diverse(rows: list[dict[str, Any]], max_count: int, min_distance: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_features: list[np.ndarray] = []
    for row in sorted(rows, key=lambda item: float(item["geometry_score"]), reverse=True):
        feat = row["_feature"]
        if selected_features:
            nearest = min(1.0 - float(np.dot(feat, other)) for other in selected_features)
            if nearest < min_distance:
                continue
        selected.append(row)
        selected_features.append(feat)
        if len(selected) >= max_count:
            break
    return selected


def render_html(report: dict[str, Any]) -> str:
    rows = []
    for item in report["class_metrics"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item['n_real']}</td>"
            f"<td>{item['n_synthetic']}</td>"
            f"<td>{item['prdc_precision']:.4f}</td>"
            f"<td>{item['prdc_density']:.4f}</td>"
            f"<td>{item['prdc_coverage']:.4f}</td>"
            f"<td>{item['nearest_real_distance_p80']:.4f}</td>"
            f"<td>{item['duplicate_rate']:.4f}</td>"
            f"<td>{item['max_source_share']:.4f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>Stage6 Feature Geometry</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f6f8fa}}</style></head>
<body>
<h1>Stage6 Feature Geometry</h1>
<table><thead><tr><th>Class</th><th>Real</th><th>Synthetic</th><th>Precision</th><th>Density</th><th>Coverage</th><th>NN p80</th><th>Duplicate rate</th><th>Max source share</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="stage6_feature_geometry_report.json">JSON report</a> | <a href="synthetic_geometry_samples.csv">sample scores</a></p>
</body></html>"""


def main() -> None:
    args = parse_args()
    if not args.encoder_run_dir and not args.data_config:
        raise ValueError("Provide --encoder-run-dir or --data-config")
    run_dir = Path(args.encoder_run_dir) if args.encoder_run_dir else None
    config_path = run_dir / "config.resolved.yaml" if run_dir else Path(args.data_config)
    cfg = load_config(
        config_path,
        [
            f"training.batch_size={args.batch_size}",
            f"runtime.num_workers={args.num_workers}",
            "runtime.persistent_workers=false",
            "runtime.prefetch_factor=2",
        ],
    )
    data_root = Path(cfg["data"]["root"])
    synthetic_csv = resolve_path(data_root, args.synthetic_csv)
    out_dir = resolve_path(data_root, args.out_dir)
    split_out_dir = resolve_path(data_root, args.split_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = [item.strip() for item in args.target_classes.split(",") if item.strip()]
    select_classes = {item.strip() for item in args.select_classes.split(",") if item.strip()}
    target_set = set(target_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(cfg)
    if args.encoder_model:
        model = timm.create_model(args.encoder_model, pretrained=True, num_classes=0).to(device)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        encoder_description = f"timm:{args.encoder_model}"
    else:
        if run_dir is None:
            raise ValueError("--encoder-run-dir is required when --encoder-model is omitted")
        model = create_model(cfg, len(bundle.class_to_idx)).to(device)
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        transform = build_transforms(cfg, train=False)
        encoder_description = str(run_dir)

    train_csv = resolve_path(data_root, cfg["data"]["train_csv"])
    train_rows = read_rows(train_csv)
    real_rows = [
        row
        for row in train_rows
        if int(row.get(cfg["data"]["synthetic_col"], 0)) == 0
    ]
    synthetic_rows = [row for row in read_rows(synthetic_csv) if str(row.get("label")) in target_set]

    real_x, real_y, real_rows = collect_features(
        model, real_rows, data_root, bundle.class_to_idx, transform, args.batch_size, args.num_workers, device, "real features"
    )
    synth_x, synth_y, synthetic_rows = collect_features(
        model, synthetic_rows, data_root, bundle.class_to_idx, transform, args.batch_size, args.num_workers, device, "synthetic features"
    )

    real_by_class: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    synth_by_class: dict[str, tuple[np.ndarray, list[dict[str, Any]], np.ndarray]] = {}
    for label in target_classes:
        class_idx = bundle.class_to_idx[label]
        real_mask = real_y == class_idx
        synth_mask = synth_y == class_idx
        real_by_class[label] = (real_x[real_mask], [row for row, flag in zip(real_rows, real_mask) if bool(flag)])
        synth_by_class[label] = (
            synth_x[synth_mask],
            [row for row, flag in zip(synthetic_rows, synth_mask) if bool(flag)],
            np.nonzero(synth_mask)[0],
        )

    confusing_models: dict[str, NearestNeighbors] = {}
    for label in target_classes:
        confusing = DEFAULT_CONFUSING.get(label, [])
        confusing_idx = [bundle.class_to_idx[item] for item in confusing if item in bundle.class_to_idx]
        mask = np.isin(real_y, confusing_idx) if confusing_idx else real_y != bundle.class_to_idx[label]
        if mask.sum() == 0:
            mask = real_y != bundle.class_to_idx[label]
        confusing_models[label] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[mask])

    class_metrics: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for label in target_classes:
        real_cls_x, real_cls_rows = real_by_class[label]
        synth_cls_x, synth_cls_rows, _ = synth_by_class[label]
        if len(real_cls_x) == 0 or len(synth_cls_x) == 0:
            continue
        metrics, nearest_real_distance, nearest_real_index, inside_real, synth_nn = class_geometry_metrics(
            label,
            real_cls_x,
            synth_cls_x,
            synth_cls_rows,
            args.real_k,
            args.diversity_min_distance,
        )
        confusing_distance = confusing_models[label].kneighbors(synth_cls_x, return_distance=True)[0][:, 0]
        feature_margin = confusing_distance - nearest_real_distance
        max_keep_distance = float(np.quantile(nearest_real_distance, args.max_real_distance_quantile))
        margin_norm = normalize(feature_margin)
        real_dist_norm = normalize(nearest_real_distance)
        synth_nn_norm = normalize(synth_nn)
        for i, row in enumerate(synth_cls_rows):
            out = dict(row)
            out.update(
                {
                    "label": label,
                    "nearest_real_distance": float(nearest_real_distance[i]),
                    "nearest_real_image_path": real_cls_rows[int(nearest_real_index[i])].get("image_path", ""),
                    "nearest_confusing_distance": float(confusing_distance[i]),
                    "feature_margin": float(feature_margin[i]),
                    "inside_real_manifold": int(bool(inside_real[i])),
                    "nearest_synthetic_distance": float(synth_nn[i]) if len(synth_nn) else 0.0,
                    "source_key": source_key(row),
                    "geometry_score": float(
                        1.00 * int(bool(inside_real[i]))
                        + 0.75 * margin_norm[i]
                        + 0.25 * synth_nn_norm[i]
                        - 0.75 * real_dist_norm[i]
                    ),
                    "passes_geometry_filter": int(
                        bool(inside_real[i])
                        and float(nearest_real_distance[i]) <= max_keep_distance
                        and float(feature_margin[i]) >= float(args.min_feature_margin)
                    ),
                }
            )
            out["_feature"] = synth_cls_x[i]
            sample_rows.append(out)
        metrics["max_keep_nearest_real_distance"] = max_keep_distance
        metrics["mean_feature_margin"] = float(np.mean(feature_margin))
        metrics["p10_feature_margin"] = float(np.quantile(feature_margin, 0.10))
        metrics["pass_geometry_filter_count"] = int(sum(1 for row in sample_rows if row["label"] == label and int(row["passes_geometry_filter"]) == 1))
        class_metrics.append(metrics)

    selected: list[dict[str, Any]] = []
    for label in sorted(select_classes):
        candidates = [
            row
            for row in sample_rows
            if row["label"] == label and int(row.get("passes_geometry_filter", 0)) == 1
        ]
        if len(candidates) < int(args.top_k_per_class):
            candidates = [row for row in sample_rows if row["label"] == label]
        selected.extend(greedy_diverse(candidates, int(args.top_k_per_class), float(args.diversity_min_distance)))

    for row in sample_rows:
        row["selected_by_stage6"] = 0
    selected_paths = {row["image_path"] for row in selected}
    for row in sample_rows:
        if row["image_path"] in selected_paths:
            row["selected_by_stage6"] = 1

    train_fields = ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"]
    real_train_rows = [{key: row.get(key, "") for key in train_fields} for row in train_rows if int(row.get(cfg["data"]["synthetic_col"], 0)) == 0]
    train_stage6 = real_train_rows + [{key: row.get(key, "") for key in train_fields} for row in selected]

    sample_fields = [key for key in sample_rows[0] if not key.startswith("_")] if sample_rows else []
    selected_fields = sample_fields
    write_rows(out_dir / args.sample_name, sample_rows, sample_fields)
    write_rows(split_out_dir / args.selected_name, selected, selected_fields)
    write_rows(split_out_dir / args.train_name, train_stage6, train_fields)

    report = {
        "encoder": encoder_description,
        "data_config": str(config_path),
        "synthetic_csv": str(synthetic_csv),
        "target_classes": target_classes,
        "select_classes": sorted(select_classes),
        "top_k_per_class": int(args.top_k_per_class),
        "real_k": int(args.real_k),
        "min_feature_margin": float(args.min_feature_margin),
        "max_real_distance_quantile": float(args.max_real_distance_quantile),
        "selected_by_class": dict(Counter(row["label"] for row in selected)),
        "class_metrics": class_metrics,
        "sample_scores_csv": str(out_dir / args.sample_name),
        "selected_csv": str(split_out_dir / args.selected_name),
        "train_csv": str(split_out_dir / args.train_name),
    }
    (out_dir / "stage6_feature_geometry_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
