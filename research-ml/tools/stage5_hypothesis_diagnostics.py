from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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


class RowDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Run Stage 5 diagnostic checks for synthetic utility hypotheses.")
    parser.add_argument("--encoder", action="append", required=True, help="name:run_dir with best.pt and config.resolved.yaml")
    parser.add_argument("--synthetic-csv", required=True, help="Synthetic manifest/pool CSV relative to data root or absolute.")
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/reports/stage5_diagnostics")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--bkl-clusters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError(f"Expected name:path, got {value}")
    name, path = value.split(":", 1)
    return name.strip(), Path(path.strip())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        RowDataset(rows, root, class_to_idx, transform),
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


def detector_metrics(x_real: np.ndarray, x_synth: np.ndarray, seed: int) -> dict[str, Any]:
    if len(x_real) < 4 or len(x_synth) < 4:
        return {"n_real": int(len(x_real)), "n_synthetic": int(len(x_synth)), "status": "too_few_samples"}
    x = np.concatenate([x_real, x_synth], axis=0)
    y = np.concatenate([np.zeros(len(x_real), dtype=int), np.ones(len(x_synth), dtype=int)])
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
    aucs: list[float] = []
    auprcs: list[float] = []
    for train_idx, test_idx in splitter.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(x_train, y[train_idx])
        score = clf.predict_proba(x_test)[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], score)))
        auprcs.append(float(average_precision_score(y[test_idx], score)))
    return {
        "n_real": int(len(x_real)),
        "n_synthetic": int(len(x_synth)),
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "status": "ok",
    }


def utility_rows(
    encoder_name: str,
    real_x: np.ndarray,
    real_y: np.ndarray,
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    synth_rows: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
) -> list[dict[str, Any]]:
    same_models = {}
    confusing_models = {}
    for idx, label in idx_to_class.items():
        same = real_x[real_y == idx]
        if len(same):
            same_models[label] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(same)
        confusing = DEFAULT_CONFUSING.get(label, [])
        confusing_idx = [class_to_idx[item] for item in confusing if item in class_to_idx]
        mask = np.isin(real_y, confusing_idx) if confusing_idx else real_y != idx
        if mask.sum() == 0:
            mask = real_y != idx
        confusing_models[label] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[mask])

    rows: list[dict[str, Any]] = []
    for feat, target_idx, row in zip(synth_x, synth_y, synth_rows):
        label = idx_to_class[int(target_idx)]
        same_dist = float(same_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        confusing_dist = float(confusing_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        rows.append(
            {
                "encoder": encoder_name,
                "image_path": row["image_path"],
                "label": label,
                "same_class_distance": same_dist,
                "nearest_confusing_distance": confusing_dist,
                "feature_margin": confusing_dist - same_dist,
            }
        )
    return rows


def topk_overlap(rows: list[dict[str, Any]], top_k: int, target_classes: list[str]) -> list[dict[str, Any]]:
    by_encoder_class: dict[tuple[str, str], set[str]] = {}
    encoders = sorted({row["encoder"] for row in rows})
    for encoder in encoders:
        for label in target_classes:
            items = [row for row in rows if row["encoder"] == encoder and row["label"] == label]
            top = sorted(items, key=lambda item: (-float(item["feature_margin"]), float(item["same_class_distance"])))[:top_k]
            by_encoder_class[(encoder, label)] = {str(row["image_path"]) for row in top}

    out: list[dict[str, Any]] = []
    for left, right in combinations(encoders, 2):
        for label in target_classes:
            a = by_encoder_class[(left, label)]
            b = by_encoder_class[(right, label)]
            union = a | b
            out.append(
                {
                    "left_encoder": left,
                    "right_encoder": right,
                    "label": label,
                    "top_k": top_k,
                    "left_count": len(a),
                    "right_count": len(b),
                    "intersection": len(a & b),
                    "jaccard": float(len(a & b) / len(union)) if union else 0.0,
                }
            )
    return out


def bkl_cluster_report(
    real_x: np.ndarray,
    real_y: np.ndarray,
    real_rows: list[dict[str, Any]],
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    synth_rows: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    clusters: int,
    seed: int,
) -> dict[str, Any]:
    bkl_idx = class_to_idx["bkl"]
    real_mask = real_y == bkl_idx
    synth_mask = synth_y == bkl_idx
    x_real = real_x[real_mask]
    x_synth = synth_x[synth_mask]
    real_bkl_rows = [row for row, flag in zip(real_rows, real_mask) if bool(flag)]
    synth_bkl_rows = [row for row, flag in zip(synth_rows, synth_mask) if bool(flag)]
    if len(x_real) < clusters or len(x_synth) == 0:
        return {"status": "too_few_samples", "n_real_bkl": int(len(x_real)), "n_synthetic_bkl": int(len(x_synth))}

    kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init=20)
    real_cluster = kmeans.fit_predict(x_real)
    synth_cluster = kmeans.predict(x_synth)
    distances = 1.0 - np.max(x_synth @ kmeans.cluster_centers_.T, axis=1)
    rows = []
    for cluster_id in range(clusters):
        real_count = int((real_cluster == cluster_id).sum())
        synth_count = int((synth_cluster == cluster_id).sum())
        synth_dist = distances[synth_cluster == cluster_id]
        rows.append(
            {
                "cluster": cluster_id,
                "real_bkl_count": real_count,
                "synthetic_bkl_count": synth_count,
                "synthetic_to_centroid_distance_mean": float(np.mean(synth_dist)) if len(synth_dist) else None,
                "synthetic_to_centroid_distance_max": float(np.max(synth_dist)) if len(synth_dist) else None,
            }
        )
    return {
        "status": "ok",
        "n_real_bkl": int(len(x_real)),
        "n_synthetic_bkl": int(len(x_synth)),
        "clusters": rows,
        "real_cluster_counts": dict(Counter(int(x) for x in real_cluster)),
        "synthetic_cluster_counts": dict(Counter(int(x) for x in synth_cluster)),
        "synthetic_examples": [
            {
                "image_path": row["image_path"],
                "cluster": int(cluster),
                "distance": float(distance),
                "source_image_id": row.get("source_image_id", ""),
            }
            for row, cluster, distance in zip(synth_bkl_rows, synth_cluster, distances)
        ],
        "real_examples": [
            {
                "image_path": row["image_path"],
                "cluster": int(cluster),
                "image_id": row.get("image_id", ""),
            }
            for row, cluster in zip(real_bkl_rows, real_cluster)
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_html(report: dict[str, Any]) -> str:
    detector_rows = []
    for item in report["detectors"]:
        detector_rows.append(
            "<tr>"
            f"<td>{html.escape(item['encoder'])}</td>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item.get('n_real', '')}</td>"
            f"<td>{item.get('n_synthetic', '')}</td>"
            f"<td>{item.get('auroc_mean', 0):.4f}</td>"
            f"<td>{item.get('auprc_mean', 0):.4f}</td>"
            "</tr>"
        )
    overlap_rows = []
    for item in report["topk_overlap"]:
        overlap_rows.append(
            "<tr>"
            f"<td>{html.escape(item['left_encoder'])}</td>"
            f"<td>{html.escape(item['right_encoder'])}</td>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item['intersection']}/{item['top_k']}</td>"
            f"<td>{item['jaccard']:.4f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>Stage5 Hypothesis Diagnostics</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin-bottom:24px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f6f8fa}}</style></head>
<body>
<h1>Stage5 Hypothesis Diagnostics</h1>
<h2>Class-wise real-vs-synthetic detector</h2>
<table><thead><tr><th>Encoder</th><th>Class</th><th>Real</th><th>Synthetic</th><th>AUROC</th><th>AUPRC</th></tr></thead><tbody>{''.join(detector_rows)}</tbody></table>
<h2>Top-k utility overlap across encoders</h2>
<table><thead><tr><th>Left</th><th>Right</th><th>Class</th><th>Intersection</th><th>Jaccard</th></tr></thead><tbody>{''.join(overlap_rows)}</tbody></table>
<p><a href="stage5_hypothesis_report.json">JSON report</a> | <a href="utility_scores_by_encoder.csv">utility scores</a></p>
</body></html>"""


def main() -> None:
    args = parse_args()
    encoder_specs = [parse_named_path(item) for item in args.encoder]
    primary_name, primary_run = encoder_specs[0]
    primary_cfg = load_config(
        primary_run / "config.resolved.yaml",
        [
            f"training.batch_size={args.batch_size}",
            f"runtime.num_workers={args.num_workers}",
            "runtime.persistent_workers=false",
            "runtime.prefetch_factor=2",
        ],
    )
    data_root = Path(primary_cfg["data"]["root"])
    out_dir = resolve_path(data_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_csv = resolve_path(data_root, args.synthetic_csv)
    target_classes = [item.strip() for item in args.target_classes.split(",") if item.strip()]
    target_set = set(target_classes)

    train_rows = [
        row
        for row in read_rows(resolve_path(data_root, primary_cfg["data"]["train_csv"]))
        if str(row.get("label")) in target_set and int(row.get("is_synthetic", 0)) == 0
    ]
    synth_rows = [row for row in read_rows(synthetic_csv) if str(row.get("label")) in target_set]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detectors: list[dict[str, Any]] = []
    utility_all: list[dict[str, Any]] = []
    bkl_report: dict[str, Any] | None = None

    for encoder_name, run_dir in encoder_specs:
        cfg = load_config(
            run_dir / "config.resolved.yaml",
            [
                f"training.batch_size={args.batch_size}",
                f"runtime.num_workers={args.num_workers}",
                "runtime.persistent_workers=false",
                "runtime.prefetch_factor=2",
            ],
        )
        bundle = make_dataloaders(cfg)
        idx_to_class = bundle.idx_to_class
        model = create_model(cfg, len(bundle.class_to_idx)).to(device)
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        transform = build_transforms(cfg, train=False)
        real_x, real_y, real_ordered = collect_features(
            model,
            train_rows,
            data_root,
            bundle.class_to_idx,
            transform,
            args.batch_size,
            args.num_workers,
            device,
            f"{encoder_name} real",
        )
        synth_x, synth_y, synth_ordered = collect_features(
            model,
            synth_rows,
            data_root,
            bundle.class_to_idx,
            transform,
            args.batch_size,
            args.num_workers,
            device,
            f"{encoder_name} synthetic",
        )

        for label in target_classes:
            class_idx = bundle.class_to_idx[label]
            metrics = detector_metrics(real_x[real_y == class_idx], synth_x[synth_y == class_idx], args.seed)
            metrics.update({"encoder": encoder_name, "label": label})
            detectors.append(metrics)

        utility_all.extend(
            utility_rows(
                encoder_name,
                real_x,
                real_y,
                synth_x,
                synth_y,
                synth_ordered,
                bundle.class_to_idx,
                idx_to_class,
            )
        )
        if encoder_name == primary_name and "bkl" in bundle.class_to_idx:
            bkl_report = bkl_cluster_report(
                real_x,
                real_y,
                real_ordered,
                synth_x,
                synth_y,
                synth_ordered,
                bundle.class_to_idx,
                args.bkl_clusters,
                args.seed,
            )

    overlap = topk_overlap(utility_all, args.top_k, target_classes)
    write_csv(out_dir / "classwise_real_vs_synthetic_detector.csv", detectors)
    write_csv(out_dir / "utility_scores_by_encoder.csv", utility_all)
    write_csv(out_dir / "topk_utility_overlap.csv", overlap)
    if bkl_report is not None and bkl_report.get("status") == "ok":
        write_csv(out_dir / "bkl_cluster_summary.csv", bkl_report["clusters"])
        write_csv(out_dir / "bkl_synthetic_cluster_assignments.csv", bkl_report["synthetic_examples"])
        write_csv(out_dir / "bkl_real_cluster_assignments.csv", bkl_report["real_examples"])

    report = {
        "encoders": [{"name": name, "run_dir": str(path)} for name, path in encoder_specs],
        "synthetic_csv": str(synthetic_csv),
        "target_classes": target_classes,
        "detectors": detectors,
        "topk_overlap": overlap,
        "bkl_clusters": bkl_report,
    }
    (out_dir / "stage5_hypothesis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
