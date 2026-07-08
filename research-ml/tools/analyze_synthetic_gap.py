from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.datasets import build_transforms, make_dataloaders
from src.models import create_model, extract_features


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


def parse_selection(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Pool must have name:path form: {value}")
    name, path = value.split(":", 1)
    return name.strip(), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real-vs-synthetic feature gap for generated image pools.")
    parser.add_argument("--baseline-run-dir", required=True, help="Run directory with best.pt and config.resolved.yaml.")
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        help="Synthetic pool in name:csv form, relative to data root or absolute. Can be repeated.",
    )
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/reports/stage2_diagnostics")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max-plot-per-domain-class", type=int, default=250)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, value: str) -> Path:
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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    loader = DataLoader(
        RowImageDataset(rows, root, class_to_idx, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    features: list[np.ndarray] = []
    ordered_rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        features.append(feats)
        ordered_rows.extend(rows[int(i)] for i in batch["row_index"])
    x = np.concatenate(features, axis=0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x, ordered_rows


def domain_metrics(x_real: np.ndarray, x_synth: np.ndarray, seed: int) -> dict[str, Any]:
    x = np.concatenate([x_real, x_synth], axis=0)
    y = np.concatenate([np.zeros(len(x_real), dtype=int), np.ones(len(x_synth), dtype=int)])
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 4:
        return {"n_real": int(len(x_real)), "n_synthetic": int(len(x_synth)), "status": "too_few_samples"}

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


def balanced_real_indices(
    real_rows: list[dict[str, Any]],
    synth_rows: list[dict[str, Any]],
    target_classes: set[str],
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    real_by_class: dict[str, list[int]] = defaultdict(list)
    synth_counts = Counter(str(row["label"]) for row in synth_rows if str(row["label"]) in target_classes)
    for idx, row in enumerate(real_rows):
        label = str(row["label"])
        if label in target_classes:
            real_by_class[label].append(idx)

    chosen: list[int] = []
    for label, synth_count in sorted(synth_counts.items()):
        candidates = np.asarray(real_by_class[label], dtype=int)
        if len(candidates) == 0:
            continue
        take = min(int(synth_count), len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return sorted(chosen)


def plot_pca(
    out_path: Path,
    real_x: np.ndarray,
    real_rows: list[dict[str, Any]],
    synth_x: np.ndarray,
    synth_rows: list[dict[str, Any]],
    pool_name: str,
    max_per_domain_class: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    plot_x: list[np.ndarray] = []
    plot_domain: list[str] = []
    plot_label: list[str] = []

    for domain, x, rows in [("real", real_x, real_rows), ("synthetic", synth_x, synth_rows)]:
        by_class: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            by_class[str(row["label"])].append(idx)
        for label, indices in by_class.items():
            idx_arr = np.asarray(indices, dtype=int)
            if len(idx_arr) > max_per_domain_class:
                idx_arr = rng.choice(idx_arr, size=max_per_domain_class, replace=False)
            plot_x.append(x[idx_arr])
            plot_domain.extend([domain] * len(idx_arr))
            plot_label.extend([label] * len(idx_arr))

    x_all = np.concatenate(plot_x, axis=0)
    xy = PCA(n_components=2, random_state=seed).fit_transform(x_all)
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = sorted(set(plot_label))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}
    markers = {"real": "o", "synthetic": "x"}
    for domain in ["real", "synthetic"]:
        for label in labels:
            mask = np.asarray([(d == domain and l == label) for d, l in zip(plot_domain, plot_label)])
            if mask.any():
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=18 if domain == "real" else 24,
                    marker=markers[domain],
                    alpha=0.65,
                    color=color_map[label],
                    label=f"{domain}:{label}",
                )
    ax.set_title(f"PCA feature projection: {pool_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_html(report: dict[str, Any]) -> str:
    rows = []
    for pool_name, pool in report["pools"].items():
        overall = pool["overall_detector"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(pool_name)}</td>"
            f"<td>{overall.get('n_real', '')}</td>"
            f"<td>{overall.get('n_synthetic', '')}</td>"
            f"<td>{overall.get('auroc_mean', 0):.4f}</td>"
            f"<td>{overall.get('auprc_mean', 0):.4f}</td>"
            f"<td><a href=\"{html.escape(pool['pca_plot'])}\">PCA plot</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage2 Synthetic Gap Diagnostics</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Stage2 Synthetic Gap Diagnostics</h1>
  <p>AUROC близкий к 1.0 означает, что synthetic features легко отделимы от real features.</p>
  <table>
    <thead><tr><th>Pool</th><th>Real</th><th>Synthetic</th><th>AUROC</th><th>AUPRC</th><th>Plot</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p><a href="real_vs_synthetic_metrics.json">real_vs_synthetic_metrics.json</a></p>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.baseline_run_dir)
    cfg = load_config(
        run_dir / "config.resolved.yaml",
        [
            f"training.batch_size={args.batch_size}",
            f"runtime.num_workers={args.num_workers}",
            "runtime.persistent_workers=false",
            "runtime.prefetch_factor=2",
        ],
    )
    data_root = Path(cfg["data"]["root"])
    out_dir = resolve_path(data_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(cfg)
    model = create_model(cfg, len(bundle.class_to_idx)).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    transform = build_transforms(cfg, train=False)

    train_csv = resolve_path(data_root, cfg["data"]["train_csv"])
    real_rows_all = [
        row
        for row in read_rows(train_csv)
        if str(row.get("label")) in target_classes and int(row.get("is_synthetic", 0)) == 0
    ]
    real_x_all, real_rows_all = collect_features(
        model,
        real_rows_all,
        data_root,
        bundle.class_to_idx,
        transform,
        args.batch_size,
        args.num_workers,
        device,
        "real features",
    )

    report: dict[str, Any] = {
        "baseline_run_dir": str(run_dir),
        "data_root": str(data_root),
        "target_classes": sorted(target_classes),
        "pools": {},
    }
    prediction_rows: list[dict[str, Any]] = []

    for pool_arg in args.pool:
        pool_name, pool_csv = parse_selection(pool_arg)
        pool_path = resolve_path(data_root, pool_csv)
        synth_rows = [row for row in read_rows(pool_path) if str(row.get("label")) in target_classes]
        synth_x, synth_rows = collect_features(
            model,
            synth_rows,
            data_root,
            bundle.class_to_idx,
            transform,
            args.batch_size,
            args.num_workers,
            device,
            f"{pool_name} features",
        )
        real_idx = balanced_real_indices(real_rows_all, synth_rows, target_classes, args.seed)
        real_x = real_x_all[real_idx]
        real_rows = [real_rows_all[idx] for idx in real_idx]

        overall = domain_metrics(real_x, synth_x, args.seed)
        per_class: dict[str, Any] = {}
        for label in sorted(target_classes):
            real_mask = np.asarray([str(row["label"]) == label for row in real_rows])
            synth_mask = np.asarray([str(row["label"]) == label for row in synth_rows])
            if real_mask.any() and synth_mask.any():
                per_class[label] = domain_metrics(real_x[real_mask], synth_x[synth_mask], args.seed)

        pca_path = out_dir / f"pca_real_synthetic_{pool_name}.png"
        plot_pca(pca_path, real_x, real_rows, synth_x, synth_rows, pool_name, args.max_plot_per_domain_class, args.seed)
        report["pools"][pool_name] = {
            "csv": str(pool_path),
            "synthetic_by_class": dict(Counter(str(row["label"]) for row in synth_rows)),
            "matched_real_by_class": dict(Counter(str(row["label"]) for row in real_rows)),
            "overall_detector": overall,
            "per_class_detector": per_class,
            "pca_plot": pca_path.name,
        }

        for row in real_rows:
            prediction_rows.append({"pool": pool_name, "domain": "real", **row})
        for row in synth_rows:
            prediction_rows.append({"pool": pool_name, "domain": "synthetic", **row})

    (out_dir / "real_vs_synthetic_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(out_dir / "real_vs_synthetic_rows.csv", index=False)
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
