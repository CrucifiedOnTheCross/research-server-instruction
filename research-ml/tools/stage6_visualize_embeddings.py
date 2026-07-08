from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
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
from sklearn.manifold import TSNE
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize real and selected synthetic samples in a fixed feature space.")
    parser.add_argument("--encoder-run-dir", required=True, help="Run directory with best.pt and config.resolved.yaml.")
    parser.add_argument("--pool", action="append", required=True, help="name:csv for selected synthetic pool. Can be repeated.")
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/reports/stage6_embedding_visuals")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--max-real-per-class", type=int, default=250)
    parser.add_argument("--max-points-tsne", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Expected name:path, got {value}")
    name, path = value.split(":", 1)
    return name.strip(), path.strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sample_real_rows(rows: list[dict[str, Any]], max_per_class: int, seed: int) -> list[dict[str, Any]]:
    if max_per_class <= 0:
        return rows
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["label"])].append(row)
    sampled: list[dict[str, Any]] = []
    for label in sorted(by_class):
        items = by_class[label]
        if len(items) <= max_per_class:
            sampled.extend(items)
            continue
        idx = rng.choice(np.arange(len(items)), size=max_per_class, replace=False)
        sampled.extend(items[int(i)] for i in sorted(idx))
    return sampled


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
    features: list[np.ndarray] = []
    ordered: list[dict[str, Any]] = []
    model.eval()
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        features.append(feats)
        ordered.extend(rows[int(i)] for i in batch["row_index"])
    x = np.concatenate(features, axis=0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x, ordered


def plot_embedding(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    target_classes: list[str],
    only_class: str | None = None,
) -> None:
    plot_df = frame.copy()
    if only_class is not None:
        plot_df = plot_df[plot_df["label"] == only_class].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"mel": "#d55e00", "akiec": "#0072b2", "bkl": "#009e73", "bcc": "#cc79a7", "nv": "#999999"}
    markers = {
        "real": "o",
        "stage3_utility": "X",
        "stage4_mel_akiec": "^",
        "stage6_geometry": "s",
    }
    sizes = {"real": 18, "stage3_utility": 48, "stage4_mel_akiec": 42, "stage6_geometry": 46}
    alphas = {"real": 0.28, "stage3_utility": 0.80, "stage4_mel_akiec": 0.80, "stage6_geometry": 0.90}

    for source in ["real", "stage3_utility", "stage4_mel_akiec", "stage6_geometry"]:
        for label in target_classes:
            part = plot_df[(plot_df["source"] == source) & (plot_df["label"] == label)]
            if part.empty:
                continue
            ax.scatter(
                part[x_col],
                part[y_col],
                s=sizes.get(source, 32),
                marker=markers.get(source, "o"),
                c=colors.get(label, "#333333"),
                alpha=alphas.get(source, 0.75),
                edgecolors="black" if source != "real" else "none",
                linewidths=0.45 if source != "real" else 0,
                label=f"{source}:{label}",
            )
    ax.set_title(title if only_class is None else f"{title}: {only_class}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_html(report: dict[str, Any]) -> str:
    image_links = []
    for item in report["plots"]:
        image_links.append(
            f"<section><h2>{html.escape(item['title'])}</h2>"
            f"<p><a href=\"{html.escape(item['file'])}\">{html.escape(item['file'])}</a></p>"
            f"<img src=\"{html.escape(item['file'])}\" style=\"max-width:100%;height:auto\"></section>"
        )
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>Stage6 Embedding Visuals</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#17202a}}section{{margin-bottom:32px}}img{{border:1px solid #d8dee4}}</style></head>
<body>
<h1>Stage6 Embedding Visuals</h1>
<p>Фиксированное пространство признаков: {html.escape(report['encoder_run_dir'])}</p>
<p><a href="embedding_coordinates.csv">embedding_coordinates.csv</a> | <a href="embedding_visual_report.json">embedding_visual_report.json</a></p>
{''.join(image_links)}
</body></html>"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.encoder_run_dir)
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
    target_classes = [item.strip() for item in args.target_classes.split(",") if item.strip()]
    target_set = set(target_classes)

    train_csv = resolve_path(data_root, cfg["data"]["train_csv"])
    real_rows = [
        {**row, "source": "real", "is_synthetic": "0"}
        for row in read_rows(train_csv)
        if str(row.get("label")) in target_set and int(row.get("is_synthetic", 0)) == 0
    ]
    real_rows = sample_real_rows(real_rows, args.max_real_per_class, args.seed)

    pool_rows: list[dict[str, Any]] = []
    for pool_arg in args.pool:
        name, path_value = parse_named_path(pool_arg)
        pool_path = resolve_path(data_root, path_value)
        rows = [
            {**row, "source": name, "is_synthetic": str(row.get("is_synthetic", 1) or 1)}
            for row in read_rows(pool_path)
            if str(row.get("label")) in target_set and int(row.get("is_synthetic", 1)) == 1
        ]
        pool_rows.extend(rows)

    rows_all = real_rows + pool_rows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(cfg)
    model = create_model(cfg, len(bundle.class_to_idx)).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    transform = build_transforms(cfg, train=False)
    features, ordered_rows = collect_features(
        model, rows_all, data_root, bundle.class_to_idx, transform, args.batch_size, args.num_workers, device, "embedding features"
    )

    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca.fit_transform(scaled)
    tsne_input = PCA(n_components=min(50, scaled.shape[1], max(2, len(scaled) - 1)), random_state=args.seed).fit_transform(scaled)
    if len(tsne_input) > args.max_points_tsne:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(np.arange(len(tsne_input)), size=args.max_points_tsne, replace=False))
    else:
        keep = np.arange(len(tsne_input))
    perplexity = min(35, max(5, (len(keep) - 1) // 3))
    tsne_xy = np.full((len(tsne_input), 2), np.nan, dtype=float)
    tsne_xy[keep] = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        metric="euclidean",
    ).fit_transform(tsne_input[keep])

    records = []
    for idx, row in enumerate(ordered_rows):
        records.append(
            {
                "image_path": row.get("image_path", ""),
                "label": row.get("label", ""),
                "source": row.get("source", ""),
                "is_synthetic": row.get("is_synthetic", ""),
                "source_image_id": row.get("source_image_id", ""),
                "pca_x": float(pca_xy[idx, 0]),
                "pca_y": float(pca_xy[idx, 1]),
                "tsne_x": float(tsne_xy[idx, 0]) if not np.isnan(tsne_xy[idx, 0]) else "",
                "tsne_y": float(tsne_xy[idx, 1]) if not np.isnan(tsne_xy[idx, 1]) else "",
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "embedding_coordinates.csv", index=False)

    plots: list[dict[str, str]] = []
    plot_specs = [
        ("pca", "pca_x", "pca_y", "PCA feature projection"),
        ("tsne", "tsne_x", "tsne_y", "t-SNE feature projection"),
    ]
    for prefix, x_col, y_col, title in plot_specs:
        plot_frame = frame.dropna(subset=[x_col, y_col])
        file_name = f"{prefix}_all_classes.png"
        plot_embedding(plot_frame, x_col, y_col, title, out_dir / file_name, target_classes)
        plots.append({"title": title, "file": file_name})
        for label in target_classes:
            file_name = f"{prefix}_{label}.png"
            plot_embedding(plot_frame, x_col, y_col, title, out_dir / file_name, target_classes, only_class=label)
            plots.append({"title": f"{title}: {label}", "file": file_name})

    report = {
        "encoder_run_dir": str(run_dir),
        "target_classes": target_classes,
        "counts": frame.groupby(["source", "label"]).size().reset_index(name="count").to_dict(orient="records"),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "tsne_points": int(len(keep)),
        "tsne_perplexity": int(perplexity),
        "plots": plots,
    }
    (out_dir / "embedding_visual_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
