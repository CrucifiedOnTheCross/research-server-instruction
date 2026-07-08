from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.neighbors import NearestNeighbors
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
        path = resolve_path(self.root, str(row["image_path"]))
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(self.class_to_idx[str(row["label"])], dtype=torch.long),
            "row_index": index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build kNN visual audit gallery for synthetic images.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/reports/stage2_diagnostics/knn_gallery")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)[:180]


def sample_rows(rows: list[dict[str, str]], max_per_class: int, seed: int) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return rows
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["label"])].append(row)
    sampled: list[dict[str, str]] = []
    for label in sorted(by_class):
        items = by_class[label]
        if len(items) <= max_per_class:
            sampled.extend(items)
        else:
            indices = rng.choice(np.arange(len(items)), size=max_per_class, replace=False)
            sampled.extend(items[int(i)] for i in sorted(indices))
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


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(draw: ImageDraw.ImageDraw, x: int, title: str, subtitle: str, size: int) -> None:
    font = load_font(14)
    small = load_font(12)
    draw.text((x + 6, 6), title, fill=(20, 20, 20), font=font)
    draw.text((x + 6, 28), subtitle[:80], fill=(90, 90, 90), font=small)
    draw.rectangle((x, 50, x + size - 1, 50 + size - 1), outline=(218, 224, 230), width=1)


def make_composite(
    out_path: Path,
    data_root: Path,
    synthetic: dict[str, Any],
    same: dict[str, Any],
    confusing: dict[str, Any],
    same_distance: float,
    confusing_distance: float,
    size: int,
) -> None:
    panels = [
        ("source", synthetic.get("source_image_path", ""), synthetic.get("source_image_id", "")),
        ("synthetic", synthetic.get("image_path", ""), synthetic.get("image_id", "")),
        (f"nearest same d={same_distance:.4f}", same.get("image_path", ""), same.get("image_id", Path(str(same.get("image_path", ""))).stem)),
        (
            f"nearest confusing d={confusing_distance:.4f}",
            confusing.get("image_path", ""),
            f"{confusing.get('label', '')} {confusing.get('image_id', Path(str(confusing.get('image_path', ''))).stem)}",
        ),
    ]
    gutter = 12
    label_h = 54
    canvas = Image.new("RGB", (size * 4 + gutter * 3, size + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, path_value, subtitle) in enumerate(panels):
        x = idx * (size + gutter)
        draw_panel(draw, x, title, str(subtitle), size)
        path = resolve_path(data_root, str(path_value))
        if path.exists():
            canvas.paste(fit_image(path, size), (x, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


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
    out_root = resolve_path(data_root, args.out_dir) / safe_name(args.pool_name)
    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(cfg)
    model = create_model(cfg, len(bundle.class_to_idx)).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    transform = build_transforms(cfg, train=False)

    train_csv = resolve_path(data_root, cfg["data"]["train_csv"])
    real_rows = [row for row in read_rows(train_csv) if int(row.get("is_synthetic", 0)) == 0]
    for row in real_rows:
        row.setdefault("image_id", Path(str(row["image_path"])).stem)

    synthetic_csv = resolve_path(data_root, args.synthetic_csv)
    synthetic_rows = [row for row in read_rows(synthetic_csv) if str(row.get("label")) in target_classes]
    synthetic_rows = sample_rows(synthetic_rows, args.max_per_class, args.seed)

    real_x, real_rows = collect_features(
        model, real_rows, data_root, bundle.class_to_idx, transform, args.batch_size, args.num_workers, device, "real features"
    )
    synth_x, synthetic_rows = collect_features(
        model, synthetic_rows, data_root, bundle.class_to_idx, transform, args.batch_size, args.num_workers, device, "synthetic features"
    )

    by_class_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(real_rows):
        by_class_indices[str(row["label"])].append(idx)
    nn_by_class: dict[str, NearestNeighbors] = {}
    for label, indices in by_class_indices.items():
        nn_by_class[label] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[indices])

    audit_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for idx, (feat, row) in enumerate(zip(synth_x, synthetic_rows), start=1):
        label = str(row["label"])
        same_indices = by_class_indices[label]
        same_dist, same_local = nn_by_class[label].kneighbors(feat[None, :], return_distance=True)
        same_real = real_rows[same_indices[int(same_local[0, 0])]]
        same_distance = float(same_dist[0, 0])

        confusing_labels = [c for c in DEFAULT_CONFUSING.get(label, []) if c in by_class_indices]
        if not confusing_labels:
            confusing_labels = [c for c in by_class_indices if c != label]
        confusing_indices = [i for c in confusing_labels for i in by_class_indices[c]]
        confusing_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[confusing_indices])
        confusing_dist, confusing_local = confusing_nn.kneighbors(feat[None, :], return_distance=True)
        confusing_real = real_rows[confusing_indices[int(confusing_local[0, 0])]]
        confusing_distance = float(confusing_dist[0, 0])
        margin = confusing_distance - same_distance

        stem = safe_name(f"{idx:04d}_{label}_{row.get('image_id', Path(str(row['image_path'])).stem)}.jpg")
        rel_pair = f"pairs/{safe_name(label)}/{stem}"
        pair_path = out_root / rel_pair
        make_composite(pair_path, data_root, row, same_real, confusing_real, same_distance, confusing_distance, args.thumb_size)

        audit = {
            "label": label,
            "synthetic_image_path": row.get("image_path", ""),
            "synthetic_image_id": row.get("image_id", ""),
            "source_image_path": row.get("source_image_path", ""),
            "source_image_id": row.get("source_image_id", ""),
            "nearest_same_path": same_real.get("image_path", ""),
            "nearest_same_label": same_real.get("label", ""),
            "nearest_same_distance": same_distance,
            "nearest_confusing_path": confusing_real.get("image_path", ""),
            "nearest_confusing_label": confusing_real.get("label", ""),
            "nearest_confusing_distance": confusing_distance,
            "feature_margin": margin,
            "pair_image": rel_pair,
        }
        audit_rows.append(audit)
        cards.append(
            "<article>"
            f"<a href=\"{html.escape(rel_pair)}\"><img loading=\"lazy\" src=\"{html.escape(rel_pair)}\" alt=\"knn pair\"></a>"
            f"<h3>{html.escape(label)} · margin={margin:.4f}</h3>"
            f"<div>same d={same_distance:.4f}; confusing {html.escape(str(confusing_real.get('label', '')))} d={confusing_distance:.4f}</div>"
            f"<div>{html.escape(str(row.get('image_id', '')))}</div>"
            "</article>"
        )

    write_rows(out_root / "knn_audit.csv", audit_rows)
    summary = {
        "pool_name": args.pool_name,
        "synthetic_csv": str(synthetic_csv),
        "total": len(audit_rows),
        "by_class": dict(Counter(row["label"] for row in audit_rows)),
        "mean_same_distance": float(np.mean([row["nearest_same_distance"] for row in audit_rows])) if audit_rows else None,
        "mean_confusing_distance": float(np.mean([row["nearest_confusing_distance"] for row in audit_rows])) if audit_rows else None,
        "mean_margin": float(np.mean([row["feature_margin"] for row in audit_rows])) if audit_rows else None,
        "negative_margin_count": int(sum(1 for row in audit_rows if row["feature_margin"] < 0)),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kNN Synthetic Audit: {html.escape(args.pool_name)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 20px; color: #17202a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 10px; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    h3 {{ font-size: 15px; margin: 8px 0 4px; }}
    div {{ color: #586069; font-size: 13px; }}
    a {{ color: #0969da; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>kNN Synthetic Audit: {html.escape(args.pool_name)}</h1>
  <p>Панели: source real -> synthetic -> nearest real same-class -> nearest real confusing-class.</p>
  <p>Total: {len(audit_rows)}. By class: {html.escape(json.dumps(summary['by_class'], ensure_ascii=False))}. Mean margin: {summary['mean_margin']:.4f}.</p>
  <p><a href="knn_audit.csv">knn_audit.csv</a> · <a href="summary.json">summary.json</a></p>
  <main class="grid">{''.join(cards)}</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(out_root)


if __name__ == "__main__":
    main()
