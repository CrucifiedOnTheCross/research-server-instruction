from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
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


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, class_to_idx: dict[str, int], transform: Any) -> None:
        self.rows = rows
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long),
            "path": str(path),
            "row_index": index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score and select synthetic samples with Stage 3 utility criteria.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--hard-cases-csv", required=True)
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/splits/stage3")
    parser.add_argument("--selected-name", default="selected_synthetic_utility.csv")
    parser.add_argument("--diagnostics-name", default="synthetic_utility_diagnostics.csv")
    parser.add_argument("--train-name", default="train_stage3_utility.csv")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--top-k-per-class", type=int, default=80)
    parser.add_argument("--min-feature-margin", type=float, default=0.02)
    parser.add_argument("--max-source-distance", type=float, default=0.28)
    parser.add_argument("--diversity-min-distance", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
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


@torch.no_grad()
def collect_features(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model.eval()
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        features.append(feats)
        targets.append(batch["target"].numpy())
        indices.extend([int(x) for x in batch["row_index"]])
    x = np.concatenate(features)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x, np.concatenate(targets), indices


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def source_key(row: dict[str, Any]) -> str:
    for key in ("source_image_id", "source_image_path"):
        value = str(row.get(key, "")).strip()
        if value:
            return Path(value).stem
    return ""


def greedy_diverse(rows: list[dict[str, Any]], max_count: int, min_distance: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_features: list[np.ndarray] = []
    for row in sorted(rows, key=lambda item: float(item["selection_score"]), reverse=True):
        feat = row["_feature"]
        if selected_features:
            distances = [1.0 - float(np.dot(feat, other)) for other in selected_features]
            if min(distances) < min_distance:
                continue
        selected.append(row)
        selected_features.append(feat)
        if len(selected) >= max_count:
            break
    return selected


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
    synthetic_csv = Path(args.synthetic_csv)
    if not synthetic_csv.is_absolute():
        synthetic_csv = data_root / synthetic_csv
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = data_root / out_dir
    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(cfg)
    model = create_model(cfg, len(bundle.class_to_idx)).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    train_rows = read_rows(data_root / cfg["data"]["train_csv"])
    real_train_rows = [row for row in train_rows if int(row.get(cfg["data"]["synthetic_col"], 0)) == 0]
    synthetic_rows = [row for row in read_rows(synthetic_csv) if row["label"] in target_classes]

    transform = build_transforms(cfg, train=False)
    synth_loader = DataLoader(
        ManifestDataset(synthetic_rows, data_root, bundle.class_to_idx, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    synth_x, synth_y, synth_indices = collect_features(model, synth_loader, device, "synthetic features")

    real_features: list[np.ndarray] = []
    real_targets: list[np.ndarray] = []
    real_paths: list[str] = []
    for batch in tqdm(bundle.loaders["train"], desc="real features", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        real_features.append(feats)
        real_targets.append(batch["target"].numpy())
        real_paths.extend(batch["path"])
    real_x = np.concatenate(real_features)
    real_x = real_x / np.linalg.norm(real_x, axis=1, keepdims=True).clip(min=1e-12)
    real_y = np.concatenate(real_targets)

    hard = pd.read_csv(args.hard_cases_csv)
    hard = hard[hard["split"].astype(str) == "train"].copy()
    hard["source_key"] = hard.apply(lambda row: Path(str(row.get("image_id") or row.get("image_path") or row.get("abs_path"))).stem, axis=1)
    hard_by_source = hard.sort_values("hard_score", ascending=False).drop_duplicates("source_key").set_index("source_key")

    idx_to_class = bundle.idx_to_class
    same_models = {}
    confusing_models = {}
    source_features: dict[str, np.ndarray] = {}
    for path, feat in zip(real_paths, real_x):
        source_features[Path(path).stem] = feat

    for cls_idx, cls_name in idx_to_class.items():
        same_models[cls_name] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[real_y == cls_idx])
        confusing = DEFAULT_CONFUSING.get(cls_name, [])
        confusing_idx = [bundle.class_to_idx[c] for c in confusing if c in bundle.class_to_idx]
        mask = np.isin(real_y, confusing_idx) if confusing_idx else real_y != cls_idx
        if mask.sum() == 0:
            mask = real_y != cls_idx
        confusing_models[cls_name] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[mask])

    annotated: list[dict[str, Any]] = []
    for feat, target, row_index in zip(synth_x, synth_y, synth_indices):
        row = dict(synthetic_rows[row_index])
        label = idx_to_class[int(target)]
        same_distance = float(same_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        confusing_distance = float(confusing_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        key = source_key(row)
        hard_row = hard_by_source.loc[key] if key in hard_by_source.index else None
        source_feat = source_features.get(key)
        source_distance = float(1.0 - np.dot(feat, source_feat)) if source_feat is not None else same_distance
        row.update(
            {
                "same_class_distance": same_distance,
                "nearest_confusing_distance": confusing_distance,
                "feature_margin": confusing_distance - same_distance,
                "source_distance": source_distance,
                "source_hard_score": float(hard_row["hard_score"]) if hard_row is not None else 0.0,
                "source_entropy": float(hard_row["entropy"]) if hard_row is not None else 0.0,
                "source_confidence": float(hard_row["confidence"]) if hard_row is not None else 0.0,
                "source_is_correct": int(hard_row["is_correct"]) if hard_row is not None else -1,
            }
        )
        row["_feature"] = feat
        annotated.append(row)

    feature_margin = normalize(np.array([float(row["feature_margin"]) for row in annotated]))
    source_hard = normalize(np.array([float(row["source_hard_score"]) for row in annotated]))
    source_distance = np.array([float(row["source_distance"]) for row in annotated])
    duplicate_penalty = np.clip((float(args.max_source_distance) - source_distance) / max(float(args.max_source_distance), 1e-12), 0.0, 1.0)
    negative_margin_penalty = np.array([1.0 if float(row["feature_margin"]) < float(args.min_feature_margin) else 0.0 for row in annotated])
    for idx, row in enumerate(annotated):
        row["normalized_feature_margin"] = float(feature_margin[idx])
        row["normalized_source_hard_score"] = float(source_hard[idx])
        row["duplicate_penalty"] = float(duplicate_penalty[idx])
        row["negative_margin_penalty"] = float(negative_margin_penalty[idx])
        row["selection_score"] = float(feature_margin[idx] + 0.35 * source_hard[idx] - 0.25 * duplicate_penalty[idx] - 0.50 * negative_margin_penalty[idx])
        row["passes_stage3_filters"] = int(
            float(row["feature_margin"]) >= float(args.min_feature_margin)
            and float(row["source_distance"]) <= float(args.max_source_distance)
        )

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        by_class[row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    for label in sorted(by_class):
        candidates = [row for row in by_class[label] if int(row["passes_stage3_filters"]) == 1]
        if len(candidates) < int(args.top_k_per_class):
            candidates = by_class[label]
        chosen = greedy_diverse(candidates, int(args.top_k_per_class), float(args.diversity_min_distance))
        for row in chosen:
            row["selected_by_stage3"] = 1
        selected.extend(chosen)

    diagnostics_path = out_dir / args.diagnostics_name
    selected_path = out_dir / args.selected_name
    train_path = out_dir / args.train_name
    fields = [key for key in annotated[0] if not key.startswith("_")] if annotated else []
    if "selected_by_stage3" not in fields:
        fields.append("selected_by_stage3")
    write_rows(diagnostics_path, annotated, fields)
    write_rows(selected_path, selected, fields)

    train_out_fields = ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"]
    train_stage3 = [{key: row.get(key, "") for key in train_out_fields} for row in real_train_rows]
    train_stage3.extend({key: row.get(key, "") for key in train_out_fields} for row in selected)
    write_rows(train_path, train_stage3, train_out_fields)

    report = {
        "baseline_run_dir": str(run_dir),
        "synthetic_csv": str(synthetic_csv),
        "hard_cases_csv": args.hard_cases_csv,
        "target_classes": sorted(target_classes),
        "input_synthetic_by_class": dict(Counter(row["label"] for row in synthetic_rows)),
        "selected_by_class": dict(Counter(row["label"] for row in selected)),
        "selected_total": len(selected),
        "top_k_per_class": int(args.top_k_per_class),
        "min_feature_margin": float(args.min_feature_margin),
        "max_source_distance": float(args.max_source_distance),
        "train_csv": str(train_path),
        "selected_csv": str(selected_path),
        "diagnostics_csv": str(diagnostics_path),
        "mean_selection_score": float(np.mean([float(row["selection_score"]) for row in selected])) if selected else 0.0,
    }
    (out_dir / "synthetic_utility_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
