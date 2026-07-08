from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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
            "label": row["label"],
            "path": str(path),
            "row_index": index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select useful synthetic samples by feature-space diagnostics.")
    parser.add_argument("--baseline-run-dir", required=True, help="Run directory with best.pt and config.resolved.yaml.")
    parser.add_argument("--synthetic-csv", required=True, help="Synthetic manifest relative to data root or absolute.")
    parser.add_argument("--out-dir", default="/srv/research/projects/default/ham10000/splits/stage2")
    parser.add_argument("--selected-name", default="selected_synthetic.csv")
    parser.add_argument("--diagnostics-name", default="synthetic_selection_diagnostics.csv")
    parser.add_argument("--report-name", default="synthetic_selection_report.json")
    parser.add_argument("--train-name", default="train_stage2_selected.csv")
    parser.add_argument("--same-quantile", type=float, default=0.95)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--selection-mode", choices=["rule", "topk", "rule_or_topk"], default="rule")
    parser.add_argument("--top-k-per-class", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--diversity-min-distance", type=float, default=0.03)
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model.eval()
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []
    for batch in tqdm(loader, desc="features", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images).detach().float().cpu().numpy()
        features.append(feats)
        targets.append(batch["target"].numpy())
        indices.extend([int(x) for x in batch["row_index"]])
    x = np.concatenate(features)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    y = np.concatenate(targets)
    return x, y, indices


def nearest_self_threshold(features: np.ndarray, quantile: float) -> float:
    if len(features) < 3:
        return 1.0
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(features)
    distances = nn.kneighbors(features, return_distance=True)[0][:, 1]
    return float(np.quantile(distances, quantile))


def greedy_diverse(candidates: list[dict[str, Any]], max_count: int, min_distance: float) -> list[dict[str, Any]]:
    if max_count <= 0:
        max_count = len(candidates)
    chosen: list[dict[str, Any]] = []
    chosen_features: list[np.ndarray] = []
    for row in sorted(candidates, key=lambda r: (-float(r["feature_margin"]), float(r["same_class_distance"]))):
        feat = row["_feature"]
        if chosen_features:
            distances = [1.0 - float(np.dot(feat, other)) for other in chosen_features]
            if min(distances) < min_distance:
                continue
        chosen.append(row)
        chosen_features.append(feat)
        if len(chosen) >= max_count:
            break
    for row in chosen:
        row.pop("_feature", None)
    return chosen


def main() -> None:
    args = parse_args()
    run_dir = Path(args.baseline_run_dir)
    cfg = load_config(run_dir / "config.resolved.yaml", [
        f"training.batch_size={args.batch_size}",
        f"runtime.num_workers={args.num_workers}",
        "runtime.persistent_workers=false",
        "runtime.prefetch_factor=2",
    ])
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

    real_rows = read_rows(data_root / cfg["data"]["train_csv"])
    synthetic_rows = [row for row in read_rows(synthetic_csv) if row["label"] in target_classes]
    transform = build_transforms(cfg, train=False)
    synth_loader = DataLoader(
        ManifestDataset(synthetic_rows, data_root, bundle.class_to_idx, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    synth_x, synth_y, synth_indices = collect_features(model, synth_loader, device)

    real_features: list[np.ndarray] = []
    real_targets: list[np.ndarray] = []
    for split_batch in tqdm(bundle.loaders["train"], desc="real features", leave=False):
        images = split_batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            feats = extract_features(model, images).detach().float().cpu().numpy()
        real_features.append(feats)
        real_targets.append(split_batch["target"].numpy())
    real_x = np.concatenate(real_features)
    real_x = real_x / np.linalg.norm(real_x, axis=1, keepdims=True).clip(min=1e-12)
    real_y = np.concatenate(real_targets)

    idx_to_class = bundle.idx_to_class
    thresholds = {}
    same_models = {}
    other_models = {}
    for cls_idx, cls_name in idx_to_class.items():
        same = real_x[real_y == cls_idx]
        thresholds[cls_name] = nearest_self_threshold(same, args.same_quantile)
        same_models[cls_name] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(same)
        confusing = DEFAULT_CONFUSING.get(cls_name, [])
        confusing_idx = [bundle.class_to_idx[c] for c in confusing if c in bundle.class_to_idx]
        other_mask = np.isin(real_y, confusing_idx) if confusing_idx else real_y != cls_idx
        if other_mask.sum() == 0:
            other_mask = real_y != cls_idx
        other_models[cls_name] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_x[other_mask])

    annotated: list[dict[str, Any]] = []
    all_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rule_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feat, target, row_index in zip(synth_x, synth_y, synth_indices):
        row = dict(synthetic_rows[row_index])
        label = idx_to_class[int(target)]
        same_dist = float(same_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        other_dist = float(other_models[label].kneighbors(feat[None, :], return_distance=True)[0][0, 0])
        margin = other_dist - same_dist
        selected_by_rule = same_dist <= thresholds[label] and margin >= args.min_margin
        row.update(
            {
                "same_class_distance": same_dist,
                "nearest_confusing_distance": other_dist,
                "feature_margin": margin,
                "same_class_threshold": thresholds[label],
                "selected_by_rule": int(selected_by_rule),
            }
        )
        row["_feature"] = feat
        annotated.append(row)
        all_by_class[label].append(row)
        if selected_by_rule:
            rule_by_class[label].append(row)

    selected: list[dict[str, Any]] = []
    top_k_per_class = int(args.top_k_per_class)
    max_per_class = int(args.max_per_class)
    if args.selection_mode == "topk" and top_k_per_class <= 0 and max_per_class <= 0:
        raise ValueError("--selection-mode topk requires --top-k-per-class or --max-per-class")
    for label in sorted(all_by_class):
        if args.selection_mode == "rule":
            candidates = rule_by_class.get(label, [])
            limit = max_per_class
        elif args.selection_mode == "topk":
            candidates = all_by_class[label]
            limit = top_k_per_class or max_per_class
        else:
            candidates_by_path = {row["image_path"]: row for row in rule_by_class.get(label, [])}
            fallback = (
                sorted(all_by_class[label], key=lambda r: (-float(r["feature_margin"]), float(r["same_class_distance"])))[:top_k_per_class]
                if top_k_per_class > 0
                else []
            )
            for row in fallback:
                candidates_by_path[row["image_path"]] = row
            candidates = list(candidates_by_path.values())
            limit = max_per_class
        selected_rows = greedy_diverse(candidates, limit, args.diversity_min_distance)
        for row in selected_rows:
            row["selected_by_mode"] = 1
        selected.extend(selected_rows)

    selected_fieldnames = [key for key in annotated[0].keys() if key != "_feature"] if annotated else []
    if "selected_by_mode" not in selected_fieldnames:
        selected_fieldnames.append("selected_by_mode")
    selected_path = out_dir / args.selected_name
    annotated_path = out_dir / args.diagnostics_name
    train_path = out_dir / args.train_name
    write_rows(selected_path, selected, selected_fieldnames)
    write_rows(annotated_path, [{k: v for k, v in row.items() if k != "_feature"} for row in annotated], selected_fieldnames)

    train_rows = [row for row in real_rows if int(row.get("is_synthetic", 0)) == 0]
    train_rows.extend({key: row.get(key, "") for key in ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"]} for row in selected)
    write_rows(train_path, train_rows, ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"])

    report = {
        "baseline_run_dir": str(run_dir),
        "synthetic_csv": str(synthetic_csv),
        "target_classes": sorted(target_classes),
        "thresholds": thresholds,
        "input_synthetic_by_class": dict(Counter(row["label"] for row in synthetic_rows)),
        "rule_pass_by_class": {label: len(rows) for label, rows in rule_by_class.items()},
        "selected_by_class": dict(Counter(row["label"] for row in selected)),
        "selected_total": len(selected),
        "train_csv": str(train_path),
        "selected_csv": str(selected_path),
        "diagnostics_csv": str(annotated_path),
        "same_quantile": args.same_quantile,
        "min_margin": args.min_margin,
        "selection_mode": args.selection_mode,
        "top_k_per_class": top_k_per_class,
        "diversity_min_distance": args.diversity_min_distance,
        "max_per_class": max_per_class,
    }
    (out_dir / args.report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
