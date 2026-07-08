from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .config import load_config
from .datasets import make_dataloaders
from .logging_utils import write_json
from .models import create_model, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory containing config.resolved.yaml and best.pt.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def collect_embeddings(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    synthetic: list[int] = []
    paths: list[str] = []
    labels: list[str] = []
    for batch in tqdm(loader, desc="embeddings", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = extract_features(model, images)
        features.append(feats.detach().float().cpu().numpy())
        targets.append(batch["target"].numpy())
        synthetic.extend([int(x) for x in batch["is_synthetic"]])
        paths.extend(batch["path"])
        labels.extend(batch["label_name"])
    return {
        "features": np.concatenate(features),
        "targets": np.concatenate(targets),
        "is_synthetic": np.asarray(synthetic, dtype=np.int64),
        "paths": paths,
        "labels": labels,
    }


def real_vs_synthetic_auc(features: np.ndarray, is_synthetic: np.ndarray) -> float | None:
    if len(np.unique(is_synthetic)) < 2 or len(is_synthetic) < 20:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        is_synthetic,
        test_size=0.35,
        random_state=42,
        stratify=is_synthetic,
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def nearest_neighbor_diagnostics(features: np.ndarray, targets: np.ndarray, is_synthetic: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    real_mask = is_synthetic == 0
    synth_mask = is_synthetic == 1
    if real_mask.sum() == 0 or synth_mask.sum() == 0:
        return result

    real_features = features[real_mask]
    real_targets = targets[real_mask]
    synth_features = features[synth_mask]
    synth_targets = targets[synth_mask]

    same_distances: list[float] = []
    other_distances: list[float] = []
    margins: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for class_id in sorted(np.unique(targets)):
        synth_idx = synth_targets == class_id
        real_same_idx = real_targets == class_id
        real_other_idx = real_targets != class_id
        if synth_idx.sum() == 0 or real_same_idx.sum() == 0 or real_other_idx.sum() == 0:
            continue
        synth_class = synth_features[synth_idx]
        same_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_features[real_same_idx])
        other_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_features[real_other_idx])
        same = same_nn.kneighbors(synth_class, return_distance=True)[0].ravel()
        other = other_nn.kneighbors(synth_class, return_distance=True)[0].ravel()
        margin = other - same
        same_distances.extend(same.tolist())
        other_distances.extend(other.tolist())
        margins.extend(margin.tolist())
        per_class[str(int(class_id))] = {
            "synthetic_count": int(synth_idx.sum()),
            "mean_same_class_cosine_distance": float(np.mean(same)),
            "mean_nearest_other_class_cosine_distance": float(np.mean(other)),
            "mean_margin_other_minus_same": float(np.mean(margin)),
            "bad_margin_fraction": float(np.mean(margin <= 0.0)),
        }

    if same_distances:
        result.update(
            {
                "mean_same_class_cosine_distance": float(np.mean(same_distances)),
                "mean_nearest_other_class_cosine_distance": float(np.mean(other_distances)),
                "mean_margin_other_minus_same": float(np.mean(margins)),
                "bad_margin_fraction": float(np.mean(np.asarray(margins) <= 0.0)),
                "per_class": per_class,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.resolved.yaml", overrides=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_dataloaders(config)

    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device)
    model = create_model(config, num_classes=len(bundle.class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model"])

    payload = collect_embeddings(model, bundle.loaders[args.split], device)
    features = payload["features"]
    targets = payload["targets"]
    is_synthetic = payload["is_synthetic"]

    diagnostics: dict[str, Any] = {
        "split": args.split,
        "num_examples": int(len(targets)),
        "num_real": int((is_synthetic == 0).sum()),
        "num_synthetic": int((is_synthetic == 1).sum()),
        "real_vs_synthetic_auc": real_vs_synthetic_auc(features, is_synthetic),
    }
    diagnostics.update(nearest_neighbor_diagnostics(features, targets, is_synthetic))
    write_json(run_dir / f"embedding_diagnostics_{args.split}.json", diagnostics)

    index = pd.DataFrame(
        {
            "path": payload["paths"],
            "label": payload["labels"],
            "target": targets,
            "is_synthetic": is_synthetic,
        }
    )
    index.to_csv(run_dir / f"embedding_index_{args.split}.csv", index=False)
    if args.save_embeddings:
        np.savez_compressed(run_dir / f"embeddings_{args.split}.npz", features=features, targets=targets, is_synthetic=is_synthetic)
    print(diagnostics)


if __name__ == "__main__":
    main()
