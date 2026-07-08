from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.datasets import build_transforms, infer_class_map, resolve_csv
from src.models import create_model


class CsvEvalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        image_col: str,
        label_col: str,
        class_to_idx: dict[str, int],
        transform: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.image_col = image_col
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row[self.image_col]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(self.class_to_idx[str(row[self.label_col])], dtype=torch.long),
            "row_index": index,
            "path": str(image_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline hard cases for Stage 3 synthetic selection.")
    parser.add_argument("--baseline-run-dir", required=True, help="Run directory with best.pt and config.resolved.yaml.")
    parser.add_argument("--out-csv", default="/srv/research/projects/default/ham10000/splits/stage3/hard_cases.csv")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs.clip(min=1e-12))).sum(axis=1)


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    split_name: str,
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    dataset = CsvEvalDataset(
        frame,
        root,
        cfg["data"]["image_col"],
        cfg["data"]["label_col"],
        class_to_idx,
        build_transforms(cfg, train=False),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    indices: list[int] = []
    paths: list[str] = []
    dtype_name = str(cfg["runtime"].get("amp", "none")).lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16 if dtype_name == "fp16" else None
    model.eval()
    for batch in tqdm(loader, desc=f"hard cases {split_name}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if bool(cfg["runtime"].get("channels_last", False)) and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            logits = model(images)
        logits_all.append(logits.detach().float().cpu().numpy())
        targets_all.append(batch["target"].numpy())
        indices.extend([int(x) for x in batch["row_index"]])
        paths.extend(batch["path"])

    logits_np = np.concatenate(logits_all)
    targets_np = np.concatenate(targets_all)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    losses = F.cross_entropy(torch.from_numpy(logits_np), torch.from_numpy(targets_np), reduction="none").numpy()

    rows = frame.iloc[indices].copy().reset_index(drop=True)
    rows["split"] = split_name
    rows["abs_path"] = paths
    rows["target"] = [idx_to_class[int(x)] for x in targets_np]
    rows["prediction"] = [idx_to_class[int(x)] for x in top1]
    rows["is_correct"] = (top1 == targets_np).astype(int)
    rows["confidence"] = probs[np.arange(len(probs)), top1]
    rows["target_prob"] = probs[np.arange(len(probs)), targets_np]
    rows["entropy"] = entropy(probs)
    rows["top1_prob"] = probs[np.arange(len(probs)), top1]
    rows["top2_prob"] = probs[np.arange(len(probs)), top2]
    rows["top1_top2_margin"] = rows["top1_prob"] - rows["top2_prob"]
    rows["loss"] = losses
    rows["hard_score"] = rows["entropy"] + (1.0 - rows["target_prob"]) + rows["loss"]
    return rows


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = infer_class_map(cfg)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    model = create_model(cfg, len(class_to_idx)).to(device)
    if bool(cfg["runtime"].get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    root = Path(cfg["data"]["root"])
    split_to_csv = {
        "train": cfg["data"]["train_csv"],
        "val": cfg["data"]["val_csv"],
        "test": cfg["data"]["test_csv"],
    }
    pieces = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        frame = pd.read_csv(resolve_csv(root, split_to_csv[split]))
        if cfg["data"]["synthetic_col"] in frame.columns:
            frame = frame[frame[cfg["data"]["synthetic_col"]].astype(int) == 0].copy()
        pieces.append(predict_split(model, frame, split, cfg, class_to_idx, idx_to_class, device, args.batch_size, args.num_workers))

    result = pd.concat(pieces, ignore_index=True)
    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}
    result["is_target_stage3_class"] = result["target"].isin(target_classes).astype(int)
    result = result.sort_values(["split", "target", "hard_score"], ascending=[True, True, False])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    summary = {
        "baseline_run_dir": str(run_dir),
        "out_csv": str(out_csv),
        "splits": sorted(result["split"].unique().tolist()),
        "rows": int(len(result)),
        "target_classes": sorted(target_classes),
        "by_split": result["split"].value_counts().to_dict(),
        "by_target": result[result["is_target_stage3_class"] == 1]["target"].value_counts().to_dict(),
    }
    (out_csv.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
