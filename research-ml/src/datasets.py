from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms

from .reproducibility import seed_worker


@dataclass
class DatasetBundle:
    loaders: dict[str, DataLoader]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    class_counts: dict[str, dict[str, int]]
    sampling_plan: dict[str, Any]


class ResizePadToSquare:
    def __init__(self, size: int, fill: tuple[int, int, int]) -> None:
        self.size = int(size)
        self.fill = tuple(int(value) for value in fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = (
            (self.size - resized.width) // 2,
            (self.size - resized.height) // 2,
        )
        canvas.paste(resized, offset)
        return canvas


class CsvImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path,
        image_col: str,
        label_col: str,
        synthetic_col: str,
        sample_weight_col: str,
        class_to_idx: dict[str, int],
        transform: Any,
        allow_synthetic: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.image_col = image_col
        self.label_col = label_col
        self.synthetic_col = synthetic_col
        self.sample_weight_col = sample_weight_col
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.frame = pd.read_csv(self.csv_path)
        if synthetic_col not in self.frame.columns:
            self.frame[synthetic_col] = 0
        if sample_weight_col not in self.frame.columns:
            self.frame[sample_weight_col] = 1.0
        if (self.frame[sample_weight_col].astype(float) <= 0).any():
            raise ValueError(f"{self.csv_path}: {sample_weight_col} must contain positive values")
        if not allow_synthetic:
            self.frame = self.frame[self.frame[synthetic_col].astype(int) == 0].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row[self.image_col]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_name = str(row[self.label_col])
        return {
            "image": image,
            "target": torch.tensor(self.class_to_idx[label_name], dtype=torch.long),
            "label_name": label_name,
            "path": str(image_path),
            "is_synthetic": int(row.get(self.synthetic_col, 0)),
            "sample_weight": float(row.get(self.sample_weight_col, 1.0)),
        }


def build_transforms(config: dict[str, Any], train: bool) -> transforms.Compose:
    aug = config["augmentation"]
    data = config["data"]
    mean = aug["normalize"]["mean"]
    std = aug["normalize"]["std"]
    pad_fill = tuple(round(float(value) * 255) for value in mean)
    if train:
        train_aug = aug["train"]
        ops: list[Any] = []
        if train_aug.get("aspect_preserving_pad", False):
            if train_aug["random_resized_crop"]:
                raise ValueError(
                    "aspect_preserving_pad and random_resized_crop are mutually exclusive"
                )
            ops.append(ResizePadToSquare(data["image_size"], pad_fill))
        elif train_aug["random_resized_crop"]:
            ops.append(transforms.RandomResizedCrop(data["image_size"], scale=tuple(train_aug["scale"])))
        else:
            ops.append(transforms.Resize((data["image_size"], data["image_size"])))
        if train_aug["horizontal_flip"] > 0:
            ops.append(transforms.RandomHorizontalFlip(train_aug["horizontal_flip"]))
        if train_aug["vertical_flip"] > 0:
            ops.append(transforms.RandomVerticalFlip(train_aug["vertical_flip"]))
        if train_aug["color_jitter"]:
            ops.append(transforms.ColorJitter(*train_aug["color_jitter"]))
        if train_aug["randaugment"]:
            ops.append(transforms.RandAugment())
    else:
        eval_aug = aug["eval"]
        if eval_aug.get("aspect_preserving_pad", False) and eval_aug["center_crop"]:
            raise ValueError(
                "augmentation.eval.aspect_preserving_pad and center_crop are mutually exclusive"
            )
        if eval_aug["center_crop"] and int(eval_aug["resize"]) < int(data["val_size"]):
            raise ValueError(
                "augmentation.eval.resize must be >= data.val_size when center_crop=true; "
                "otherwise torchvision pads the evaluation image."
            )
        if eval_aug.get("aspect_preserving_pad", False):
            ops = [ResizePadToSquare(data["val_size"], pad_fill)]
        else:
            ops = [transforms.Resize(eval_aug["resize"])]
        if eval_aug["center_crop"]:
            ops.append(transforms.CenterCrop(data["val_size"]))
        elif not eval_aug.get("aspect_preserving_pad", False):
            ops.append(transforms.Resize((data["val_size"], data["val_size"])))

    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def resolve_csv(root: Path, path: str) -> Path:
    csv_path = Path(path)
    if csv_path.is_absolute():
        return csv_path
    return root / csv_path


def infer_class_map(config: dict[str, Any]) -> dict[str, int]:
    data = config["data"]
    root = Path(data["root"])
    if data["class_map"]:
        return {str(k): int(v) for k, v in data["class_map"].items()}

    labels: set[str] = set()
    for key in ("train_csv", "val_csv", "test_csv"):
        csv_path = resolve_csv(root, data[key])
        frame = pd.read_csv(csv_path, usecols=[data["label_col"]])
        labels.update(frame[data["label_col"]].astype(str).unique().tolist())
    return {label: idx for idx, label in enumerate(sorted(labels))}


def count_classes(dataset: CsvImageDataset) -> dict[str, int]:
    counts = dataset.frame[dataset.label_col].astype(str).value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in dataset.class_to_idx}


class ClassBalancedUndersampler(Sampler[int]):
    def __init__(self, labels: np.ndarray, seed: int) -> None:
        self.seed = int(seed)
        self.epoch = 0
        self.class_indices = {
            int(label): np.flatnonzero(labels == label)
            for label in sorted(np.unique(labels).tolist())
        }
        if not self.class_indices or any(len(indices) == 0 for indices in self.class_indices.values()):
            raise ValueError("Undersampling requires at least one sample for every observed class")
        self.samples_per_class = min(len(indices) for indices in self.class_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = [
            rng.choice(indices, size=self.samples_per_class, replace=False)
            for indices in self.class_indices.values()
        ]
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_class * len(self.class_indices)


def dataset_labels(dataset: CsvImageDataset) -> np.ndarray:
    return dataset.frame[dataset.label_col].astype(str).map(dataset.class_to_idx).to_numpy()


def make_weighted_sampler(
    dataset: CsvImageDataset, generator: torch.Generator
) -> WeightedRandomSampler:
    labels = dataset_labels(dataset)
    counts = np.bincount(labels, minlength=len(dataset.class_to_idx))
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def make_sampling_plan(
    dataset: CsvImageDataset,
    sampler_name: str,
    sampler: Sampler[int] | None,
) -> dict[str, Any]:
    counts = count_classes(dataset)
    labels = list(dataset.class_to_idx)
    if sampler_name == "weighted":
        expected = {label: len(dataset) / len(labels) for label in labels}
        replacement = True
    elif sampler_name == "undersample":
        if not isinstance(sampler, ClassBalancedUndersampler):
            raise TypeError("undersample plan requires ClassBalancedUndersampler")
        expected = {label: sampler.samples_per_class for label in labels}
        replacement = False
    else:
        expected = {label: counts[label] for label in labels}
        replacement = False
    return {
        "sampler": sampler_name,
        "replacement": replacement,
        "dataset_rows": len(dataset),
        "samples_per_epoch": len(sampler) if sampler is not None else len(dataset),
        "dataset_class_counts": counts,
        "expected_class_samples_per_epoch": expected,
    }


def make_dataloaders(config: dict[str, Any]) -> DatasetBundle:
    data = config["data"]
    runtime = config["runtime"]
    root = Path(data["root"])
    class_to_idx = infer_class_map(config)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    datasets: dict[str, CsvImageDataset] = {
        "train": CsvImageDataset(
            resolve_csv(root, data["train_csv"]),
            root,
            data["image_col"],
            data["label_col"],
            data["synthetic_col"],
            data["sample_weight_col"],
            class_to_idx,
            build_transforms(config, train=True),
            allow_synthetic=True,
        ),
        "val": CsvImageDataset(
            resolve_csv(root, data["val_csv"]),
            root,
            data["image_col"],
            data["label_col"],
            data["synthetic_col"],
            data["sample_weight_col"],
            class_to_idx,
            build_transforms(config, train=False),
            allow_synthetic=bool(data["allow_synthetic_in_eval"]),
        ),
        "test": CsvImageDataset(
            resolve_csv(root, data["test_csv"]),
            root,
            data["image_col"],
            data["label_col"],
            data["synthetic_col"],
            data["sample_weight_col"],
            class_to_idx,
            build_transforms(config, train=False),
            allow_synthetic=bool(data["allow_synthetic_in_eval"]),
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(int(runtime["seed"]))
    loaders: dict[str, DataLoader] = {}
    train_sampling_plan: dict[str, Any] | None = None
    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        if split == "train":
            sampler_name = str(config["imbalance"]["sampler"])
            if sampler_name == "weighted":
                sampler = make_weighted_sampler(dataset, generator)
                shuffle = False
            elif sampler_name == "undersample":
                sampler = ClassBalancedUndersampler(
                    dataset_labels(dataset),
                    seed=int(runtime["seed"]),
                )
                shuffle = False
            elif sampler_name != "none":
                raise ValueError(f"Unknown sampler: {sampler_name}")
            train_sampling_plan = make_sampling_plan(dataset, sampler_name, sampler)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(runtime["num_workers"]),
            pin_memory=bool(runtime["pin_memory"]),
            prefetch_factor=int(runtime["prefetch_factor"]) if int(runtime["num_workers"]) > 0 else None,
            persistent_workers=bool(runtime["persistent_workers"]) and int(runtime["num_workers"]) > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    class_counts = {split: count_classes(dataset) for split, dataset in datasets.items()}
    if train_sampling_plan is None:
        raise RuntimeError("Training sampling plan was not created")
    return DatasetBundle(
        loaders=loaders,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        class_counts=class_counts,
        sampling_plan=train_sampling_plan,
    )
