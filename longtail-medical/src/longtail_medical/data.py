from __future__ import annotations

import csv
from pathlib import Path


def build_transforms(config: dict, train: bool):
    from torchvision import transforms

    size = int(config["data"]["image_size"])
    resize = int(config["data"]["resize_size"])
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        jitter = config["augmentation"]["train"]["color_jitter"]
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(config["augmentation"]["train"]["rotation_degrees"]),
            transforms.ColorJitter(*jitter),
            transforms.ToTensor(),
            normalization,
        ])
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalization,
    ])


class ManifestDataset:
    def __init__(self, manifest: str | Path, transform=None):
        self.manifest = Path(manifest)
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"image_id", "image_path", "label", "lesion_id", "split"}
        if not self.rows or not required.issubset(self.rows[0]):
            raise ValueError(f"Invalid or empty manifest: {self.manifest}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"]), row["image_id"], row["lesion_id"]
