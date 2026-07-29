from __future__ import annotations

import numpy as np
from PIL import Image


class CropDarkFieldOfView:
    def __init__(
        self,
        threshold: int = 8,
        margin_fraction: float = 0.02,
        analysis_size: int = 512,
        min_removed_fraction: float = 0.01,
    ) -> None:
        self.threshold = int(threshold)
        self.margin_fraction = float(margin_fraction)
        self.analysis_size = int(analysis_size)
        self.min_removed_fraction = float(min_removed_fraction)
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if not 0 <= self.margin_fraction <= 0.25:
            raise ValueError("margin_fraction must be between 0 and 0.25")
        if self.analysis_size < 32:
            raise ValueError("analysis_size must be at least 32")
        if not 0 <= self.min_removed_fraction <= 0.25:
            raise ValueError("min_removed_fraction must be between 0 and 0.25")

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        analysis = rgb.copy()
        analysis.thumbnail(
            (self.analysis_size, self.analysis_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(analysis)
        foreground = array.mean(axis=2) > self.threshold
        rows, columns = np.nonzero(foreground)
        if rows.size == 0 or columns.size == 0:
            return rgb
        scale_x = rgb.width / analysis.width
        scale_y = rgb.height / analysis.height
        left = round(int(columns.min()) * scale_x)
        top = round(int(rows.min()) * scale_y)
        right = round((int(columns.max()) + 1) * scale_x)
        bottom = round((int(rows.max()) + 1) * scale_y)
        margin = round(max(right - left, bottom - top) * self.margin_fraction)
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(rgb.width, right + margin)
        bottom = min(rgb.height, bottom + margin)
        if left == 0 and top == 0 and right == rgb.width and bottom == rgb.height:
            return rgb
        if (right - left) * (bottom - top) < 0.25 * rgb.width * rgb.height:
            return rgb
        retained_fraction = (
            (right - left) * (bottom - top) / (rgb.width * rgb.height)
        )
        if 1.0 - retained_fraction < self.min_removed_fraction:
            return rgb
        return rgb.crop((left, top, right, bottom))
