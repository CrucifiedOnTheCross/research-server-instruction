from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.generate_synthetic_img2img import excluded_source_ids, task_image_stem


class GenerationResumeTests(unittest.TestCase):
    def test_task_image_stem_is_deterministic(self) -> None:
        task = {
            "label": "mel",
            "source_id": "ISIC_1",
            "strength": 0.05,
            "index": 0,
            "seed": 123,
        }
        self.assertEqual(
            task_image_stem(task),
            "mel_ISIC_1_s0p05_00_123",
        )

    def test_excluded_sources_are_loaded_from_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "existing.csv"
            manifest.write_text(
                "image_id,source_image_id\nsynthetic_1,ISIC_1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                excluded_source_ids(root, ["existing.csv"]),
                {"ISIC_1"},
            )


if __name__ == "__main__":
    unittest.main()
