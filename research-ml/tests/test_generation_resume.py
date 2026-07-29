from __future__ import annotations

import unittest

from tools.generate_synthetic_img2img import task_image_stem


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


if __name__ == "__main__":
    unittest.main()
