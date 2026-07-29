from __future__ import annotations

import unittest

from PIL import Image

from src.datasets import ResizePadToSquare


class Stage14PProtocolTest(unittest.TestCase):
    def test_resize_pad_preserves_full_frame_and_aspect_ratio(self) -> None:
        image = Image.new("RGB", (600, 450), (200, 10, 20))
        image.putpixel((0, 225), (255, 0, 0))
        image.putpixel((599, 225), (0, 255, 0))
        transformed = ResizePadToSquare(384, (124, 116, 104))(image)
        self.assertEqual(transformed.size, (384, 384))
        self.assertEqual(transformed.getpixel((10, 10)), (124, 116, 104))
        self.assertGreater(transformed.getpixel((0, 192))[0], 200)
        self.assertGreater(transformed.getpixel((383, 192))[1], 200)


if __name__ == "__main__":
    unittest.main()
