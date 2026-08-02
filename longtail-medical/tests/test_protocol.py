import unittest

from tools.prepare_isic2019_monica import lesion_overlap, official_label


class ProtocolTest(unittest.TestCase):
    def test_lesion_overlap_counts_images_and_groups(self):
        left = [{"lesion_id": "L1"}, {"lesion_id": "L1"}, {"lesion_id": "L2"}]
        right = [{"lesion_id": "L1"}, {"lesion_id": "L3"}]
        self.assertEqual(
            lesion_overlap(left, right),
            {
                "shared_lesion_ids": 1,
                "images_in_a_with_shared_lesion": 2,
                "images_in_b_with_shared_lesion": 1,
            },
        )

    def test_empty_lesion_id_is_not_a_group(self):
        self.assertEqual(lesion_overlap([{"lesion_id": ""}], [{"lesion_id": ""}])["shared_lesion_ids"], 0)

    def test_official_class_order(self):
        row = {name: "0.0" for name in ["NV", "MEL", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]}
        row["SCC"] = "1.0"
        self.assertEqual(official_label(row), 5)


if __name__ == "__main__":
    unittest.main()
