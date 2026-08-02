import unittest

from tools.build_stage2_protocols import select_exact_groups


class Stage2ProtocolTest(unittest.TestCase):
    def test_exact_group_selection_never_splits_group(self):
        groups = [
            ("a", [{"image": "a1"}, {"image": "a2"}]),
            ("b", [{"image": "b1"}]),
            ("c", [{"image": "c1"}, {"image": "c2"}, {"image": "c3"}]),
        ]
        selected, remaining = select_exact_groups(groups, 3)
        selected_ids = {row["image"] for row in selected}
        remaining_ids = {row["image"] for _, rows in remaining for row in rows}
        self.assertEqual(len(selected), 3)
        self.assertFalse(selected_ids & remaining_ids)
        for _, rows in groups:
            ids = {row["image"] for row in rows}
            self.assertTrue(ids <= selected_ids or ids <= remaining_ids)

    def test_impossible_exact_selection_fails(self):
        groups = [("a", [{}, {}]), ("b", [{}, {}])]
        with self.assertRaises(RuntimeError):
            select_exact_groups(groups, 3)


if __name__ == "__main__":
    unittest.main()
