from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.analyze_stage10_results import latest_run_by_seed


class AnalysisRunDiscoveryTest(unittest.TestCase):
    def test_ignores_non_numeric_diagnostic_suffixes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "20260728-104537_42"
            newer = root / "20260728-110000_42"
            diagnostic = root / "20260728-045231_42_invalid_ema_reload"
            for path in (valid, newer, diagnostic):
                path.mkdir()

            self.assertEqual(latest_run_by_seed(root), {42: newer})


if __name__ == "__main__":
    unittest.main()
