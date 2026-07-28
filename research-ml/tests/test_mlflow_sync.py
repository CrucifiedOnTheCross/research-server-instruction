from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.sync_mlflow_runs import existing_source_dirs, normalize_source_dir


class FakeClient:
    def search_experiments(self):
        return [SimpleNamespace(experiment_id="live"), SimpleNamespace(experiment_id="historical")]

    def search_runs(self, experiment_ids, max_results):
        source = "outputs/example/run_42" if experiment_ids == ["live"] else None
        tags = {"source_run_dir": source} if source else {}
        return [SimpleNamespace(data=SimpleNamespace(tags=tags))]


class MlflowSyncTests(unittest.TestCase):
    def test_relative_and_absolute_source_dirs_share_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                relative = normalize_source_dir("outputs/example/run_42")
                absolute = normalize_source_dir(str(Path(directory) / "outputs/example/run_42"))
            finally:
                os.chdir(previous)
        self.assertEqual(relative, absolute)

    def test_live_experiments_are_included_in_duplicate_detection(self) -> None:
        sources = existing_source_dirs(FakeClient())
        self.assertEqual(sources, {normalize_source_dir("outputs/example/run_42")})


if __name__ == "__main__":
    unittest.main()
