from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_evaluation_protocol import git_state


class FreezeGitStateTests(unittest.TestCase):
    def test_untracked_file_content_changes_frozen_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "initial",
                ],
                check=True,
            )
            untracked = repo / "source.py"
            untracked.write_text("value = 1\n", encoding="utf-8")
            before = git_state(repo)
            untracked.write_text("value = 2\n", encoding="utf-8")
            after = git_state(repo)
            self.assertEqual(before["dirty_diff_sha256"], after["dirty_diff_sha256"])
            self.assertNotEqual(
                before["untracked_manifest_sha256"],
                after["untracked_manifest_sha256"],
            )
            self.assertEqual(before["untracked_files"][0]["path"], "source.py")


if __name__ == "__main__":
    unittest.main()
