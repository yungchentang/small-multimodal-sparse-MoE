from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FILES = (
    REPO_ROOT / "model" / "olmoe_adapter.py",
    REPO_ROOT / "model" / "encoders.py",
    REPO_ROOT / "scripts" / "env_check.py",
)


class AllHuggingFaceModelLoadsPinnedTests(unittest.TestCase):
    def test_legacy_production_files_have_no_direct_from_pretrained_calls(self) -> None:
        violations = []
        for path in PRODUCTION_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                ):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
