"""Guard canonical training entry points against unpinned HF loading."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FILES = (
    Path("training/olmoe_required_runs.py"),
    Path("training/olmoe_real_subset_runs.py"),
    Path("training/olmoe_short_run.py"),
    Path("scripts/diagnose_e0_sanity.py"),
)


def parse(relative_path: Path) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


class CanonicalHuggingFaceLoadingTests(unittest.TestCase):
    def test_canonical_files_have_no_direct_from_pretrained_calls(self) -> None:
        direct_calls = []
        for relative_path in CANONICAL_FILES:
            for node in ast.walk(parse(relative_path)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                ):
                    direct_calls.append(f"{relative_path}:{node.lineno}")

        self.assertEqual(
            direct_calls,
            [],
            "Canonical files must load Hugging Face sources through hf_sources.load_pretrained",
        )

    def test_canonical_files_use_pinned_loading_helper(self) -> None:
        for relative_path in CANONICAL_FILES:
            tree = parse(relative_path)
            imports_helper = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "hf_sources"
                and any(alias.name == "load_pretrained" for alias in node.names)
                for node in tree.body
            )
            calls_helper = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "load_pretrained"
                for node in ast.walk(tree)
            )

            with self.subTest(path=str(relative_path)):
                self.assertTrue(imports_helper)
                self.assertTrue(calls_helper)


if __name__ == "__main__":
    unittest.main()
