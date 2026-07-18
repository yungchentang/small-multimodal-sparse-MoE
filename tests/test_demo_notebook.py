import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"


class DemoNotebookStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_sources = [
            "".join(cell["source"])
            for cell in cls.cells
            if cell["cell_type"] == "code"
        ]
        cls.all_source = chr(10).join(
            "".join(cell["source"]) for cell in cls.cells
        )

    def test_notebook_has_minimal_valid_structure(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["nbformat_minor"], 5)
        self.assertEqual(
            set(self.notebook["metadata"]), {"kernelspec", "language_info"}
        )
        self.assertTrue(self.cells)
        self.assertTrue(all(cell.get("metadata") == {} for cell in self.cells))

    def test_notebook_has_no_saved_outputs(self):
        for cell in self.cells:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_avoids_disallowed_evidence_references(self):
        source = self.all_source.lower()
        disallowed = ("smoke", "synthetic", "sealed")
        for fragment in disallowed:
            self.assertNotIn(fragment, source)

    def test_environment_configuration_is_explicit(self):
        self.assertIn(
            'os.environ.get("ACDL_RUN_ROOT", "outputs/final_selected")',
            self.all_source,
        )
        self.assertIn('os.environ["ACDL_DATA_DIR"]', self.all_source)
        self.assertNotIn("sparse-moe-mm-rank49", self.all_source)
        self.assertNotIn("matplotlib", self.all_source)

    def test_required_walkthrough_cells_are_present(self):
        required_code_fragments = (
            ("run_manifest", "data_manifest", 'run_manifest["splits"]'),
            (
                "E0_top8_teacher_baseline/metrics.json",
                "E1_hard_top2/metrics.json",
                "E2_calibrated_top2/metrics.json",
                "E3_final_multimodal_top2/metrics.json",
            ),
            (
                "E3_final_multimodal_top2_text_eval",
                "source_checkpoint_size_bytes",
                "CHECKPOINT_PATH.stat()",
            ),
            (
                "_repr_svg_",
                "<svg",
                "perplexity",
                "capacity_overflow_ratio_mean",
                "inactive_expert_ratio_mean",
            ),
            (
                "MODE=conditional-eval",
                "EVAL_PATH=shared_prefix",
                "PREFIX_CONTROL=real",
                "PER_QUERY_OUTPUT",
            ),
        )
        for fragments in required_code_fragments:
            self.assertTrue(
                any(
                    all(fragment in source for fragment in fragments)
                    for source in self.code_sources
                ),
                f"No code cell contains all required fragments: {fragments}",
            )

    def test_conditional_eval_execution_is_explicitly_opt_in(self):
        self.assertIn(
            'os.environ.get("ACDL_RUN_INFERENCE", "0") == "1"',
            self.all_source,
        )
        self.assertIn("subprocess.run(", self.all_source)
        self.assertIn("check=True", self.all_source)

    def test_conditional_eval_outputs_are_explained(self):
        markdown_sources = [
            "".join(cell["source"])
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        ]
        self.assertTrue(
            any(
                "Development-only" in source
                and "metrics.json" in source
                and "per_query.jsonl" in source
                for source in markdown_sources
            )
        )


if __name__ == "__main__":
    unittest.main()
