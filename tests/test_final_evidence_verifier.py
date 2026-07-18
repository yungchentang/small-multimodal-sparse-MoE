from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import build_evaluation_result_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


class FinalEvidenceVerifierTest(unittest.TestCase):
    def test_cli_prints_one_standalone_digest_marker(self) -> None:
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            manifest, "build_manifest"
        ), mock.patch.object(manifest, "sha256_file", return_value=digest):
            output = Path(tmp) / "result-manifest.json"
            stream = io.StringIO()
            with redirect_stdout(stream):
                manifest.main(["--spec", "spec.json", "--output", str(output)])

        lines = stream.getvalue().splitlines()
        marker = f"RUNAI_RESULT_MANIFEST_SHA256={digest}"
        self.assertEqual(lines[-1], marker)
        self.assertEqual(lines.count(marker), 1)

    def test_runai_launcher_forwards_fail_closed_paths(self) -> None:
        run_script = (ROOT / "run.sh").read_text(encoding="utf-8")
        submit_script = (ROOT / "scripts" / "submit_runai.sh").read_text(
            encoding="utf-8"
        )
        wrapper = (
            ROOT / "scripts" / "submit_final_evidence_verifier.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("evidence-verifier)", run_script)
        self.assertIn('--spec "${EVIDENCE_MANIFEST_SPEC:?', run_script)
        self.assertIn('--output "${EVIDENCE_MANIFEST_OUTPUT:?', run_script)
        for variable in ("EVIDENCE_MANIFEST_SPEC", "EVIDENCE_MANIFEST_OUTPUT"):
            self.assertIn(
                f'--environment {variable}="${{{variable}:-}}"', submit_script
            )
        self.assertIn('if [ -e "$EVIDENCE_MANIFEST_OUTPUT" ]', wrapper)
        self.assertIn("MODE=evidence-verifier GPU=0", wrapper)


if __name__ == "__main__":
    unittest.main()
