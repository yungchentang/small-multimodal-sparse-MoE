import unittest

from scripts.collect_runai_evidence import safe_job_name


class RunAIEvidenceTest(unittest.TestCase):
    def test_accepts_runai_names(self):
        self.assertEqual(safe_job_name("sme-match3-s42-260709"), "sme-match3-s42-260709")

    def test_rejects_shell_metacharacters(self):
        with self.assertRaises(ValueError):
            safe_job_name("job;rm")


if __name__ == "__main__":
    unittest.main()
