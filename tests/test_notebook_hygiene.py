"""Exercise the publication guard against synthetic notebook data exposure."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(shutil.which("git"), "Notebook publication checks require Git")
class NotebookHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "scripts" / "check_notebooks.py",
            self.root / "scripts" / "check_notebooks.py",
        )
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [{
                "cell_type": "code", "metadata": {}, "source": ["1 + 1"],
                "outputs": [], "execution_count": None,
            }],
        }

    def run_guard(self, *, tracked=False):
        (self.root / "example.ipynb").write_text(json.dumps(self.notebook), encoding="utf-8")
        if tracked:
            subprocess.run(["git", "add", "example.ipynb"], cwd=self.root, check=True)
        return subprocess.run(
            [sys.executable, "scripts/check_notebooks.py"], cwd=self.root,
            capture_output=True, text=True,
        )

    def test_source_only_notebook_passes(self):
        result = self.run_guard(tracked=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_untracked_notebook_with_image_output_is_rejected(self):
        self.notebook["cells"][0]["outputs"] = [{
            "output_type": "display_data", "data": {"image/png": "synthetic"}, "metadata": {},
        }]
        result = self.run_guard()
        self.assertEqual(result.returncode, 1)
        self.assertIn("execution results", result.stderr)

    def test_identity_metadata_is_rejected(self):
        for scope in [self.notebook, self.notebook["cells"][0]]:
            with self.subTest(scope="notebook" if scope is self.notebook else "cell"):
                scope["metadata"] = {"executionInfo": {"userId": "synthetic-user"}}
                result = self.run_guard(tracked=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("metadata", result.stderr)
                scope["metadata"] = {}

    def test_markdown_attachment_is_rejected(self):
        self.notebook["cells"] = [{
            "cell_type": "markdown", "metadata": {}, "source": ["example"],
            "attachments": {"image.png": {"image/png": "synthetic"}},
        }]
        result = self.run_guard(tracked=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("attachments", result.stderr)


if __name__ == "__main__":
    unittest.main()
