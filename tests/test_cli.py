from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from photo_filter.cli import DEFAULT_MODEL, main
from photo_filter.detector import Detection
from photo_filter.workflow import load_model


class BundledDetectorTests(unittest.TestCase):
    def test_default_detector_dry_run_and_normal_output_creation(self):
        model = load_model(DEFAULT_MODEL)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "sorted"
            source.mkdir()
            original = source / "photo.jpg"
            original.write_bytes(b"fixture bytes")

            def inspect(*args):
                self.assertFalse(output.exists(), "Output must be created after scoring")
                return Detection(100, 0.5, 512, 1080)

            for dry_run in [True, False]:
                stdout, stderr = io.StringIO(), io.StringIO()
                command = ["sort", str(source), "--output", str(output)]
                if dry_run:
                    command.append("--dry-run")
                with redirect_stdout(stdout), redirect_stderr(stderr), \
                     patch("photo_filter.cli.require_decoder"), \
                     patch("photo_filter.workflow.detect", side_effect=inspect):
                    result = main(command)
                self.assertEqual(result, 0)
                summary = json.loads(stderr.getvalue().splitlines()[-1])
                self.assertEqual(summary["dry_run"], dry_run)
                self.assertEqual(summary["output"], str(output))
                if dry_run:
                    self.assertFalse(output.exists())
                    self.assertIn("no files or output directory", stderr.getvalue())
                    self.assertEqual(json.loads(stdout.getvalue())["model_id"], model["model_id"])
                else:
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn("after scoring and collision checks", stderr.getvalue())
                    self.assertEqual((output / "bad/photo.jpg").read_bytes(), original.read_bytes())
                    record = json.loads((output / "report.jsonl").read_text())
                    self.assertEqual(record["model_id"], model["model_id"])

    def test_empty_input_explains_error_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "sorted"
            source.mkdir()
            stderr = io.StringIO()
            with redirect_stderr(stderr), patch("photo_filter.cli.require_decoder"):
                result = main(["sort", str(source), "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertIn("No supported images found", stderr.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
