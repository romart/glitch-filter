from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from photo_filter.cli import main
from photo_filter.detector import Detection
from photo_filter.progress import Progress


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_updates_are_throttled_with_rate_eta_and_final_count(self):
        stream = io.StringIO()
        now = [0.0]
        with Progress("Scoring", 4, stream=stream, clock=lambda: now[0]) as tracker:
            initial = stream.getvalue()
            now[0] = 0.25
            tracker.advance()
            self.assertEqual(stream.getvalue(), initial)
            now[0] = 1.0
            tracker.advance()
            self.assertIn("2/4 (50.0%) | 2.0 img/s | elapsed 0:01 | ETA 0:01", stream.getvalue())
            now[0] = 2.0
            tracker.advance()
            tracker.advance()
        final = stream.getvalue().splitlines()[-1]
        self.assertIn("4/4 (100.0%)", final)
        self.assertTrue(final.endswith("done"))
        self.assertNotIn("\r", stream.getvalue())

    def test_interruption_finishes_terminal_line_without_claiming_completion(self):
        stream = Terminal()
        with self.assertRaises(KeyboardInterrupt):
            with Progress("Scoring", 4, stream=stream) as tracker:
                tracker.advance()
                raise KeyboardInterrupt()
        self.assertTrue(stream.getvalue().endswith("stopped\n"))
        self.assertIn("1/4 (25.0%)", stream.getvalue())
        self.assertNotIn("100.0%", stream.getvalue())


class CLIProgressTests(unittest.TestCase):
    def test_dry_run_progress_is_stderr_only_and_can_be_disabled(self):
        model = {"good_threshold": 3, "bad_threshold": 8, "model_id": "test"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            (source / "photo.jpg").write_bytes(b"fixture")
            output = root / "output"
            for option in [[], ["--progress", "always"], ["--no-progress"]]:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr), \
                     patch("photo_filter.cli.require_decoder"), \
                     patch("photo_filter.cli.load_model", return_value=model), \
                     patch("photo_filter.workflow.detect", return_value=Detection(10, 0.5, 512, 1080)):
                    result = main(["sort", str(source), "--output", str(output),
                                   "--model", "unused.json", "--dry-run", *option])
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(stdout.getvalue())["decision"], "bad")
                self.assertFalse(output.exists())
                self.assertEqual("Scoring:" in stderr.getvalue(), option == ["--progress", "always"])
                self.assertNotIn("Copying:", stderr.getvalue())

    def test_audit_progress_preserves_json_stdout_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label, month in [("good", 1), ("bad", 2)]:
                (root / label).mkdir()
                for day in range(1, 4):
                    (root / label / f"camera_2025{month:02}{day:02}.jpg").write_bytes(f"{label}-{day}".encode())
            output = root / "manifest.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), \
                 patch("photo_filter.cli.require_decoder"), \
                 patch("photo_filter.workflow.detect", return_value=Detection(10, None, 512, 1080)):
                result = main(["audit", str(root), "--output", str(output), "--progress", "always"])
            self.assertEqual(result, 0)
            self.assertEqual(sum(json.loads(stdout.getvalue()).values()), 6)
            self.assertEqual(len(json.loads(output.read_text())["samples"]), 6)
            self.assertIn("Indexing:", stderr.getvalue())
            self.assertIn("6/6 (100.0%)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
