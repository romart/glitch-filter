import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from photo_filter.detector import DecodeError, Detection, detect, repetition_profile
from photo_filter.workflow import (assign_splits, calibrate, copy_exclusive, decide,
                                   discover, sort_images)


MODEL = {"good_threshold": 3.0, "bad_threshold": 8.0, "model_id": "test"}


class DetectionTests(unittest.TestCase):
    def test_streaks_have_stronger_scores_than_noise_or_flat_images(self):
        rng = np.random.default_rng(42)
        natural = rng.integers(0, 256, (128, 512), dtype=np.uint8)
        glitched = natural.copy()
        glitched[-24:] = glitched[-24]
        self.assertGreater(repetition_profile(glitched)[1].max(), 20)
        self.assertLess(repetition_profile(natural)[1].max(), 3)
        self.assertEqual(repetition_profile(np.zeros_like(natural))[1].max(), 0)

    def test_single_horizontal_boundary_is_not_a_streak(self):
        image = np.zeros((128, 512), dtype=np.uint8)
        image[64:] = 255
        self.assertEqual(repetition_profile(image)[1].max(), 0)

    def test_uniform_gray_collapse_is_detected(self):
        image = np.full((128, 512), 128, dtype=np.uint8)
        with patch("photo_filter.detector.decode", return_value=image):
            self.assertGreaterEqual(detect(Path("gray.jpg"), 8).score, 100)

    def test_black_letterbox_is_not_gray_collapse(self):
        image = np.zeros((128, 512), dtype=np.uint8)
        with patch("photo_filter.detector.decode", return_value=image):
            self.assertEqual(detect(Path("black.jpg"), 8).score, 0)

    def test_gray_collapse_with_narrow_edge_or_noise_is_detected(self):
        rng = np.random.default_rng(17)
        noisy = np.clip(rng.normal(128, 4, (128, 512)), 0, 255).astype(np.uint8)
        edged = np.full((128, 512), 128, dtype=np.uint8)
        edged[:, :8] = 255
        for image in [noisy, edged]:
            with patch("photo_filter.detector.decode", return_value=image):
                self.assertGreaterEqual(detect(Path("gray.jpg"), 8).score, 100)


class CalibrationTests(unittest.TestCase):
    def test_thresholds_ignore_test_labels_and_scores(self):
        rows = [{"split": "validation", "label": label, "score": score, "error": None}
                for label, score in [("good", 1), ("good", 2), ("bad", 10), ("bad", 12)]]
        manifest = {"samples": rows + [{"split": "test", "label": "good", "score": 999, "error": None}]}
        model = calibrate(manifest)
        self.assertGreater(model["good_threshold"], 2)
        self.assertLess(model["bad_threshold"], 10)
        self.assertEqual(decide(None, model), "unreadable")
        self.assertEqual(decide(model["good_threshold"], model), "review")
        self.assertEqual(decide(model["bad_threshold"], model), "bad")
        manifest["samples"][-1]["score"] = -999
        other = calibrate(manifest)
        self.assertEqual(other["good_threshold"], model["good_threshold"])
        self.assertEqual(other["bad_threshold"], model["bad_threshold"])

    def test_overlapping_scores_are_sent_to_review(self):
        manifest = {"samples": [{"split": "validation", "label": label, "score": score, "error": None}
                                 for label, score in [("good", 5), ("bad", 3)]]}
        model = calibrate(manifest)
        self.assertEqual(decide(3, model), "review")
        self.assertEqual(decide(5, model), "review")

    def test_day_and_duplicate_groups_never_cross_splits(self):
        rows = [{"path": f"{label}/{day}/{i}", "label": label, "group": f"day{day}",
                 "sha256": f"{label}-{day}-{i}"}
                for label, start in [("good", 0), ("bad", 10)]
                for day in range(start, start + 6) for i in range(2)]
        rows[2]["sha256"] = rows[0]["sha256"]
        assign_splits(rows, 42)
        for key in ("group", "sha256"):
            groups = {}
            for row in rows:
                groups.setdefault(row[key], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_conflicting_duplicate_labels_are_rejected(self):
        rows = [{"path": label, "label": label, "group": label, "sha256": "same"} for label in ["good", "bad"]]
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            assign_splits(rows, 42)


class SortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.original = self.source / "day" / "photo.jpg"
        self.original.parent.mkdir()
        self.original.write_bytes(b"original-photo-bytes")
        self.output = self.source / "results"
        self.patcher = patch("photo_filter.workflow.detect", return_value=Detection(10, 0.5, 512, 1080))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_dry_run_writes_nothing_even_with_explicit_report_and_move(self):
        report = self.root / "reports" / "run.jsonl"
        before = sorted(str(p) for p in self.root.rglob("*"))
        rows = sort_images(self.source, self.output, MODEL, dry_run=True, move=True, report=report)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)
        self.assertEqual(rows[0]["status"], "planned")

    def test_copy_preserves_original_and_report_and_excludes_output(self):
        rows = sort_images(self.source, self.output, MODEL)
        destination = self.output / "bad/day/photo.jpg"
        self.assertEqual(destination.read_bytes(), self.original.read_bytes())
        self.assertEqual(rows[0]["status"], "copied")
        self.assertEqual(json.loads((self.output / "report.jsonl").read_text())["status"], "copied")
        self.assertEqual(list(discover(self.source, exclude=self.output)), [self.original])

    def test_collision_aborts_before_any_photo_is_copied(self):
        collision = self.output / "bad/day/photo.jpg"
        collision.parent.mkdir(parents=True)
        collision.write_bytes(b"existing")
        (self.source / "another.jpg").write_bytes(b"another")
        with self.assertRaises(FileExistsError):
            sort_images(self.source, self.output, MODEL)
        self.assertEqual(collision.read_bytes(), b"existing")
        self.assertFalse((self.output / "bad/another.jpg").exists())
        self.assertFalse((self.output / "report.jsonl").exists())

    def test_move_removes_original_only_after_success(self):
        rows = sort_images(self.source, self.output, MODEL, move=True)
        self.assertFalse(self.original.exists())
        self.assertEqual(Path(rows[0]["destination"]).read_bytes(), b"original-photo-bytes")
        self.assertEqual(rows[0]["status"], "moved")

    def test_failed_copy_leaves_source_and_records_failure(self):
        with patch("photo_filter.workflow.copy_exclusive", side_effect=OSError("disk full")):
            rows = sort_images(self.source, self.output, MODEL, move=True)
        self.assertTrue(self.original.exists())
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("disk full", (self.output / "report.jsonl").read_text())

    def test_unreadable_file_is_quarantined_with_error(self):
        with patch("photo_filter.workflow.detect", side_effect=DecodeError("broken jpeg")):
            rows = sort_images(self.source, self.output, MODEL)
        self.assertEqual(rows[0]["decision"], "unreadable")
        self.assertEqual(rows[0]["error"], "broken jpeg")
        self.assertTrue(self.original.exists())

    def test_ancestor_output_is_rejected(self):
        with self.assertRaises(ValueError):
            sort_images(self.source, self.root, MODEL)

    def test_output_symlink_escape_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.output.mkdir()
        (self.output / "bad").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes output"):
            sort_images(self.source, self.output, MODEL)
        self.assertEqual(list(outside.iterdir()), [])

    def test_exclusive_copy_never_overwrites_existing_file(self):
        destination = self.root / "exists.jpg"
        destination.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            copy_exclusive(self.original, destination)
        self.assertEqual(destination.read_bytes(), b"keep me")


if __name__ == "__main__":
    unittest.main()
