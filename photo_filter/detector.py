"""Measure persistent vertical streaking without a neural runtime or GPU."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

DETECTOR_VERSION = "row-repetition-v3"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class DecodeError(ValueError):
    pass


@dataclass(frozen=True)
class Detection:
    score: float
    onset_fraction: float | None
    width: int
    height: int


def require_decoder():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required. Install ffmpeg and ensure it is on PATH.")


def decode(path: Path) -> np.ndarray:
    # Keep native vertical resolution: thin bottom strips must survive resizing.
    # One process/thread per image; no hardware decoding or GPU initialization.
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-hwaccel", "none",
             "-threads", "1", "-i", str(path.resolve()), "-vf", "scale=512:ih",
             "-filter_threads", "1", "-frames:v", "1", "-c:v", "pgm",
             "-threads", "1", "-f", "image2pipe", "-"],
            capture_output=True, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DecodeError("decoder timed out after 30 seconds") from exc
    if result.returncode:
        raise DecodeError(result.stderr.decode(errors="replace").strip()[-1000:] or "decoder failed")
    match = re.match(rb"P5\s+(\d+)\s+(\d+)\s+255\s", result.stdout)
    if match is None:
        raise DecodeError("decoder returned no grayscale image")
    width, height = map(int, match.groups())
    pixels = result.stdout[match.end():]
    if len(pixels) != width * height or height < 16:
        raise DecodeError("invalid decoded image dimensions or data")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)


def repetition_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ratios across 8-row bands; flat sky is excluded by a texture floor."""
    a = gray.astype(np.float32)
    starts = np.arange(0, len(a) - 7, 4)
    values = []
    for y in starts:
        band = a[y:y + 8]
        segments = np.array_split(band, 4, axis=1)
        ratios = []
        for segment in segments:
            dx = float(np.abs(np.diff(segment, axis=1)).mean())
            dy = float(np.abs(np.diff(segment, axis=0)).mean())
            ratios.append(dx / (dy + 0.025) if dx >= 0.1 else 0.0)
        values.append(float(np.median(ratios)))
    values = np.asarray(values)
    # Require three overlapping windows (16 rows), suppressing isolated edges.
    persistent = np.minimum.reduce([values[:-2], values[1:-1], values[2:]])
    return starts[:len(persistent)], persistent


def detect(path: Path, onset_threshold: float | None = None) -> Detection:
    gray = decode(path)
    starts, profile = repetition_profile(gray)
    score = float(profile.max())
    # Decoder concealment can replace the lower image with nearly uniform gray.
    # Exclude black/white letterboxing; this is a separate observed defect family.
    bottom = gray[-max(16, len(gray) // 20):].astype(np.float32)
    uniform_region = 8 < float(bottom.mean()) < 247 and float(bottom.std()) < 2.0
    # Neutral decoder gray can contain noise or a narrow intact edge.
    neutral_gray = float((np.abs(bottom - 128) <= 12).mean()) > 0.95
    flat_gray = uniform_region or neutral_gray
    if flat_gray:
        score = max(score, 100.0)
    onset = None
    if onset_threshold is not None:
        hits = np.flatnonzero(profile >= onset_threshold)
        if len(hits):
            onset = float(starts[hits[0]] / len(gray))
        elif flat_gray:
            # A whole-frame collapse has no trustworthy onset boundary.
            onset = None
    return Detection(score, onset, int(gray.shape[1]), int(gray.shape[0]))
