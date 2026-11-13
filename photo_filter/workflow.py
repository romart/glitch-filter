"""Dataset auditing, calibration, evaluation, and collision-safe sorting."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile

from .detector import DETECTOR_VERSION, EXTENSIONS, DecodeError, detect

SCHEMA_VERSION = 1


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement leaves an existing report intact on failure.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def discover(root: Path, exclude: Path | None = None):
    root = root.resolve()
    # Do not follow symlinks outside the requested input tree.
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink()
                         and (exclude is None or not (Path(directory) / d).resolve().is_relative_to(exclude)))
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink() and path.suffix.lower() in EXTENSIONS:
                yield path


def capture_group(path: Path) -> str:
    match = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", path.stem)
    if match:
        try:
            return datetime(*map(int, match.groups())).strftime("%Y%m%d")
        except ValueError:
            pass
    # Unknown dates are conservatively one session, rather than per-file splits.
    return "unknown-session"


def assign_splits(rows, seed):
    # Union days linked by exact duplicates before assigning any split.
    parents = {r["group"]: r["group"] for r in rows}

    def find(group):
        while parents[group] != group:
            group = parents[group]
        return group

    hashes = {}
    for row in rows:
        previous = hashes.get(row["sha256"])
        if previous:
            if previous["label"] != row["label"]:
                raise ValueError(f"Conflicting labels for identical files: {previous['path']} and {row['path']}")
            parents[find(row["group"])] = find(previous["group"])
        hashes[row["sha256"]] = row
    for row in rows:
        row["group"] = find(row["group"])
    assignments = {}
    rng = random.Random(seed)
    for label in ("good", "bad"):
        groups = sorted({r["group"] for r in rows if r["label"] == label})
        if len(groups) < 3:
            raise ValueError(f"{label}: need at least three independent capture days/sessions for train/validation/test")
        rng.shuffle(groups)
        count = max(1, round(len(groups) * 0.2))
        proposed = {g: "train" for g in groups}
        proposed.update({g: "validation" for g in groups[:count]})
        proposed.update({g: "test" for g in groups[count:2 * count]})
        for group in groups:
            assignments.setdefault(group, proposed[group])
    for row in rows:
        row["split"] = assignments[row["group"]]
    for label in ("good", "bad"):
        for split in ("train", "validation", "test"):
            if not any(r["label"] == label and r["split"] == split for r in rows):
                raise ValueError(f"Grouped split has no {label} {split} samples; choose another seed or add sessions")


def audit(dataset: Path, seed: int, workers: int):
    rows = []
    for label in ("good", "bad"):
        if not (dataset / label).is_dir():
            raise ValueError(f"Missing labeled directory: {dataset / label}")
        for path in discover(dataset / label):
            rows.append({"path": path.relative_to(dataset.resolve()).as_posix(), "label": label,
                         "group": capture_group(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "bytes": path.stat().st_size})
    if not rows:
        raise ValueError("No supported images found")
    assign_splits(rows, seed)

    def inspect(row):
        try:
            return {**row, **asdict(detect(dataset / row["path"])), "error": None}
        except DecodeError as exc:
            return {**row, "score": None, "error": str(exc)}

    inspected = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(inspect, rows), 1):
            inspected.append(row)
            if i % 100 == 0 or i == len(rows):
                print(f"Audited {i}/{len(rows)}", file=sys.stderr)
    return {"schema_version": SCHEMA_VERSION, "detector_version": DETECTOR_VERSION,
            "dataset": str(dataset.resolve()), "seed": seed,
            "split_strategy": "seeded capture-day groups; exact duplicates linked; approximate 60/20/20 by group",
            "samples": inspected}


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("detector_version") != DETECTOR_VERSION:
        raise ValueError("Manifest version is incompatible; run audit again")
    return manifest


def calibrate(manifest):
    validation = [r for r in manifest["samples"] if r["split"] == "validation" and not r["error"]]
    good = [r["score"] for r in validation if r["label"] == "good"]
    bad = [r["score"] for r in validation if r["label"] == "bad"]
    if not good or not bad:
        raise ValueError("Calibration requires readable validation samples in both classes")
    upper_good, lower_bad = max(good), min(bad)
    if upper_good < lower_bad:
        gap = lower_bad - upper_good
        # Fixed relative margins above observed good scores avoid a sparse bad
        # validation set pushing rejection thresholds unnecessarily high.
        good_threshold = upper_good + min(gap / 3, max(0.5, upper_good * 0.1))
        bad_threshold = upper_good + min(2 * gap / 3, max(1.0, upper_good * 0.5))
    else:
        # Abstain across the overlap; strict good comparison, inclusive bad.
        good_threshold = lower_bad
        bad_threshold = math.nextafter(upper_good, math.inf)
    payload = {"schema_version": SCHEMA_VERSION, "detector_version": DETECTOR_VERSION,
               "good_threshold": good_threshold, "bad_threshold": bad_threshold,
               "calibration": {"split": "validation", "good_count": len(good), "bad_count": len(bad),
                               "max_good_score": upper_good, "min_bad_score": lower_bad},
               "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               "score_description": "row-repetition ratio; larger indicates vertical smearing; not a probability"}
    payload["model_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload


def load_model(path):
    model = json.loads(path.read_text())
    if model.get("schema_version") != SCHEMA_VERSION or model.get("detector_version") != DETECTOR_VERSION:
        raise ValueError("Detector configuration is incompatible")
    low, high = model["good_threshold"], model["bad_threshold"]
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        raise ValueError("Expected finite thresholds: 0 <= good_threshold < bad_threshold")
    return model


def decide(score, model):
    if score is None:
        return "unreadable"
    if score < model["good_threshold"]:
        return "good"
    if score >= model["bad_threshold"]:
        return "bad"
    return "review"


def evaluate(manifest, model, split):
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if model["manifest_sha256"] != fingerprint:
        raise ValueError("Configuration was calibrated against a different manifest")
    rows = [{**r, "decision": decide(r["score"], model)} for r in manifest["samples"] if r["split"] == split]
    if not rows:
        raise ValueError(f"No samples in split: {split}")
    confusion = {label: {decision: 0 for decision in ("good", "bad", "review", "unreadable")} for label in ("good", "bad")}
    for row in rows:
        confusion[row["label"]][row["decision"]] += 1
    tp, fp = confusion["bad"]["bad"], confusion["good"]["bad"]
    bad_count = sum(confusion["bad"].values())
    return {"split": split, "count": len(rows), "model_id": model["model_id"],
            "confusion": confusion, "bad_precision": tp / (tp + fp) if tp + fp else None,
            "bad_recall": tp / bad_count if bad_count else None,
            "good_incorrectly_rejected": fp,
            "bad_incorrectly_accepted": confusion["bad"]["good"],
            "review_rate": sum(r["decision"] == "review" for r in rows) / len(rows),
            "unreadable_count": sum(r["decision"] == "unreadable" for r in rows),
            "limitations": "Same camera; labels are confounded with capture season. This does not establish accuracy on other cameras or glitch types.",
            "samples": rows}


def copy_exclusive(source: Path, destination: Path):
    # Exclusive creation protects even against a collision after preflight.
    created = False
    try:
        with source.open("rb") as src, destination.open("xb") as dst:
            created = True
            shutil.copyfileobj(src, dst)
        shutil.copystat(source, destination)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def sort_images(source: Path, output: Path, model, *, dry_run=False, move=False, report: Path | None = None):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f"Input is not a directory: {source}")
    if source.is_relative_to(output):
        raise ValueError("Output must not equal or contain the input directory")
    paths = list(discover(source, exclude=output))
    if not paths:
        raise ValueError("No supported images found")
    rows = []
    for i, path in enumerate(paths, 1):
        try:
            detection = asdict(detect(path, model["bad_threshold"]))
            error = None
        except DecodeError as exc:
            detection, error = {"score": None, "onset_fraction": None}, str(exc)
        decision = decide(detection["score"], model)
        destination = output / decision / path.relative_to(source)
        rows.append({"source": str(path), "destination": str(destination), **detection,
                     "decision": decision, "error": error, "model_id": model["model_id"],
                     "status": "planned"})
        if i % 100 == 0:
            print(f"Scored {i}/{len(paths)}", file=sys.stderr)
    report = report.resolve() if report else output / "report.jsonl"
    # Preflight every target before creating directories or copying any photo.
    targets = [Path(r["destination"]) for r in rows] + [report]
    if len(set(targets)) != len(targets):
        raise ValueError("Report path collides with an image destination")
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        # Resolve parent to reject pre-existing symlinks that escape output.
        if target != report and not target.resolve().is_relative_to(output):
            raise ValueError(f"Destination escapes output directory: {target}")
    if report in paths:
        raise ValueError("Report path collides with an input image")
    if dry_run:
        return rows
    report.parent.mkdir(parents=True, exist_ok=True)
    # Record per-file progress so interrupted runs retain an audit trail.
    with report.open("x") as handle:
        for row in rows:
            src, dst = Path(row["source"]), Path(row["destination"])
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                copy_exclusive(src, dst)
                if move:
                    src.unlink()
                row["status"] = "moved" if move else "copied"
            except OSError as exc:
                row["status"], row["operation_error"] = "failed", str(exc)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    return rows
