# Photo processor project

## Purpose and current scope

Build a local tool that detects visually glitched photos and sorts them into
separate directories. The user requested a design suggestion first and suggested
a simple neural classifier, then authorized an end-to-end implementation. The
user explicitly prohibited starting GPU training because the GPU may be busy.
The implementation uses CPU NumPy and FFmpeg only; do not initialize a GPU or
start GPU training without a new explicit request. The repository started
without commits or code.

## Implemented workflow

- Package: `photo_filter/`, configured by `pyproject.toml`; Python >=3.10, NumPy,
  and the external FFmpeg executable. No torch dependency.
- `python3 -m photo_filter audit images` produces a scored manifest with seeded
  capture-day splits and linked exact duplicates; `calibrate MANIFEST` selects
  thresholds from validation samples only; `evaluate MANIFEST --model MODEL`
  reports decisions on a chosen split (test by default).
- `python3 -m photo_filter sort INPUT --output OUTPUT --model MODEL` copies
  images with a JSONL operation report. `--dry-run` writes nothing; `--move`
  explicitly enables removal after successful copy. Existing destinations and
  reports are rejected, relative paths are preserved, and nested output trees
  are excluded from discovery.
- See `README.md` for setup and full commands, `RESULTS.md` for measured results,
  and ignored `artifacts/` for local manifests, configuration, and sample output.
- Run `python3 -m unittest discover -s tests -v` for meaningful behavior checks.
- The first evaluation was inspected during development; revised results must
  not be described as a blind benchmark. A fresh labeled set is needed for that.

## Reference data

- Use `images/good/` and `images/bad/` as the labeled dataset. The user's reference
  to `image/bad/` means the existing `images/bad/` directory.
- Initial inventory: 609 good JPEGs and 191 bad JPEGs; no byte-identical duplicates.
- Corrected inventory: 610 good and 190 bad. The user verified
  `yaros_20250826103411.jpg` is good; it was moved to `images/good/`. It is a
  confirmed correct acceptance, not a detector miss. Updated results catch all
  190 bad images, with one good image sent to review and no good rejections.
- The row-repetition profile is vectorized with per-row integer prefix sums.
  This preserves original float32 mean rounding and all 800 scores exactly.
  Measured scoring time dropped from ~32 ms to ~1.5 ms per image; CPU decoding
  is now the main bottleneck. See `RESULTS.md` for benchmark scope and details.
- All good images and 189 bad images are 1920x1080; two bad images are 1280x720.
- `images/bad/img.jpg`: corruption in the lower portion.
- `images/bad/yaros_20250422150642.jpg`: essentially the whole image is glitched.
- `images/bad/yaros_20250624113422.jpg`: approximately the lower half is glitched.
- Inspected bad samples show vertical streaks caused by repeated or nearly
  repeated rows extending toward the bottom. This is an observed defect family,
  not proof that every bad image has the same defect.
- Good samples are dated January 3–7, 2025. Bad samples mostly come from August
  2025, with additional examples from other dates. Scene, season, construction
  progress, and resolution can therefore become misleading class shortcuts.
- Treat reference images as input fixtures. Preserve their bytes, labels, and
  paths; keep generated outputs outside the dataset directories.

## Design guidance

1. Audit decoding, dimensions, and labels; save a reproducible dataset manifest.
   Header validity alone does not establish that a photo is visually correct.
2. Benchmark an interpretable row-repetition detector. Measure adjacent-row
   differences in horizontal segments and sustained low differences toward the
   bottom; include horizontal texture so naturally flat areas do not dominate.
   Inspect high-resolution bottom strips as well as the full frame. Report a
   suspected onset row when supported by this detector.
3. If the baseline misses important cases, compare a small pretrained image
   classifier (e.g. ResNet-18 with a binary head) using PyTorch/torchvision.
   Start with a frozen backbone and train the head, then consider fine-tuning.
   Keep this as a candidate until held-out measurements support it.
4. Preserve the full image and bottom edge during preprocessing. Avoid default
   center crops, random crops that remove corruption, and vertical flips. If
   downscaling loses small defects, add a bottom-strip view or overlapping tiles.
   Do not assign every tile the image's bad label: some tiles can be intact.
5. Expose separate training/evaluation and sorting commands. A proposed sorting
   interface is `photo-filter sort INPUT --output OUTPUT --model MODEL --dry-run`;
   this interface is implemented by the CPU baseline.
6. Route results into `good/`, `bad/`, `review/`, and `unreadable/`. Use two
   validation-selected score thresholds, with uncertain scores sent to review.
   Scores must not be presented as calibrated probabilities without validation.
7. Default sorting to copying; require an explicit option for moving. Preserve
   relative paths, prevent overwrites, exclude the output tree from discovery,
   and write a CSV or JSONL report with source, destination, score, decision,
   decode failures, and model/configuration version.

## Evaluation and development guidance

- Split by capture day or session, keeping adjacent and near-duplicate frames
  together. Keep derived/augmented samples with their originals. Record the
  split and random seed; tune thresholds on validation data, not the test set.
- Date grouping does not remove the January-versus-August class confound. Seek
  intact and glitched examples from the same dates/scenes before claiming broad
  accuracy. Never use filenames, timestamps, paths, or file sizes as predictors.
- Report bad-image precision/recall, good images incorrectly rejected, review
  rate, unreadable count, and a confusion matrix with counts. Inspect errors and
  the three named examples individually; report their split membership.
- CPU inference should work; make GPU acceleration optional. Choose dependencies
  and packaging when implementation begins rather than assuming an environment.
- Add meaningful checks for sorting behavior: dry-run writes nothing, originals
  survive copy mode, collisions cannot overwrite files, and recursive discovery
  cannot consume its own output. Assess detection with held-out labeled images.
- Keep checkpoints, caches, output directories, and reports out of source control.
  Do not silently add the image collection to git or commit changes without a
  user request. Update this file as implementation establishes real commands.
