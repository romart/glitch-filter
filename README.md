# Photo glitch filter

A local CPU tool for detecting vertical image smearing and gray decoder
concealment, then sorting photos into `good`, `bad`, `review`, and `unreadable`
directories. It uses NumPy and FFmpeg; it does not initialize a GPU or run neural
training.

## Install

Use Python 3.10 or later and install FFmpeg through your operating system's package
manager (`sudo apt install ffmpeg` on Debian/Ubuntu). Then, from this repository:

```sh
./install.sh
source .venv/bin/activate
photo-filter --help
```

The installer creates or reuses `.venv`, installs the package and Python
dependencies, and checks the CLI version. It needs network access when
dependencies are not cached. It does not install system packages, calibrate the
detector, process photos, or start training. Use `./install.sh --help` for details;
set `PHOTO_FILTER_PYTHON=/path/to/python` to select a Python executable. On
Debian/Ubuntu, install `python3-venv` if virtual environment creation fails.

The equivalent manual setup is:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
source .venv/bin/activate
photo-filter --help
```

Keep the environment active for the commands below. Without activation, use
`.venv/bin/python` in place of `python3`, or `.venv/bin/photo-filter` for the CLI.

If your current Python already has NumPy and FFmpeg is on PATH, all commands also
work directly as `python3 -m photo_filter ...` without installing the package.

## Quick start

The calibrated detector is committed at `photo_filter/models/detector.json` and
included in installed packages. Sorting uses it by default, so calibration and
`--model` are optional. After installing and activating the environment, copy
photos into a new output directory with:

```sh
photo-filter sort /path/to/photos --output /path/to/sorted
```

The command scans subdirectories, shows progress, preserves the originals, and
writes classified copies plus `report.jsonl`. Use `--model /path/to/detector.json`
to select a custom detector. The bundled default works from any working directory.

## Prepare and evaluate a detector

To recalibrate instead of using the bundled detector, the reference collection
is `images/good/` and `images/bad/`. Run:

```sh
python3 -m photo_filter audit images --output artifacts/manifest.json
python3 -m photo_filter calibrate artifacts/manifest.json --output artifacts/detector.json
python3 -m photo_filter evaluate artifacts/manifest.json --model artifacts/detector.json --output artifacts/evaluation.json
```

`audit` fully decodes supported images, records SHA-256 hashes and repetition
scores, and assigns reproducible train/validation/test splits. It groups files by
capture date parsed from filenames, links exact duplicates, and rejects identical
files with conflicting labels. Unknown dates form one conservative session. You
need at least three independent groups per class. The default seed is 42; use
`--seed` to change it and `--workers 1` to limit decoding concurrency. The default
is two workers, with FFmpeg restricted to CPU decoding and one thread per worker.

`audit` and `sort` show progress automatically in a terminal: completed images,
percentage, images per second, elapsed time, and estimated time remaining for
each phase. Indexing/discovery show a count until the total is known. Sorting
tracks scoring and copying/moving separately; dry-run only discovers and scores.
Progress goes to stderr and refreshes at most once per second, so stdout remains
valid JSON/JSONL. Use `--progress always` for periodic lines in redirected logs,
or `--no-progress` (equivalently `--progress never`) to disable it. The default
`--progress auto` suppresses progress when stderr is redirected. Final summaries
and errors are still printed.

`calibrate` selects two thresholds using **validation samples only** and saves a
small JSON configuration. No neural model is trained. Below the lower threshold
is `good`, at or above the upper threshold is `bad`, and the interval between is
`review`. Scores are measurements, not probabilities. When class scores overlap,
the overlap becomes the review interval.

`evaluate` uses the saved scores from the audit snapshot, rather than decoding
images again. It checks the manifest fingerprint against the configuration and
reports counts, precision, recall, review rate, unreadable files, and each sample's
decision. Use `--split train` or `--split validation` for the other partitions.
Rerun the audit after changing the dataset or detector version.

## Sort a directory

Preview decisions without creating any files, including a report:

```sh
photo-filter sort /path/to/incoming --output /path/to/sorted --dry-run
```

The preview prints one JSON object per image to stdout and a summary to stderr.
When satisfied, copy the photos and save `sorted/report.jsonl`:

```sh
photo-filter sort /path/to/incoming --output /path/to/sorted
```

`--dry-run` intentionally creates no output directory, photos, or report. In a
normal run, the output directory is created **after scoring and collision checks
finish**, when copying starts. It can therefore be absent while scoring is still
in progress. The CLI prints the selected mode and output location. A missing
input, no supported images, or a failed collision check exits with an error before
creating output. Only decision subdirectories that receive files are created.

For example, `incoming/day1/photo.jpg` becomes
`sorted/bad/day1/photo.jpg` when classified as bad. Relative paths are preserved.
Supported extensions are JPG/JPEG, PNG, WebP, BMP, and TIF/TIFF, case insensitive.
Decode errors go to `unreadable` with the decoder error recorded. Other file types
are skipped. Scores, estimated smear onset fractions where available, configuration
IDs, destinations, and copy outcomes appear in the report. Reported width is the
512-pixel analysis width, and height is the decoded vertical resolution.

Copying is the default. `--move` removes a source only after its copy succeeds.
`--report PATH` chooses a different JSONL report location. The sorter refuses to
overwrite photos or reports; use a fresh output directory for another run. It
checks all targets before writing, excludes a nested output directory from input
discovery, and skips input symlinks. File-operation failures are recorded and
produce a nonzero exit status. A failed run can leave completed copies; the report
records progress and the source remains available unless a move succeeded.

To reproduce the sample sorting run without altering `images/`:

```sh
python3 -m photo_filter sort images --output artifacts/sorted --model artifacts/detector.json
```

## Detection and limitations

The detector resizes only the horizontal axis to 512 pixels, preserving vertical
resolution. Across overlapping eight-row bands, it compares horizontal texture
with differences between adjacent rows in four horizontal segments. Three
consecutive bands must support a repetition score, covering at least 16 rows.
Separate checks detect a mostly uniform lower region or neutral gray decoder
fill. A suspected onset is an estimate, not a segmentation mask; gray-fill cases
may have no onset estimate.

This targets the corruption observed in this camera's samples. Other corruption
types, genuine flat gray areas, and glitches thinner than the supported bands
can be misclassified. Natural black/white letterboxing is excluded from the gray
fill rule; entirely black images are not necessarily identified as corrupt.

The corrected collection has 610 good and 190 bad photos. The user confirmed
`yaros_20250826103411.jpg` is good; it was moved from `images/bad/` to
`images/good/`. Good samples are mostly from January and bad samples mostly from
August, with different construction stages.
Date grouping prevents adjacent frames from crossing splits but cannot remove
that seasonal class confound. Near duplicates across different days are not
automatically detected unless their bytes are identical. Add intact and glitched
images from overlapping dates and other cameras before claiming broad accuracy.

The detector was revised during development after inspecting the first evaluation
and training failures. The saved test results are therefore development results,
not a fresh blind benchmark. See [RESULTS.md](RESULTS.md) for counts and known
review cases. A pretrained neural classifier remains a possible later comparison if
new defect families require it; GPU training must be explicitly requested.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

Tests cover streak and gray-fill detection, calibration boundaries and isolation
from test scores, grouped splits, conflicting labels, dry-run, copying, moving,
collision prevention, failures, output exclusion, symlink escapes, progress
updates/interruption, and JSON output isolation. Generated
manifests, configurations, reports, sorted copies, and build products are ignored
by git. The reference image collection is not added to git automatically.
