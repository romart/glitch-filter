# Development results

Measured on the supplied 800-image collection with `row-repetition-v3`, seed 42,
and configuration ID `b84c865600103b12`. CPU only; no GPU initialization or neural
training. The configuration was calibrated using 121 good and 7 bad validation
images. Thresholds are approximately 6.992 for acceptance and 9.535 for rejection.

| Split | Labeled good | Labeled bad | Bad detected | Good rejected | Bad accepted | Review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 374 | 111 | 111 | 0 | 0 | 1 good |
| Validation | 121 | 7 | 7 | 0 | 0 | 0 |
| Test | 115 | 72 | 72 | 0 | 0 | 0 |
| Entire collection | 610 | 190 | 190 | 0 | 0 | 1 good |

All images decoded successfully. On the entire collection, bad-image precision
is 100%, bad-image recall is 100%, and the review rate is 0.125%. These are
development measurements on this collection, not guarantees for new images.

The test partition is January 4 for good images, and August 25 and 27 for bad
images. Validation is January 6 for good images, and August 29 plus the
November-named screenshot for bad images. Remaining capture-day groups are
assigned to train. All three originally named examples belong to train and are
classified as bad:

| Example | Score | Decision |
| --- | ---: | --- |
| `images/bad/img.jpg` | 335.90 | bad |
| `images/bad/yaros_20250422150642.jpg` | 229.92 | bad |
| `images/bad/yaros_20250624113422.jpg` | 552.28 | bad |

Label correction and review case:

- `images/good/yaros_20250826103411.jpg`: the user visually verified that this
  photo is good. It was moved from `images/bad/` to `images/good/`. Its good
  decision (score 6.941, acceptance cutoff 6.992) is correct, and the previous
  apparent miss was a labeling error. The manifest, configuration fingerprint,
  and evaluation reports have been regenerated. Capture-day split assignments
  and thresholds did not change.
- `images/good/yaros_20250105104015.jpg`: sent to review, score 7.344.

The first detector achieved 69/72 bad detections in the test partition and 0/115
good rejections. During development its test results and training failures were
inspected, and the detector was revised to handle weak streaks and gray fill.
The final test partition therefore is not a fresh blind benchmark. Seasonal
differences between labels and the small number of bad validation samples also
limit what can be inferred. Seek a fresh, mixed-date labeled collection for the
next independent evaluation.

Reproduce the measurements using the audit, calibrate, and evaluate commands in
[README.md](README.md). Local detailed reports are in `artifacts/evaluation.json`,
`artifacts/train-evaluation.json`, and `artifacts/validation-evaluation.json`.
The manifest includes hashes, capture groups, split membership, and all scores.
Corrected sample copies and the per-file JSONL report are in
`artifacts/corrected-sorted/`. Older directories contain historical runs with the
original label and configuration ID.

## CPU performance

A 32-photo sequential benchmark on the current shared machine measured:

| Stage | Before vectorization | After vectorization |
| --- | ---: | ---: |
| Decode and horizontal resize, including FFmpeg startup | 54.47 ms/image | 54.56 ms/image |
| Row-repetition profile | 31.98 ms/image | 1.46 ms/image |
| Decode + profile throughput | 11.57 images/s | 17.85 images/s |

Each run sampled 16 images evenly across each sorted class directory. These are
wall-clock measurements, excluding copying and reporting, subject to system
load and filesystem caches. No GPU was used. Profile computation became about
22 times faster; the measured combined throughput improved about 54%.

The optimization computes per-row differences once and reuses integer prefix
sums across overlapping bands, removing the Python loop over bands. Float32
mean rounding is preserved. A fresh audit of all 800 images produced exactly
the same scores and manifest fields as the original implementation on corrected
labels, so detector version and thresholds remain compatible. Decoding now
dominates the runtime; further scoring-kernel optimization has little potential
to improve total throughput. The full 800-photo sort, including copies and the
JSONL report, completed in 43.83 seconds (18.25 images/s). All 800 source and copy
hashes matched the corrected audit. Timing is recorded locally in
`artifacts/sort-benchmark.json`.

Verification includes 18 automated tests, a package wheel build, a real CLI
dry-run/copy check with all three named glitches plus a good and unreadable file,
and a full 800-image copy run. Source and destination hashes are verified against
the audit manifest after sorting; originals remain in `images/`.
