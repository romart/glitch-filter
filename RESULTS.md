# Development results

Measured on the supplied 800-image collection with `row-repetition-v3`, seed 42,
and configuration ID `8f834e8cf5ce653a`. CPU only; no GPU initialization or neural
training. The configuration was calibrated using 121 good and 7 bad validation
images. Thresholds are approximately 6.992 for acceptance and 9.535 for rejection.

| Split | Labeled good | Labeled bad | Bad detected | Good rejected | Bad accepted | Review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 373 | 112 | 111 | 0 | 1 | 1 good |
| Validation | 121 | 7 | 7 | 0 | 0 | 0 |
| Test | 115 | 72 | 72 | 0 | 0 | 0 |
| Entire collection | 609 | 191 | 190 | 0 | 1 | 1 good |

All images decoded successfully. On the entire collection, bad-image precision
is 100%, bad-image recall is 99.48%, and the review rate is 0.125%. These are
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

Known exceptions:

- `images/bad/yaros_20250826103411.jpg`: accepted as good, score 6.941. The
  acceptance cutoff is 6.992. Inspection shows a mostly intact building scene
  with blocky detail, without the broad repeated-row smearing or gray fill
  targeted by this detector. The exact corruption mechanism is unconfirmed;
  native bottom rows also retain substantial differences instead of repeating.
  This is a detector coverage limitation, not a decoding failure. Keep it as a
  regression example for an additional artifact detector or label review.
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
Sorted sample copies and the per-file JSONL report are in `artifacts/sorted/`.

Verification includes 18 automated tests, a package wheel build, a real CLI
dry-run/copy check with all three named glitches plus a good and unreadable file,
and a full 800-image copy run. Source and destination hashes are verified against
the audit manifest after sorting; originals remain in `images/`.
