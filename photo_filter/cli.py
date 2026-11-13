import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from . import __version__
from .detector import require_decoder
from .workflow import audit, calibrate, evaluate, load_manifest, load_model, sort_images, write_json

DEFAULT_MODEL = Path(__file__).with_name("models") / "detector.json"


def parser():
    root = argparse.ArgumentParser(description="CPU-only vertical-smear detector and photo sorter")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("audit", help="Decode labeled data, score it, and record grouped splits")
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    progress_options(p)
    p = commands.add_parser("calibrate", help="Select thresholds using validation samples only")
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/detector.json"))
    p = commands.add_parser("evaluate", help="Evaluate saved scores on one split")
    p.add_argument("manifest", type=Path)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help="Detector JSON (default: bundled calibrated detector)")
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    p = commands.add_parser("sort", help="Score and copy images into decision directories")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help="Detector JSON (default: bundled calibrated detector)")
    p.add_argument("--report", type=Path, help="JSONL report, defaults to OUTPUT/report.jsonl")
    p.add_argument("--dry-run", action="store_true", help="Print planned JSONL to stdout; write no files")
    p.add_argument("--move", action="store_true", help="Remove each source only after a successful copy")
    progress_options(p)
    return root


def progress_options(command):
    command.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                         help="Progress on stderr: auto for terminals, always for logs, never to disable")
    command.add_argument("--no-progress", dest="progress", action="store_const", const="never",
                         help="Disable progress output")


def main(argv=None):
    args = parser().parse_args(argv)
    mode = getattr(args, "progress", "never")
    progress = mode == "always" or (mode == "auto" and sys.stderr.isatty())
    try:
        if args.command == "audit":
            require_decoder()
            manifest = audit(args.dataset, args.seed, args.workers, progress=progress)
            write_json(args.output, manifest)
            counts = Counter((r["split"], r["label"]) for r in manifest["samples"])
            print(json.dumps({f"{s}/{l}": n for (s, l), n in sorted(counts.items())}, indent=2))
            print(f"Manifest: {args.output}", file=sys.stderr)
        elif args.command == "calibrate":
            model = calibrate(load_manifest(args.manifest))
            write_json(args.output, model)
            print(json.dumps(model, indent=2))
        elif args.command == "evaluate":
            report = evaluate(load_manifest(args.manifest), load_model(args.model), args.split)
            write_json(args.output, report)
            print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
        elif args.command == "sort":
            require_decoder()
            model = load_model(args.model)
            if args.dry_run:
                print("Dry run: no files or output directory will be created. Remove --dry-run to copy photos.",
                      file=sys.stderr)
            else:
                print(f"Output will be created after scoring and collision checks: {args.output.resolve()}",
                      file=sys.stderr)
            rows = sort_images(args.input, args.output, model,
                               dry_run=args.dry_run, move=args.move, report=args.report, progress=progress)
            if args.dry_run:
                for row in rows:
                    print(json.dumps(row, allow_nan=False))
            print(json.dumps({"count": len(rows), "dry_run": args.dry_run, "output": str(args.output.resolve()),
                              "decisions": dict(Counter(r["decision"] for r in rows)),
                              "statuses": dict(Counter(r["status"] for r in rows))}), file=sys.stderr)
            if any(r["status"] == "failed" for r in rows):
                return 1
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"photo-filter: {exc}", file=sys.stderr)
        return 2
    return 0
