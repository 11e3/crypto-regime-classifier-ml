#!/usr/bin/env python
"""Copy trained models to crypto-quant-system.

Usage:
    python copy_models.py
    python copy_models.py --dry-run  # Preview without copying
    python copy_models.py --include-csv  # Include CSV files too
"""

import argparse
import shutil
from pathlib import Path


# Model files to copy
MODEL_FILES = [
    # Deep learning models (v2 = 2-class)
    "regime_classifier_lstm_v2.pt",
    "regime_classifier_transformer_v2.pt",
    "regime_classifier_cnn_lstm_v2.pt",
    # Ensemble models
    "rf_xgb_lstm_ensemble.joblib",
    "supervised_dl_ensemble.joblib",
    "hybrid_ensemble.joblib",
]

# Optional: v1 models (4-class)
V1_MODEL_FILES = [
    "regime_classifier_lstm_v1.pt",
    "regime_classifier_transformer_v1.pt",
    "regime_classifier_cnn_lstm_v1.pt",
]

# CSV files (optional)
CSV_FILES = [
    "model_comparison.csv",
    "walk_forward_lstm_results.csv",
    "walk_forward_transformer_results.csv",
    "walk_forward_quick_results.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Copy models to crypto-quant-system")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument("--include-v1", action="store_true", help="Include v1 (4-class) models")
    parser.add_argument("--include-csv", action="store_true", help="Include CSV result files")
    parser.add_argument(
        "--src", type=str, default="models", help="Source directory (default: models)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="../crypto-quant-system/models",
        help="Destination directory (default: ../crypto-quant-system/models)",
    )
    return parser.parse_args()


def copy_models(src_dir: Path, dst_dir: Path, files: list, dry_run: bool = False):
    """Copy model files from src to dst."""
    copied = []
    skipped = []

    for filename in files:
        src_path = src_dir / filename
        dst_path = dst_dir / filename

        if not src_path.exists():
            skipped.append(filename)
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would copy: {src_path} -> {dst_path}")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {filename}")

        copied.append(filename)

    return copied, skipped


def main():
    args = parse_args()

    # Resolve paths
    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()

    print("=" * 60)
    print("Copy Models to crypto-quant-system")
    print("=" * 60)
    print(f"Source:      {src_dir}")
    print(f"Destination: {dst_dir}")

    if args.dry_run:
        print("\n[DRY-RUN MODE - No files will be copied]\n")

    # Validate directories
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return 1

    if not dst_dir.exists():
        if args.dry_run:
            print(f"[DRY-RUN] Would create directory: {dst_dir}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created destination directory: {dst_dir}")

    # Build file list
    files_to_copy = MODEL_FILES.copy()

    if args.include_v1:
        files_to_copy.extend(V1_MODEL_FILES)

    if args.include_csv:
        files_to_copy.extend(CSV_FILES)

    # Copy files
    print(f"\nCopying {len(files_to_copy)} files...")
    print("-" * 40)

    copied, skipped = copy_models(src_dir, dst_dir, files_to_copy, args.dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Copied:  {len(copied)} files")
    print(f"Skipped: {len(skipped)} files (not found)")

    if skipped:
        print("\nSkipped files:")
        for f in skipped:
            print(f"  - {f}")

    if copied and not args.dry_run:
        print("\nCopied files:")
        for f in copied:
            print(f"  + {f}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
