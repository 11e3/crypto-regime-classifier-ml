#!/usr/bin/env python
"""Upload trained model to Google Cloud Storage."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from src.utils.gcs import upload_to_gcs, GCSClient, HAS_GCS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload regime classifier model to GCS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.pkl file)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("GCS_BUCKET", "crypto-regime-classifier"),
        help="GCS bucket name",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Don't update the 'latest' pointer",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to JSON file with model metrics",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the model version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )

    return parser.parse_args()


def main():
    """Main upload function."""
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Generate version if not provided
    version = args.version
    if version is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    # Prepare metadata
    metadata = {
        "version": version,
        "upload_time": datetime.now().isoformat(),
        "source_file": str(model_path),
        "file_size_bytes": model_path.stat().st_size,
    }

    if args.description:
        metadata["description"] = args.description

    # Load metrics if provided
    if args.metrics:
        metrics_path = Path(args.metrics)
        if metrics_path.exists():
            with open(metrics_path) as f:
                metadata["metrics"] = json.load(f)

    # Print upload info
    print("=" * 60)
    print("Crypto Regime Classifier - GCS Upload")
    print("=" * 60)
    print(f"\nModel file: {model_path}")
    print(f"File size: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"Version: {version}")
    print(f"Bucket: {args.bucket}")
    print(f"Update latest: {not args.no_latest}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        print(f"  - gs://{args.bucket}/models/regime_classifier_{version}.pkl")
        if not args.no_latest:
            print(f"  - gs://{args.bucket}/models/regime_classifier_latest.pkl")
        print(f"  - gs://{args.bucket}/models/metadata/{version}.json")
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
        return 0

    if not HAS_GCS:
        print("\nError: google-cloud-storage is not installed.")
        print("Install with: pip install google-cloud-storage")
        print("\nAlternatively, use gsutil directly:")
        print(f"  gsutil cp {model_path} gs://{args.bucket}/models/regime_classifier_{version}.pkl")
        return 1

    # Upload
    print("\nUploading...")
    try:
        upload_to_gcs(
            local_path=model_path,
            version=version,
            bucket_name=args.bucket,
            update_latest=not args.no_latest,
            metadata=metadata,
        )

        print("\n" + "=" * 60)
        print("Upload Complete!")
        print("=" * 60)
        print(f"\nModel URL: gs://{args.bucket}/models/regime_classifier_{version}.pkl")
        if not args.no_latest:
            print(f"Latest URL: gs://{args.bucket}/models/regime_classifier_latest.pkl")

    except Exception as e:
        print(f"\nError uploading: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
