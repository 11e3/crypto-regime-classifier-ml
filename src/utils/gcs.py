"""Google Cloud Storage utilities."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# GCS import with fallback for local development
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    storage = None


# Default bucket name (can be overridden via environment variable)
DEFAULT_BUCKET = os.environ.get("GCS_BUCKET", "crypto-regime-classifier")
DEFAULT_MODEL_PREFIX = "models/"


class GCSClient:
    """Google Cloud Storage client wrapper.

    Usage:
        client = GCSClient(bucket_name="my-bucket")
        client.upload("local/model.pkl", "models/v1.pkl")
        client.download("models/v1.pkl", "local/model.pkl")
    """

    def __init__(
        self,
        bucket_name: str = None,
        project: str = None,
    ):
        """Initialize GCS client.

        Args:
            bucket_name: GCS bucket name
            project: GCP project ID (optional)
        """
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage is required. "
                "Install with: pip install google-cloud-storage"
            )

        self.bucket_name = bucket_name or DEFAULT_BUCKET
        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(self.bucket_name)

    def upload(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        content_type: str = None,
    ):
        """Upload file to GCS.

        Args:
            local_path: Local file path
            remote_path: Remote path in bucket
            content_type: Optional content type
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(str(local_path), content_type=content_type)
        print(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")

    def download(
        self,
        remote_path: str,
        local_path: Union[str, Path],
    ):
        """Download file from GCS.

        Args:
            remote_path: Remote path in bucket
            local_path: Local file path to save to
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob = self.bucket.blob(remote_path)
        blob.download_to_filename(str(local_path))
        print(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")

    def list_models(self, prefix: str = DEFAULT_MODEL_PREFIX) -> list[str]:
        """List all model files in bucket.

        Args:
            prefix: Path prefix to filter

        Returns:
            List of model paths
        """
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs if blob.name.endswith(".pkl")]

    def get_latest_model_path(self, prefix: str = DEFAULT_MODEL_PREFIX) -> Optional[str]:
        """Get path to the latest model.

        Args:
            prefix: Path prefix

        Returns:
            Path to latest model or None
        """
        # First check for explicit 'latest' pointer
        latest_path = f"{prefix}regime_classifier_latest.pkl"
        blob = self.bucket.blob(latest_path)
        if blob.exists():
            return latest_path

        # Otherwise find most recent by timestamp
        models = self.list_models(prefix)
        if not models:
            return None

        # Sort by modification time
        model_times = []
        for model_path in models:
            blob = self.bucket.blob(model_path)
            blob.reload()
            model_times.append((model_path, blob.updated))

        model_times.sort(key=lambda x: x[1], reverse=True)
        return model_times[0][0]

    def delete(self, remote_path: str):
        """Delete file from GCS.

        Args:
            remote_path: Remote path to delete
        """
        blob = self.bucket.blob(remote_path)
        blob.delete()
        print(f"Deleted gs://{self.bucket_name}/{remote_path}")


def upload_to_gcs(
    local_path: Union[str, Path],
    version: str = None,
    bucket_name: str = None,
    update_latest: bool = True,
    metadata: dict = None,
):
    """Upload model to GCS with versioning.

    Args:
        local_path: Local model file path
        version: Version string (e.g., "v1", "v2"). Auto-generated if None.
        bucket_name: GCS bucket name
        update_latest: Whether to also update the 'latest' pointer
        metadata: Optional metadata to save alongside model
    """
    local_path = Path(local_path)

    if not HAS_GCS:
        print("Warning: GCS not available. Skipping upload.")
        return

    # Auto-generate version if not provided
    if version is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    client = GCSClient(bucket_name=bucket_name)

    # Upload model
    remote_path = f"{DEFAULT_MODEL_PREFIX}regime_classifier_{version}.pkl"
    client.upload(local_path, remote_path)

    # Update 'latest' pointer
    if update_latest:
        latest_path = f"{DEFAULT_MODEL_PREFIX}regime_classifier_latest.pkl"
        client.upload(local_path, latest_path)
        print(f"Updated latest pointer: gs://{client.bucket_name}/{latest_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = f"{DEFAULT_MODEL_PREFIX}metadata/{version}.json"
        metadata["version"] = version
        metadata["upload_time"] = datetime.now().isoformat()
        metadata["model_path"] = remote_path

        # Save locally first
        temp_metadata = Path(f"/tmp/{version}_metadata.json")
        with open(temp_metadata, "w") as f:
            json.dump(metadata, f, indent=2)

        client.upload(temp_metadata, metadata_path, content_type="application/json")
        temp_metadata.unlink()  # Clean up


def download_from_gcs(
    remote_path: str = None,
    local_path: Union[str, Path] = None,
    bucket_name: str = None,
    version: str = "latest",
) -> Path:
    """Download model from GCS.

    Args:
        remote_path: Full remote path (optional)
        local_path: Local path to save to
        bucket_name: GCS bucket name
        version: Version string or "latest"

    Returns:
        Path to downloaded file
    """
    if not HAS_GCS:
        raise ImportError("GCS not available")

    client = GCSClient(bucket_name=bucket_name)

    # Determine remote path
    if remote_path is None:
        if version == "latest":
            remote_path = client.get_latest_model_path()
            if remote_path is None:
                raise FileNotFoundError("No models found in bucket")
        else:
            remote_path = f"{DEFAULT_MODEL_PREFIX}regime_classifier_{version}.pkl"

    # Determine local path
    if local_path is None:
        local_path = Path(f"models/{Path(remote_path).name}")

    local_path = Path(local_path)
    client.download(remote_path, local_path)

    return local_path
