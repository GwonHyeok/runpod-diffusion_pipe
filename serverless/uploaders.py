"""
Uploader abstraction for RunPod Serverless LoRA Training
Supports multiple cloud storage backends (GCP, RunPod)
"""

import os
import json
import base64
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from datetime import timedelta

from utils import retry

logger = logging.getLogger(__name__)


class Uploader(ABC):
    """Abstract base class for file uploaders"""

    @abstractmethod
    async def upload(self, files: List[Path], job_id: str) -> List[str]:
        """
        Upload files and return accessible URLs

        Args:
            files: List of file paths to upload
            job_id: Job ID for organizing uploads

        Returns:
            List of URLs for uploaded files
        """
        pass


class GCPUploader(Uploader):
    """Google Cloud Storage uploader with signed URLs"""

    def __init__(self, bucket_name: str, credentials_json: str):
        """
        Initialize GCP uploader

        Args:
            bucket_name: GCS bucket name
            credentials_json: Service account JSON credentials (as string)
        """
        self.bucket_name = bucket_name
        self.credentials_json = credentials_json
        self._client = None
        self._bucket = None

    def _initialize_client(self):
        """Lazy initialization of GCS client"""
        if self._client is not None:
            return

        try:
            from google.cloud import storage
            from google.oauth2 import service_account

            # Parse credentials JSON (supports both plain JSON and base64-encoded JSON)
            try:
                # First, try to decode as base64 (safer for environment variables)
                try:
                    decoded_bytes = base64.b64decode(self.credentials_json)
                    credentials_dict = json.loads(decoded_bytes.decode("utf-8"))
                    logger.info("GCP credentials loaded from base64-encoded JSON")
                except Exception:
                    # If base64 decoding fails, try parsing as plain JSON
                    credentials_dict = json.loads(self.credentials_json)
                    logger.info("GCP credentials loaded from plain JSON")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid GCP credentials JSON format: {e}")

            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict
            )

            # Initialize client
            self._client = storage.Client(
                credentials=credentials, project=credentials_dict.get("project_id")
            )

            # Get bucket
            self._bucket = self._client.bucket(self.bucket_name)

            logger.info(
                f"GCP Storage client initialized for bucket: {self.bucket_name}"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCP Storage client: {e}")

    @retry(max_attempts=3, backoff=2.0)
    async def upload(self, files: List[Path], job_id: str) -> List[str]:
        """
        Upload files to GCS with parallel uploads and return signed URLs

        Args:
            files: List of file paths to upload
            job_id: Job ID for organizing uploads (used as prefix)

        Returns:
            List of signed URLs (7-day expiration)
        """
        if not files:
            logger.warning("No files to upload")
            return []

        # Initialize client on first use
        self._initialize_client()

        logger.info(
            f"Starting upload of {len(files)} files to GCS bucket: {self.bucket_name}"
        )

        # Use semaphore to limit concurrent uploads (5 at a time)
        semaphore = asyncio.Semaphore(5)

        async def upload_single_file(file_path: Path) -> str:
            """Upload a single file and return signed URL"""
            async with semaphore:
                # Determine blob name (object path in GCS)
                # Format: {job_id}/{filename}
                blob_name = f"{job_id}/{file_path.name}"

                # Log upload start
                file_size_mb = file_path.stat().st_size / (1024**2)
                logger.info(
                    f"Uploading {file_path.name} ({file_size_mb:.1f} MB) to gs://{self.bucket_name}/{blob_name}"
                )

                try:
                    # Run upload in thread pool (GCS SDK is blocking)
                    loop = asyncio.get_event_loop()
                    url = await loop.run_in_executor(
                        None, self._upload_and_sign, file_path, blob_name
                    )

                    logger.info(f"Uploaded: {file_path.name} → {blob_name}")
                    return url

                except Exception as e:
                    logger.error(f"Upload failed for {file_path.name}: {e}")
                    raise RuntimeError(f"GCP upload failed for {file_path.name}: {e}")

        # Upload all files in parallel (up to 5 concurrent)
        upload_tasks = [upload_single_file(f) for f in files]
        upload_urls = await asyncio.gather(*upload_tasks)

        logger.info(f"Successfully uploaded {len(upload_urls)} files to GCS")
        return upload_urls

    def _upload_and_sign(self, file_path: Path, blob_name: str) -> str:
        """
        Upload file and generate signed URL (blocking operation)

        Args:
            file_path: Local file path
            blob_name: GCS object name

        Returns:
            Signed URL with 7-day expiration
        """
        # Create blob
        blob = self._bucket.blob(blob_name)

        # Upload file (raises exception if fails)
        try:
            blob.upload_from_filename(str(file_path))
        except Exception as e:
            raise RuntimeError(f"GCS upload failed for {blob_name}: {e}")

        # Generate signed URL (7 days expiration)
        signed_url = blob.generate_signed_url(
            version="v4", expiration=timedelta(days=7), method="GET"
        )

        return signed_url


class RunPodUploader(Uploader):
    """RunPod native uploader using rp_upload utility"""

    @retry(max_attempts=3, backoff=2.0)
    async def upload(self, files: List[Path], job_id: str) -> List[str]:
        """
        Upload files using RunPod's rp_upload utility

        Uses runpod.serverless.utils.rp_upload module to upload to S3-compatible storage.
        Requires environment variables:
        - BUCKET_ENDPOINT_URL: S3 endpoint URL (e.g., https://bucket.s3.region.amazonaws.com)
        - BUCKET_ACCESS_KEY_ID: S3 access key
        - BUCKET_SECRET_ACCESS_KEY: S3 secret key

        Documentation: https://docs.runpod.io/serverless/development/environment-variables

        Args:
            files: List of file paths to upload
            job_id: Job ID for organizing uploads

        Returns:
            List of URLs from S3-compatible storage
        """
        if not files:
            logger.warning("No files to upload")
            return []

        logger.info(f"Starting upload of {len(files)} files using RunPod rp_upload")

        try:
            from runpod.serverless.utils import rp_upload
        except ImportError:
            raise RuntimeError(
                "runpod.serverless.utils.rp_upload not available. "
                "Ensure RunPod SDK is installed and BUCKET_* environment variables are set."
            )

        upload_urls = []

        for file_path in files:
            # Log upload start
            file_size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"Uploading {file_path.name} ({file_size_mb:.1f} MB)")

            try:
                # Use RunPod's upload_image function
                # This uploads to S3-compatible storage configured via environment variables
                # The function returns a URL to the uploaded file
                url = rp_upload.upload_image(job_id, str(file_path))

                upload_urls.append(url)
                logger.info(f"Uploaded: {url}")

            except Exception as e:
                logger.error(f"Upload failed for {file_path.name}: {e}")
                raise RuntimeError(f"RunPod upload failed for {file_path.name}: {e}")

        logger.info(f"Successfully uploaded {len(upload_urls)} files using rp_upload")
        return upload_urls


def create_uploader(job_id: str) -> Uploader:
    """
    Factory function to create appropriate uploader based on environment variables

    Priority:
    1. If GCP_SERVICE_ACCOUNT_JSON and GCS_BUCKET_NAME are set, use GCPUploader
    2. Otherwise, use RunPodUploader (fallback)

    Args:
        job_id: Job ID for logging purposes

    Returns:
        Configured Uploader instance
    """
    gcp_credentials = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")

    if gcp_credentials and gcs_bucket_name:
        logger.info(
            f"[Job {job_id}] Using GCP Storage uploader (bucket: {gcs_bucket_name})"
        )
        return GCPUploader(
            bucket_name=gcs_bucket_name, credentials_json=gcp_credentials
        )
    else:
        logger.info(f"[Job {job_id}] Using RunPod native uploader (fallback)")
        return RunPodUploader()


# Export for easy imports
__all__ = ["Uploader", "GCPUploader", "RunPodUploader", "create_uploader"]
