"""
Download manager for RunPod Serverless LoRA Training
Handles downloading media files from URLs and caching models
"""

import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlparse
import mimetypes

from config import MODEL_CONFIGS
from utils import retry, run_subprocess

logger = logging.getLogger(__name__)

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Download limits
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB per file
DOWNLOAD_TIMEOUT = 600  # 10 minutes per file


class AssetDownloader:
    """Download images and videos from URLs"""

    def __init__(self, dest_dir: Path):
        self.dest_dir = Path(dest_dir)
        self.image_dir = self.dest_dir / "image_dataset_here"
        self.video_dir = self.dest_dir / "video_dataset_here"

    async def download_batch(
        self,
        image_urls: List[str],
        video_urls: List[str]
    ) -> Path:
        """
        Download all assets concurrently

        Args:
            image_urls: List of image URLs
            video_urls: List of video URLs

        Returns:
            Path to dataset directory containing image_dataset_here and video_dataset_here
        """
        logger.info(f"Starting batch download: {len(image_urls)} images, {len(video_urls)} videos")

        # Create directories
        if image_urls:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        if video_urls:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        # Download concurrently
        tasks = []

        # Queue image downloads
        for i, url in enumerate(image_urls):
            ext = self._get_extension_from_url(url, default='.jpg')
            filename = f"image_{i:04d}{ext}"
            dest_path = self.image_dir / filename
            tasks.append(self._download_file(url, dest_path, "image"))

        # Queue video downloads
        for i, url in enumerate(video_urls):
            ext = self._get_extension_from_url(url, default='.mp4')
            filename = f"video_{i:04d}{ext}"
            dest_path = self.video_dir / filename
            tasks.append(self._download_file(url, dest_path, "video"))

        # Execute all downloads
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.error(f"{len(failures)} downloads failed")
            for exc in failures[:5]:  # Log first 5 errors
                logger.error(f"Download error: {exc}")
            if len(failures) == len(tasks):
                raise RuntimeError("All downloads failed")

        successes = len([r for r in results if r is True])
        logger.info(f"Download complete: {successes}/{len(tasks)} succeeded")

        return self.dest_dir

    @retry(max_attempts=3, backoff=2.0)
    async def _download_file(
        self,
        url: str,
        dest_path: Path,
        file_type: str = "file"
    ) -> bool:
        """
        Download a single file with retry logic

        Args:
            url: URL to download from
            dest_path: Destination file path
            file_type: Type of file for logging (image/video/file)

        Returns:
            True if successful

        Raises:
            ValueError: If file is too large or invalid type
            aiohttp.ClientError: If download fails
        """
        logger.info(f"Downloading {file_type}: {url}")

        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()

                # Validate content length
                content_length = int(response.headers.get('content-length', 0))
                if content_length > MAX_FILE_SIZE:
                    raise ValueError(
                        f"File too large: {content_length / 1024**2:.1f} MB "
                        f"(max: {MAX_FILE_SIZE / 1024**2:.1f} MB)"
                    )

                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if file_type == "image" and not content_type.startswith('image/'):
                    logger.warning(f"Unexpected content-type for image: {content_type}")
                elif file_type == "video" and not content_type.startswith('video/'):
                    logger.warning(f"Unexpected content-type for video: {content_type}")

                # Download file
                downloaded_size = 0
                with open(dest_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Safety check
                        if downloaded_size > MAX_FILE_SIZE:
                            raise ValueError(f"Download exceeded max size: {MAX_FILE_SIZE}")

                file_size_mb = downloaded_size / 1024**2
                logger.info(f"Downloaded {dest_path.name} ({file_size_mb:.1f} MB)")

                return True

    def _get_extension_from_url(self, url: str, default: str = '') -> str:
        """
        Extract file extension from URL

        Args:
            url: URL to parse
            default: Default extension if none found

        Returns:
            File extension including dot (e.g., '.jpg')
        """
        parsed = urlparse(url)
        # Extract just the filename, not the full path (security: prevent path traversal)
        filename = parsed.path.split('/')[-1] if parsed.path else ''
        if not filename:
            return default

        ext = Path(filename).suffix.lower()

        if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            return ext

        return default


class ModelCacher:
    """Manage model downloads and caching in /runpod-volume/models/"""

    def __init__(self, cache_dir: Path = Path("/runpod-volume/models")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(max_attempts=3, backoff=5.0)
    async def get_or_download(
        self,
        model_type: str,
        hf_token: Optional[str] = None
    ) -> Path:
        """
        Check if model exists in cache, download if missing

        Args:
            model_type: Model type (flux, sdxl, etc.)
            hf_token: Hugging Face token for gated models

        Returns:
            Path to model in cache

        Raises:
            ValueError: If model_type is invalid
            RuntimeError: If download fails
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")

        model_config = MODEL_CONFIGS[model_type]
        model_path = self.cache_dir / model_config["path_in_volume"]

        # Check cache
        if self._validate_cached_model(model_path, model_type):
            logger.info(f"Using cached model: {model_path}")
            return model_path

        # Download model
        logger.info(f"Model not cached, downloading: {model_config['repo']}")
        await self._download_model(
            repo_id=model_config["repo"],
            local_dir=model_path,
            hf_token=hf_token,
            model_type=model_type
        )

        # Verify download
        if not self._validate_cached_model(model_path, model_type):
            raise RuntimeError(f"Model download verification failed: {model_path}")

        logger.info(f"Model cached successfully: {model_path}")
        return model_path

    def _validate_cached_model(self, model_path: Path, model_type: str) -> bool:
        """
        Validate that cached model exists and is complete

        Args:
            model_path: Path to model directory or file
            model_type: Type of model

        Returns:
            True if model is valid
        """
        if not model_path.exists():
            return False

        # For single file models (e.g., SDXL checkpoint)
        if model_path.is_file():
            size_mb = model_path.stat().st_size / 1024**2
            logger.info(f"Found cached model file: {model_path.name} ({size_mb:.1f} MB)")
            return size_mb > 100  # Should be at least 100MB

        # For directory-based models (e.g., Flux, Wan)
        if model_path.is_dir():
            # Check for .safetensors files
            safetensors_files = list(model_path.glob("**/*.safetensors"))
            if safetensors_files:
                total_size = sum(f.stat().st_size for f in safetensors_files) / 1024**2
                logger.info(
                    f"Found cached model: {len(safetensors_files)} files, "
                    f"{total_size:.1f} MB total"
                )
                return total_size > 100  # Should be at least 100MB

            # Check for other model files
            model_files = list(model_path.glob("**/*.bin")) + list(model_path.glob("**/*.pth"))
            if model_files:
                logger.info(f"Found cached model: {len(model_files)} files")
                return True

        return False

    async def _download_model(
        self,
        repo_id: str,
        local_dir: Path,
        hf_token: Optional[str],
        model_type: str
    ):
        """
        Download model from Hugging Face Hub

        Args:
            repo_id: Hugging Face repo ID
            local_dir: Local directory to save model
            hf_token: Hugging Face token (required for gated models)
            model_type: Type of model for logging
        """
        logger.info(f"Downloading model {repo_id} to {local_dir}")

        # Create parent directory
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        # Build hf download command
        cmd = ["huggingface-cli", "download", repo_id, "--local-dir", str(local_dir)]

        # Add token if provided
        if hf_token:
            cmd.extend(["--token", hf_token])

        # Set environment
        env = {}
        if hf_token:
            env["HF_TOKEN"] = hf_token

        # Download (this may take a long time)
        try:
            await run_subprocess(
                cmd,
                env=env,
                timeout=3600,  # 1 hour timeout
                log_output=True
            )
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            # Clean up partial download
            if local_dir.exists():
                import shutil
                shutil.rmtree(local_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to download model {repo_id}: {e}")


# Convenience function for the handler
async def download_assets_and_models(
    image_urls: List[str],
    video_urls: List[str],
    model_type: str,
    hf_token: Optional[str],
    workspace_dir: Path
) -> Tuple[Path, Path]:
    """
    Download both assets and models concurrently

    Args:
        image_urls: List of image URLs
        video_urls: List of video URLs
        model_type: Model type to download
        hf_token: Hugging Face token
        workspace_dir: Workspace directory for assets

    Returns:
        Tuple of (model_path, dataset_path)
    """
    logger.info("Starting parallel download of models and assets")

    # Create downloaders
    asset_downloader = AssetDownloader(workspace_dir)
    model_cacher = ModelCacher()

    # Download concurrently
    model_path, dataset_path = await asyncio.gather(
        model_cacher.get_or_download(model_type, hf_token),
        asset_downloader.download_batch(image_urls, video_urls)
    )

    logger.info(f"Downloads complete - Model: {model_path}, Dataset: {dataset_path}")
    return model_path, dataset_path


# Export
__all__ = ["AssetDownloader", "ModelCacher", "download_assets_and_models"]
