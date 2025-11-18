"""
Caption manager for RunPod Serverless LoRA Training
Orchestrates image and video captioning using JoyCaption and Gemini API
"""

import asyncio
import logging
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

from utils import run_subprocess, retry

logger = logging.getLogger(__name__)


class CaptionManager:
    """Orchestrate image and video captioning"""

    def __init__(
        self,
        trigger_word: Optional[str] = None,
        caption_prompt: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.trigger_word = trigger_word
        self.caption_prompt = caption_prompt or (
            "Write a detailed description for this image in 50 words or less. "
            "Do NOT mention any text that is in the image."
        )
        self.gemini_api_key = gemini_api_key

    async def run(
        self,
        image_dir: Optional[Path] = None,
        video_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run captioning for images and/or videos

        Args:
            image_dir: Directory containing images (if None, skip image captioning)
            video_dir: Directory containing videos (if None, skip video captioning)

        Returns:
            Dictionary with captioning results:
            {
                "images_captioned": 10,
                "videos_captioned": 5,
                "errors": []
            }
        """
        logger.info("Starting captioning process")

        tasks = []
        results = {
            "images_captioned": 0,
            "videos_captioned": 0,
            "errors": []
        }

        # Queue captioning tasks
        if image_dir and image_dir.exists():
            tasks.append(self._caption_images(image_dir))

        if video_dir and video_dir.exists():
            if not self.gemini_api_key:
                error_msg = "Video captioning requested but gemini_api_key not provided"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            else:
                tasks.append(self._caption_videos(video_dir))

        if not tasks:
            logger.warning("No captioning tasks to run")
            return results

        # Run captioning tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                error_msg = f"Captioning task {i} failed: {result}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            elif isinstance(result, dict):
                results["images_captioned"] += result.get("images_captioned", 0)
                results["videos_captioned"] += result.get("videos_captioned", 0)
                results["errors"].extend(result.get("errors", []))

        logger.info(
            f"Captioning complete: {results['images_captioned']} images, "
            f"{results['videos_captioned']} videos"
        )

        if results["errors"]:
            logger.warning(f"Encountered {len(results['errors'])} errors during captioning")

        return results

    @retry(max_attempts=2, backoff=1.0)
    async def _caption_images(self, image_dir: Path) -> Dict[str, Any]:
        """
        Caption images using JoyCaption model

        This directly uses the existing joy_caption_batch.py script
        by importing and calling it with proper parameters

        Args:
            image_dir: Directory containing images to caption

        Returns:
            Dictionary with results
        """
        logger.info(f"Captioning images in {image_dir}")

        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not images:
            logger.warning(f"No images found in {image_dir}")
            return {"images_captioned": 0, "errors": []}

        logger.info(f"Found {len(images)} images to caption")

        try:
            # Import the existing JoyCaption script
            # We'll use the JoyCaptionManager class directly
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "Captioning" / "JoyCaption"))

            from joy_caption_batch import JoyCaptionManager

            # Create caption manager
            captioner = JoyCaptionManager(timeout_minutes=60)

            # Load model
            captioner.load_model()

            # Process images
            captioned_count = 0
            errors = []

            for image_path in images:
                try:
                    # Check if caption already exists
                    caption_path = image_path.with_suffix('.txt')
                    if caption_path.exists():
                        logger.debug(f"Caption already exists for {image_path.name}, skipping")
                        continue

                    # Load image and generate caption
                    with Image.open(image_path) as img:
                        caption = captioner.generate_caption(img, self.caption_prompt)

                    # Add trigger word if specified
                    if self.trigger_word:
                        caption = f"{self.trigger_word}, {caption}"

                    # Save caption
                    caption_path.write_text(caption, encoding='utf-8')
                    captioned_count += 1
                    logger.info(f"Captioned {image_path.name}: {caption[:50]}...")

                except Exception as e:
                    error_msg = f"Failed to caption {image_path.name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Unload model to free GPU memory
            captioner.unload_model()

            return {
                "images_captioned": captioned_count,
                "errors": errors
            }

        except Exception as e:
            error_msg = f"Image captioning failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(max_attempts=2, backoff=1.0)
    async def _caption_videos(self, video_dir: Path) -> Dict[str, Any]:
        """
        Caption videos using Gemini API via TripleX

        This uses the existing video_captioner.sh approach but orchestrated in Python

        Args:
            video_dir: Directory containing videos to caption

        Returns:
            Dictionary with results
        """
        logger.info(f"Captioning videos in {video_dir}")

        # Count videos
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

        if not videos:
            logger.warning(f"No videos found in {video_dir}")
            return {"videos_captioned": 0, "errors": []}

        logger.info(f"Found {len(videos)} videos to caption")

        try:
            # Setup TripleX environment (if not already done)
            triplex_dir = Path("/tmp/TripleX")
            gemini_script = triplex_dir / "captioners" / "gemini.py"

            if not gemini_script.exists():
                logger.info("Setting up TripleX repository...")
                # Clean up any partial installation
                if triplex_dir.exists():
                    shutil.rmtree(triplex_dir, ignore_errors=True)

                logger.info("Cloning TripleX repository...")
                await run_subprocess(
                    ["git", "clone", "https://github.com/Hearmeman24/TripleX.git", str(triplex_dir)],
                    timeout=300
                )

            # Setup conda environment (if not already done)
            conda_prefix = Path("/tmp/TripleX_miniconda")
            conda_env_path = conda_prefix / "envs" / "TripleX"

            if not conda_env_path.exists():
                logger.info("Setting up Miniconda environment for TripleX...")
                await self._setup_triplex_conda(triplex_dir, conda_prefix)

            # Run Gemini captioner
            logger.info("Running Gemini video captioner...")

            env = {
                "GEMINI_API_KEY": self.gemini_api_key,
                "PATH": f"{conda_env_path}/bin:/usr/local/bin:/usr/bin:/bin"
            }

            # Use conda run to execute in the correct environment
            cmd = [
                "conda", "run",
                "-p", str(conda_env_path),
                "--no-capture-output",
                "python",
                str(triplex_dir / "captioners" / "gemini.py"),
                "--dir", str(video_dir),
                "--max_frames", "1"
            ]

            result = await run_subprocess(
                cmd,
                env=env,
                timeout=7200,  # 2 hour timeout for video captioning
                log_output=True
            )

            # Count captioned videos
            captioned_count = sum(
                1 for v in videos
                if v.with_suffix('.txt').exists()
            )

            logger.info(f"Video captioning complete: {captioned_count}/{len(videos)} captioned")

            return {
                "videos_captioned": captioned_count,
                "errors": []
            }

        except Exception as e:
            error_msg = f"Video captioning failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def _setup_triplex_conda(self, triplex_dir: Path, conda_prefix: Path):
        """
        Setup Miniconda and TripleX environment

        Args:
            triplex_dir: TripleX repository directory
            conda_prefix: Conda installation prefix
        """
        logger.info("Installing Miniconda...")

        # Download and install Miniconda
        miniconda_installer = "/tmp/miniconda.sh"

        # Download Miniconda installer
        await run_subprocess(
            [
                "wget",
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
                "-O", miniconda_installer
            ],
            timeout=300
        )

        # Install Miniconda
        await run_subprocess(
            ["bash", miniconda_installer, "-b", "-p", str(conda_prefix)],
            timeout=300
        )

        # Create conda environment
        logger.info("Creating TripleX conda environment...")

        conda_bin = conda_prefix / "bin" / "conda"

        await run_subprocess(
            [
                str(conda_bin), "create",
                "-p", str(conda_prefix / "envs" / "TripleX"),
                "python=3.10",
                "-y"
            ],
            timeout=600
        )

        # Install requirements
        logger.info("Installing TripleX requirements...")

        pip_bin = conda_prefix / "envs" / "TripleX" / "bin" / "pip"

        await run_subprocess(
            [
                str(pip_bin), "install",
                "-r", str(triplex_dir / "requirements.txt")
            ],
            timeout=600
        )

        logger.info("TripleX environment setup complete")


# Export
__all__ = ["CaptionManager"]
