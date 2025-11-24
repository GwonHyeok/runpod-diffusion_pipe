"""
RunPod Serverless Handler for LoRA Training
Main entry point for async training pipeline
"""

import runpod
import logging
import traceback
import time
import shutil
from pathlib import Path
from typing import Dict, Any

# Import our modules
from config import TrainingConfig
from downloader import download_assets_and_models
from caption_manager import CaptionManager
from training_manager import TrainingManager
from utils import validate_cuda, upload_results, setup_logging, retry

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


@retry(
    max_attempts=1, backoff=1.0
)  # No retry at top level (handled by individual components)
async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod Serverless async handler

    Expected input format:
    {
        "input": {
            "model_type": "flux",
            "image_urls": ["https://...", ...],
            "video_urls": ["https://...", ...],
            "image_caption_urls": ["https://...", ...],  # Optional: pre-made caption .txt files
            "video_caption_urls": ["https://...", ...],  # Optional: pre-made caption .txt files
            "caption_mode": "images" | "videos" | "both" | "skip",
            "trigger_word": "alice",
            "caption_prompt": "custom prompt...",
            "hf_token": "hf_xxxxx",
            "gemini_api_key": "AIzaSyxxxx",
            "training_params": {
                "epochs": 100,
                "lr": 2e-5,
                "rank": 32,
                ...
            },
            "output_bucket": "runpod-volume"
        }
    }

    Returns:
    {
        "status": "success" | "failed",
        "output": {
            "download_urls": ["https://...", ...],
            "metrics": {...},
            "files": [...]
        },
        "error": "error message if failed",
        "execution_time": 1234.56
    }
    """
    start_time = time.time()
    job_id = job.get("id", "unknown")
    workspace_dir = None
    workspace_cleanup_done = False  # Track if cleanup already happened

    logger.info("=" * 80)
    logger.info(f"JOB STARTED: {job_id}")
    logger.info("=" * 80)

    try:
        # 1. Parse and validate configuration
        logger.info("Step 1: Parsing configuration...")
        input_data = job.get("input", {})

        try:
            config = TrainingConfig.from_request(input_data)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                "status": "failed",
                "error": f"Invalid configuration: {str(e)}",
                "execution_time": time.time() - start_time,
            }

        logger.info(f"Configuration validated:")
        logger.info(f"  Model: {config.model_type}")
        logger.info(f"  Images: {len(config.image_urls)}")
        logger.info(f"  Videos: {len(config.video_urls)}")
        logger.info(f"  Image captions: {len(config.image_caption_urls)}")
        logger.info(f"  Video captions: {len(config.video_caption_urls)}")
        logger.info(f"  Caption mode: {config.caption_mode}")
        logger.info(f"  Epochs: {config.training_params.get('epochs')}")

        # 2. CUDA validation (fail fast)
        logger.info("Step 2: Validating CUDA...")
        validate_cuda()

        # 3. Setup workspace
        workspace_dir = Path(f"/tmp/runpod-job-{job_id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Workspace: {workspace_dir}")

        # 4. Download assets and models in parallel
        logger.info("Step 3: Downloading assets and models...")
        model_path, dataset_path = await download_assets_and_models(
            image_urls=config.image_urls,
            video_urls=config.video_urls,
            model_type=config.model_type,
            hf_token=config.hf_token,
            workspace_dir=workspace_dir,
            image_caption_urls=config.image_caption_urls,
            video_caption_urls=config.video_caption_urls,
        )

        logger.info(f"Downloads complete:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Dataset: {dataset_path}")

        # 5. Run captioning (if needed)
        caption_results = {}
        if config.needs_image_captioning() or config.needs_video_captioning():
            logger.info("Step 4: Running captioning...")

            caption_manager = CaptionManager(
                trigger_word=config.trigger_word,
                caption_prompt=config.caption_prompt,
                gemini_api_key=config.gemini_api_key,
            )

            # Only caption directories that need it (no provided caption URLs)
            image_dir = None
            video_dir = None

            if config.needs_image_captioning():
                image_dir = dataset_path / "image_dataset_here"

            if config.needs_video_captioning():
                video_dir = dataset_path / "video_dataset_here"

            caption_results = await caption_manager.run(
                image_dir=image_dir, video_dir=video_dir
            )

            logger.info(f"Captioning complete:")
            logger.info(
                f"  Images captioned: {caption_results.get('images_captioned', 0)}"
            )
            logger.info(
                f"  Videos captioned: {caption_results.get('videos_captioned', 0)}"
            )

            if caption_results.get("errors"):
                logger.warning(f"  Captioning errors: {len(caption_results['errors'])}")
        else:
            skip_reason = (
                "caption files provided"
                if (config.has_image_captions() or config.has_video_captions())
                else "mode='skip'"
            )
            logger.info(f"Step 4: Skipping captioning ({skip_reason})")

        # 6. Setup and run training
        logger.info("Step 5: Setting up training...")

        training_manager = TrainingManager(
            model_path=model_path,
            model_type=config.model_type,
            dataset_path=dataset_path,
            training_params=config.training_params,
            workspace_dir=workspace_dir,
        )

        logger.info("Step 6: Running training...")
        output_dir = await training_manager.train()

        # Get training metrics
        metrics = training_manager.get_metrics()
        output_files = training_manager.get_output_files()

        logger.info(f"Training complete:")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Files generated: {len(output_files)}")
        logger.info(f"  Metrics: {metrics}")

        # 7. Upload results (BEFORE cleanup in finally block)
        logger.info("Step 7: Uploading results...")

        upload_urls = await upload_results(
            output_dir=output_dir,
            job_id=job_id,
            file_patterns=["**/*.safetensors", "**/*.log"],
        )

        logger.info(f"Upload complete: {len(upload_urls)} files uploaded")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Success response
        logger.info("=" * 80)
        logger.info(f"JOB COMPLETED SUCCESSFULLY: {job_id}")
        logger.info(
            f"Total execution time: {execution_time:.1f}s ({execution_time/60:.1f} minutes)"
        )
        logger.info("=" * 80)

        # Mark workspace for cleanup (set flag to skip cleanup in finally)
        # This ensures upload completes before cleanup
        workspace_cleanup_done = True

        return {
            "status": "success",
            "output": {
                "download_urls": upload_urls,
                "metrics": metrics,
                "files": output_files,
                "caption_results": caption_results,
                "model_type": config.model_type,
                "epochs_completed": metrics.get("epochs_completed", 0),
            },
            "execution_time": execution_time,
        }

    except Exception as e:
        # Error handling
        execution_time = time.time() - start_time
        error_trace = traceback.format_exc()

        logger.error("=" * 80)
        logger.error(f"JOB FAILED: {job_id}")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Traceback:\n{error_trace}")

        return {
            "status": "failed",
            "error": str(e),
            "traceback": error_trace,
            "execution_time": execution_time,
        }

    finally:
        # Cleanup workspace (only if not already done in success path)
        if not workspace_cleanup_done and workspace_dir and workspace_dir.exists():
            try:
                logger.info(f"Cleaning up workspace: {workspace_dir}")
                shutil.rmtree(workspace_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Workspace cleanup failed: {e}")

        # Free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")


# Start RunPod serverless
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless LoRA Training Handler")
    logger.info("Ready to accept jobs...")

    runpod.serverless.start({"handler": handler})
