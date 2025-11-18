"""
Utility functions for RunPod Serverless LoRA Training
Includes retry logic, CUDA validation, subprocess helpers, and upload utilities
"""

import asyncio
import subprocess
import logging
import traceback
import time
from functools import wraps
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import torch

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3, backoff: float = 2.0, exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        backoff: Base for exponential backoff (seconds)
        exceptions: Tuple of exceptions to catch and retry

    Example:
        @retry(max_attempts=3, backoff=2.0)
        async def download_file(url):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    wait_time = backoff**attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    wait_time = backoff**attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_cuda():
    """
    CUDA compatibility check (from interactive_start_training.sh lines 299-349)
    Raises RuntimeError if CUDA is unavailable or incompatible

    This performs the same checks as the bash script to ensure CUDA 12.8 compatibility
    """
    logger.info("Validating CUDA availability and compatibility...")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This serverless function requires NVIDIA GPU with CUDA 12.8. "
            "Please deploy with GPU-enabled RunPod instance."
        )

    # Get CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA version: {cuda_version}")

    # Test kernel compatibility (similar to interactive_start_training.sh line 334-341)
    try:
        # Simple GPU operation to verify kernel compatibility
        x = torch.randn(100, 100, device="cuda")
        y = x * 2
        result = y.sum().item()
        del x, y
        torch.cuda.empty_cache()

        logger.info(f"CUDA test passed. GPU: {torch.cuda.get_device_name(0)}")

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "no kernel image" in error_msg or "cuda error" in error_msg:
            raise RuntimeError(
                f"CUDA kernel compatibility error: {e}\n"
                "This usually means the GPU architecture is not compatible with CUDA 12.8.\n"
                "Recommended GPUs: H100, H200, or other Ada/Hopper architecture GPUs."
            )
        raise RuntimeError(f"CUDA test failed: {e}")

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU validated: {gpu_name} ({gpu_memory:.1f} GB)")


async def run_subprocess(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    log_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run subprocess asynchronously with proper error handling

    Args:
        cmd: Command and arguments as list
        env: Environment variables (merged with os.environ)
        cwd: Working directory
        timeout: Timeout in seconds
        log_output: Whether to log stdout/stderr

    Returns:
        CompletedProcess with returncode, stdout, stderr

    Raises:
        subprocess.CalledProcessError: If command returns non-zero exit code
        asyncio.TimeoutError: If command exceeds timeout
    """
    import os

    # Merge environment variables
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    cmd_str = " ".join(cmd)
    logger.info(f"Running subprocess: {cmd_str}")
    if cwd:
        logger.info(f"Working directory: {cwd}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            cwd=str(cwd) if cwd else None,
        )

        # Wait for process with optional timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        if log_output:
            if stdout_str:
                logger.info(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                logger.info(f"STDERR:\n{stderr_str}")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, output=stdout_str, stderr=stderr_str
            )

        return subprocess.CompletedProcess(
            cmd, process.returncode, stdout_str, stderr_str
        )

    except asyncio.TimeoutError:
        logger.error(f"Subprocess timeout after {timeout}s: {cmd_str}")
        try:
            process.kill()
            await process.wait()
        except:
            pass
        raise asyncio.TimeoutError(f"Command timed out after {timeout}s: {cmd_str}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess failed with exit code {e.returncode}: {cmd_str}")
        logger.error(f"STDERR: {e.stderr}")
        raise

    except Exception as e:
        logger.error(f"Subprocess error: {e}\n{traceback.format_exc()}")
        raise


async def upload_results(
    output_dir: Path, job_id: str, file_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Upload training results using configured storage backend (GCP or RunPod)

    Storage backend is determined by environment variables:
    - If GCP_SERVICE_ACCOUNT_JSON and GCS_BUCKET_NAME are set, uses GCP Storage
    - Otherwise, falls back to RunPod's rp_upload utility

    Args:
        output_dir: Directory containing training outputs
        job_id: Job ID for organizing uploads
        file_patterns: List of glob patterns to match (default: ["*.safetensors", "*.log"])

    Returns:
        List of accessible URLs for uploaded files (signed URLs with 7-day expiration for GCP)

    Raises:
        RuntimeError: If upload fails after retries
    """
    # Import uploader factory
    from uploaders import create_uploader

    if file_patterns is None:
        file_patterns = ["**/*.safetensors", "**/*.log", "**/checkpoint-*"]

    logger.info(f"Uploading results from {output_dir}")

    # Validate output directory exists
    if not output_dir.exists():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")

    # Collect files to upload
    files_to_upload = []
    for pattern in file_patterns:
        files_to_upload.extend(output_dir.glob(pattern))

    # Filter out directories
    files_to_upload = [f for f in files_to_upload if f.is_file()]

    if not files_to_upload:
        logger.warning(f"No files found to upload in {output_dir}")
        return []

    logger.info(f"Found {len(files_to_upload)} files to upload")

    # Create appropriate uploader based on environment
    uploader = create_uploader(job_id)

    # Upload files and return URLs
    try:
        upload_urls = await uploader.upload(files_to_upload, job_id)
        logger.info(f"Upload complete: {len(upload_urls)} files uploaded")
        return upload_urls

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


def setup_logging(level: str = "INFO"):
    """
    Configure logging for the serverless handler

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_training_metrics(log_output: str) -> Dict[str, Any]:
    """
    Parse training metrics from training log output

    Args:
        log_output: Training stdout/stderr

    Returns:
        Dictionary containing parsed metrics (loss, epoch, steps, etc.)
    """
    metrics = {"final_loss": None, "epochs_completed": 0, "total_steps": 0}

    # Parse common patterns from training logs
    lines = log_output.split("\n")
    for line in lines:
        # Look for epoch completion
        if "epoch" in line.lower() and "/" in line:
            try:
                # Example: "Epoch 50/80"
                parts = line.split("/")
                if len(parts) >= 2:
                    metrics["epochs_completed"] = max(
                        metrics["epochs_completed"], int(parts[0].split()[-1])
                    )
            except (ValueError, IndexError):
                pass

        # Look for loss values
        if "loss" in line.lower():
            try:
                # Example: "loss: 0.0234"
                for part in line.split():
                    if part.replace(".", "").replace("-", "").isdigit():
                        loss_val = float(part)
                        if 0 < loss_val < 10:  # Reasonable loss range
                            metrics["final_loss"] = loss_val
            except ValueError:
                pass

    logger.info(f"Parsed metrics: {metrics}")
    return metrics


# Export commonly used functions
__all__ = [
    "retry",
    "validate_cuda",
    "run_subprocess",
    "upload_results",
    "setup_logging",
    "parse_training_metrics",
]
