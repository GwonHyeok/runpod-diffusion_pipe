"""
Training manager for RunPod Serverless LoRA Training
Handles TOML configuration generation and training execution
"""

import logging
import toml
from pathlib import Path
from typing import Dict, Any

from config import MODEL_CONFIGS
from utils import run_subprocess, parse_training_metrics

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manage training configuration and execution"""

    def __init__(
        self,
        model_path: Path,
        model_type: str,
        dataset_path: Path,
        training_params: Dict[str, Any],
        workspace_dir: Path,
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.dataset_path = Path(dataset_path)
        self.training_params = training_params
        self.workspace_dir = Path(workspace_dir)

        # Directories
        self.config_dir = self.workspace_dir / "config"
        self.output_dir = self.workspace_dir / "output"
        self.diffusion_pipe_dir = Path("/diffusion_pipe")

        # Model config
        self.model_config = MODEL_CONFIGS[model_type]

        # Training metrics
        self.metrics = {}

    def prepare_config_files(self):
        """
        Prepare TOML configuration files for training

        1. Copy base TOML from toml_files/
        2. Update paths (dataset, output, model)
        3. Apply custom parameter overrides
        4. Generate dataset.toml
        """
        logger.info("Preparing training configuration files")

        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare model config TOML
        model_toml_path = self._prepare_model_toml()

        # Prepare dataset TOML
        dataset_toml_path = self._prepare_dataset_toml()

        logger.info(f"Configuration files ready:")
        logger.info(f"  Model config: {model_toml_path}")
        logger.info(f"  Dataset config: {dataset_toml_path}")

        return model_toml_path, dataset_toml_path

    def _prepare_model_toml(self) -> Path:
        """
        Prepare model-specific TOML configuration

        Returns:
            Path to generated TOML file
        """
        # Load base TOML template
        base_toml_path = (
            Path("/workspace") / "toml_files" / self.model_config["toml_file"]
        )

        if not base_toml_path.exists():
            # Fallback to repository location
            base_toml_path = (
                Path(__file__).parent.parent
                / "toml_files"
                / self.model_config["toml_file"]
            )

        if not base_toml_path.exists():
            raise FileNotFoundError(
                f"TOML template not found: {self.model_config['toml_file']}\n"
                f"Searched in:\n"
                f"  - /workspace/toml_files/\n"
                f"  - {Path(__file__).parent.parent}/toml_files/\n"
                f"Please ensure the toml_files directory is properly copied to the container."
            )

        logger.info(f"Loading base TOML: {base_toml_path}")

        with open(base_toml_path, "r") as f:
            config = toml.load(f)

        # Update output directory
        config["output_dir"] = str(self.output_dir / f"{self.model_type}_lora")

        # Update dataset path
        dataset_toml_path = self.config_dir / "dataset.toml"
        config["dataset"] = str(dataset_toml_path)

        # Update model paths
        if "model" in config:
            if "diffusers_path" in config["model"]:
                config["model"]["diffusers_path"] = str(self.model_path)
            elif "checkpoint_path" in config["model"]:
                config["model"]["checkpoint_path"] = str(self.model_path)
            elif "ckpt_path" in config["model"]:
                config["model"]["ckpt_path"] = str(self.model_path)

        # Apply training_params overrides to model TOML
        # These override the base TOML values if provided by user
        for key, value in self.training_params.items():
            # Skip dataset-specific params (handled in dataset.toml)
            if key in [
                "num_repeats",
                "video_num_repeats",
                "resolution",
                "enable_ar_bucket",
                "min_ar",
                "max_ar",
                "num_ar_buckets",
            ]:
                continue

            if "." in key:
                # Handle nested keys like "optimizer.lr"
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Top-level keys
                config[key] = value

        # Save to config directory
        output_path = self.config_dir / self.model_config["toml_file"]
        with open(output_path, "w") as f:
            toml.dump(config, f)

        logger.info(f"Model TOML configured with:")
        logger.info(f"  epochs={config.get('epochs')}")
        logger.info(f"  lr={config.get('lr')}")
        logger.info(f"  rank={config.get('adapter', {}).get('rank', 'N/A')}")
        logger.info(f"  output_dir={config['output_dir']}")

        return output_path

    def _prepare_dataset_toml(self) -> Path:
        """
        Prepare dataset.toml configuration

        Returns:
            Path to generated dataset.toml
        """
        logger.info("Preparing dataset.toml")

        # Load base dataset.toml template
        base_dataset_toml = Path("/workspace") / "dataset.toml"
        if not base_dataset_toml.exists():
            base_dataset_toml = Path(__file__).parent.parent / "dataset.toml"

        with open(base_dataset_toml, "r") as f:
            dataset_config = toml.load(f)

        # Apply dataset-specific training_params overrides
        if "resolution" in self.training_params:
            dataset_config["resolutions"] = [self.training_params["resolution"]]

        if "enable_ar_bucket" in self.training_params:
            dataset_config["enable_ar_bucket"] = self.training_params[
                "enable_ar_bucket"
            ]

        if "min_ar" in self.training_params:
            dataset_config["min_ar"] = self.training_params["min_ar"]

        if "max_ar" in self.training_params:
            dataset_config["max_ar"] = self.training_params["max_ar"]

        if "num_ar_buckets" in self.training_params:
            dataset_config["num_ar_buckets"] = self.training_params["num_ar_buckets"]

        # Extract default num_repeats from base TOML
        base_directories = dataset_config.get("directory", [])
        default_image_repeats = 1
        default_video_repeats = 5

        # Try to find defaults in base TOML
        for base_dir in base_directories:
            base_path = base_dir.get("path", "")
            if "image_dataset_here" in base_path:
                default_image_repeats = base_dir.get("num_repeats", 1)
            elif "video_dataset_here" in base_path:
                default_video_repeats = base_dir.get("num_repeats", 5)

        # Get user-provided num_repeats or use defaults
        image_repeats = self.training_params.get("num_repeats", default_image_repeats)
        video_repeats = self.training_params.get(
            "video_num_repeats", default_video_repeats
        )

        # Configure directories
        image_dir = self.dataset_path / "image_dataset_here"
        video_dir = self.dataset_path / "video_dataset_here"

        # Update directory paths
        directories = []

        # Add image directory if exists
        if image_dir.exists() and any(image_dir.iterdir()):
            directories.append({"path": str(image_dir), "num_repeats": image_repeats})
            logger.info(f"Added image dataset: {image_dir} (repeats={image_repeats})")

        # Add video directory if exists
        if video_dir.exists() and any(video_dir.iterdir()):
            directories.append({"path": str(video_dir), "num_repeats": video_repeats})
            logger.info(f"Added video dataset: {video_dir} (repeats={video_repeats})")

        if not directories:
            raise ValueError("No dataset directories found with files")

        dataset_config["directory"] = directories

        # Save dataset.toml
        output_path = self.config_dir / "dataset.toml"
        with open(output_path, "w") as f:
            toml.dump(dataset_config, f)

        logger.info(f"Dataset TOML configured with {len(directories)} directories")

        return output_path

    async def train(self) -> Path:
        """
        Execute LoRA training using DeepSpeed

        Returns:
            Path to output directory containing trained LoRA

        Raises:
            RuntimeError: If training fails
        """
        logger.info("=" * 80)
        logger.info("STARTING LORA TRAINING")
        logger.info("=" * 80)

        # Prepare config files
        model_toml_path, dataset_toml_path = self.prepare_config_files()

        # Update dependencies first
        logger.info("Updating training dependencies...")
        await self._update_dependencies()

        # Execute training
        logger.info(f"Starting training with model type: {self.model_type}")
        logger.info(f"Config: {model_toml_path}")

        try:
            result = await self._run_deepspeed_training(model_toml_path)

            # Parse metrics from output
            self.metrics = parse_training_metrics(result.stdout + result.stderr)

            logger.info("=" * 80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Metrics: {self.metrics}")

            # Find and return output directory
            output_dir = self.output_dir / f"{self.model_type}_lora"
            if not output_dir.exists():
                raise RuntimeError(f"Output directory not found: {output_dir}")

            # Log output files
            safetensors_files = list(output_dir.glob("**/*.safetensors"))
            logger.info(f"Generated {len(safetensors_files)} LoRA checkpoint(s)")
            for f in safetensors_files[:5]:  # Log first 5
                size_mb = f.stat().st_size / 1024**2
                logger.info(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")

            return output_dir

        except Exception as e:
            logger.error("=" * 80)
            logger.error("TRAINING FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            raise RuntimeError(f"Training failed: {e}")

    async def _update_dependencies(self):
        """Update training dependencies"""
        logger.info("Updating transformers and peft...")

        try:
            # Update transformers
            await run_subprocess(
                ["pip", "install", "transformers", "-U"], timeout=300, log_output=False
            )

            # Update peft
            await run_subprocess(
                ["pip", "install", "--upgrade", "peft>=0.17.0"],
                timeout=300,
                log_output=False,
            )

            logger.info("Dependencies updated successfully")

        except Exception as e:
            logger.warning(f"Dependency update failed (continuing anyway): {e}")

    async def _run_deepspeed_training(self, config_path: Path):
        """
        Run DeepSpeed training command

        This replicates the training command from interactive_start_training.sh line 1080:
        NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
          deepspeed --num_gpus=1 train.py --deepspeed --config <config>

        Args:
            config_path: Path to TOML config file

        Returns:
            subprocess.CompletedProcess with training output
        """
        logger.info("Launching DeepSpeed training...")

        # Build command
        cmd = [
            "deepspeed",
            "--num_gpus=1",
            "train.py",
            "--deepspeed",
            "--config",
            str(config_path),
        ]

        # Set environment variables
        env = {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        }

        # Run training
        result = await run_subprocess(
            cmd,
            env=env,
            cwd=self.diffusion_pipe_dir,
            timeout=None,  # No timeout for training (can take hours)
            log_output=True,
        )

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return self.metrics

    def get_output_files(self) -> list:
        """Get list of output LoRA files"""
        output_dir = self.output_dir / f"{self.model_type}_lora"
        if not output_dir.exists():
            return []

        return [
            str(f.relative_to(output_dir)) for f in output_dir.glob("**/*.safetensors")
        ]


# Export
__all__ = ["TrainingManager"]
