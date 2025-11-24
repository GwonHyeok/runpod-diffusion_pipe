"""
Configuration management for RunPod Serverless LoRA Training
Handles request validation and default values
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import logging

logger = logging.getLogger(__name__)

# Model type definitions
MODEL_TYPE = Literal["flux", "sdxl", "wan13", "wan14b_t2v", "wan14b_i2v", "qwen"]
CAPTION_MODE = Literal["images", "videos", "both", "skip"]

# Model configurations
MODEL_CONFIGS = {
    "flux": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "requires_hf_token": True,
        "toml_file": "flux.toml",
        "path_in_volume": "flux",
    },
    "sdxl": {
        "repo": "timoshishi/sdXL_v10VAEFix",
        "requires_hf_token": False,
        "toml_file": "sdxl.toml",
        "path_in_volume": "sdXL_v10VAEFix.safetensors",
    },
    "wan13": {
        "repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "requires_hf_token": False,
        "toml_file": "wan13_video.toml",
        "path_in_volume": "Wan/Wan2.1-T2V-1.3B",
    },
    "wan14b_t2v": {
        "repo": "Wan-AI/Wan2.1-T2V-14B",
        "requires_hf_token": False,
        "toml_file": "wan14b_t2v.toml",
        "path_in_volume": "Wan/Wan2.1-T2V-14B",
    },
    "wan14b_i2v": {
        "repo": "Wan-AI/Wan2.1-I2V-14B-480P",
        "requires_hf_token": False,
        "toml_file": "wan14b_i2v.toml",
        "path_in_volume": "Wan/Wan2.1-I2V-14B-480P",
    },
    "qwen": {
        "repo": "Qwen/Qwen-Image",
        "requires_hf_token": False,
        "toml_file": "qwen_toml.toml",
        "path_in_volume": "Qwen-Image",
    },
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "epochs": 80,
    "micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 4,
    "save_every_n_epochs": 10,
    "checkpoint_every_n_minutes": 120,
    "warmup_steps": 100,
    "gradient_clipping": 1.0,
    "activation_checkpointing": True,
    "lr": 2e-4,
    "rank": 32,
    "optimizer_type": "adamw_optimi",
    "weight_decay": 0.01,
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "resolution": 1024,
    "enable_ar_bucket": True,
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 7,
    "num_repeats": 1,
}


@dataclass
class TrainingConfig:
    """Complete training configuration from user request"""

    # Required fields
    model_type: str

    # Dataset URLs
    image_urls: List[str] = field(default_factory=list)
    video_urls: List[str] = field(default_factory=list)

    # Caption file URLs (optional - if provided, skip auto-captioning)
    image_caption_urls: List[str] = field(default_factory=list)
    video_caption_urls: List[str] = field(default_factory=list)

    # Captioning configuration
    caption_mode: str = "skip"
    trigger_word: Optional[str] = None
    caption_prompt: str = (
        "Write a detailed description for this image in 50 words or less. Do NOT mention any text that is in the image."
    )

    # API Keys
    hf_token: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Training parameters (merged with defaults)
    training_params: Dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output_bucket: str = "runpod-volume"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()
        self._merge_defaults()

    def _validate(self):
        """Validate configuration constraints"""

        # Validate model type
        if self.model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {list(MODEL_CONFIGS.keys())}"
            )

        # Validate caption mode
        valid_modes = ["images", "videos", "both", "skip"]
        if self.caption_mode not in valid_modes:
            raise ValueError(
                f"Invalid caption_mode '{self.caption_mode}'. "
                f"Must be one of: {valid_modes}"
            )

        # Validate dataset exists
        if not self.image_urls and not self.video_urls:
            raise ValueError(
                "At least one of image_urls or video_urls must be provided"
            )

        # Validate API keys based on model and caption mode
        model_config = MODEL_CONFIGS[self.model_type]

        if model_config["requires_hf_token"] and not self.hf_token:
            raise ValueError(
                f"Model '{self.model_type}' requires hf_token (Hugging Face token)"
            )

        if self.caption_mode in ["videos", "both"] and not self.gemini_api_key:
            raise ValueError("Video captioning requires gemini_api_key")

        # Validate URLs format
        all_urls = (
            self.image_urls
            + self.video_urls
            + self.image_caption_urls
            + self.video_caption_urls
        )
        for url in all_urls:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL: {url}")

        # Validate caption mode matches dataset
        if self.caption_mode == "images" and not self.image_urls:
            raise ValueError("caption_mode 'images' requires image_urls")
        if self.caption_mode == "videos" and not self.video_urls:
            raise ValueError("caption_mode 'videos' requires video_urls")

        logger.info(
            f"Configuration validated: model={self.model_type}, "
            f"images={len(self.image_urls)}, videos={len(self.video_urls)}, "
            f"caption_mode={self.caption_mode}"
        )

    def _merge_defaults(self):
        """Merge user params with defaults"""
        merged = DEFAULT_TRAINING_PARAMS.copy()
        merged.update(self.training_params)
        self.training_params = merged

        logger.info(
            f"Training params: epochs={self.training_params['epochs']}, "
            f"lr={self.training_params['lr']}, rank={self.training_params['rank']}"
        )

    @classmethod
    def from_request(cls, input_data: Dict[str, Any]) -> "TrainingConfig":
        """
        Parse and validate request JSON

        Expected input format:
        {
            "model_type": "flux",
            "image_urls": ["https://...", ...],
            "video_urls": ["https://...", ...],
            "image_caption_urls": ["https://...", ...],  # Optional: caption .txt files
            "video_caption_urls": ["https://...", ...],  # Optional: caption .txt files
            "caption_mode": "images",
            "trigger_word": "alice",
            "hf_token": "hf_xxxxx",
            "gemini_api_key": "AIzaSyxxxx",
            "training_params": {
                "epochs": 100,
                "lr": 2e-5,
                ...
            }
        }
        """
        try:
            config = cls(**input_data)
            return config
        except TypeError as e:
            raise ValueError(f"Invalid request format: {e}")

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return MODEL_CONFIGS[self.model_type]

    def has_images(self) -> bool:
        """Check if image dataset exists"""
        return len(self.image_urls) > 0

    def has_videos(self) -> bool:
        """Check if video dataset exists"""
        return len(self.video_urls) > 0

    def has_image_captions(self) -> bool:
        """Check if image caption URLs are provided"""
        return len(self.image_caption_urls) > 0

    def has_video_captions(self) -> bool:
        """Check if video caption URLs are provided"""
        return len(self.video_caption_urls) > 0

    def needs_image_captioning(self) -> bool:
        """Check if image captioning is needed"""
        # Skip captioning if caption files are provided
        if self.has_image_captions():
            return False
        return self.caption_mode in ["images", "both"] and self.has_images()

    def needs_video_captioning(self) -> bool:
        """Check if video captioning is needed"""
        # Skip captioning if caption files are provided
        if self.has_video_captions():
            return False
        return self.caption_mode in ["videos", "both"] and self.has_videos()


# Export for easy imports
__all__ = ["TrainingConfig", "MODEL_CONFIGS", "DEFAULT_TRAINING_PARAMS"]
