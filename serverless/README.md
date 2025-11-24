# RunPod Serverless LoRA Training

Async handler for training LoRA (Low-Rank Adaptation) models in RunPod Serverless environment.

## Features

- **Async Execution**: Async handler for long-running training jobs
- **Full Customization**: Control all TOML parameters via JSON
- **Automatic Captioning**: JoyCaption (images) + Gemini API (videos)
- **Model Caching**: Save and reuse models in `/runpod-volume`
- **Retry Logic**: 3 retries on network errors
- **URL Inputs**: Download images/videos from URLs
- **Pre-signed URL Outputs**: Upload results via `rp upload`

## Supported Models

- **Flux** (black-forest-labs/FLUX.1-dev) - Requires HF Token
- **SDXL** (Stable Diffusion XL)
- **Wan 1.3B** (Text-to-Video)
- **Wan 14B T2V** (Text-to-Video 14B)
- **Wan 14B I2V** (Image-to-Video 14B)
- **Qwen Image**

## Architecture

```
handler.py (entry point)
    ↓
├── config.py (configuration validation)
├── downloader.py (model + dataset downloads)
├── caption_manager.py (JoyCaption + Gemini)
├── training_manager.py (TOML generation + DeepSpeed execution)
└── utils.py (Retry, CUDA, Upload)
```

## Deployment

### 1. Build Docker Image

```bash
cd /Users/ghyeok/Desktop/runpod-diffusion_pipe
docker build -f Dockerfile.serverless -t your-registry/runpod-lora-training:latest .
docker push your-registry/runpod-lora-training:latest
```

### 2. Create RunPod Serverless Endpoint

On the RunPod website:

1. **Serverless** → **New Endpoint**
2. **Container Image**: `your-registry/runpod-lora-training:latest`
3. **GPU Type**: H100 or H200 recommended
4. **Min Workers**: 0 (allow cold starts)
5. **Max Workers**: Number of concurrent jobs needed
6. **Timeout**: 24 hours (for long training)
7. **Network Volume**: Create and mount (`/runpod-volume`)
8. **Environment Variables** (Optional):

   **Storage Configuration:**

   - `GCP_SERVICE_ACCOUNT_JSON`: Service account JSON credentials (Base64-encoded recommended)
   - `GCS_BUCKET_NAME`: GCS bucket name for storing trained models
   - If these are not set, the system falls back to RunPod's native `rp_upload` utility

   **Logging Configuration:**

   - `JOY_CAPTION_USE_LOG_FILE`: Enable file logging for JoyCaption (default: `false`)
     - Set to `true`, `1`, or `yes` to enable logging to `${NETWORK_VOLUME}/logs/joy_caption_batch.log`
     - Requires `NETWORK_VOLUME` environment variable to be set
     - If not set or `false`, logs only to stdout (recommended for serverless)

### 3. API Call

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Create job
job = runpod.Endpoint("ENDPOINT_ID").run({
    "input": {
        "model_type": "flux",
        "image_urls": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg"
        ],
        "caption_mode": "images",
        "trigger_word": "alice",
        "hf_token": "hf_xxxxxxxxxxxxx",
        "training_params": {
            "epochs": 50,
            "lr": 2e-4,
            "rank": 32,
            "save_every_n_epochs": 10
        }
    }
})

# Check job status
status = runpod.Endpoint("ENDPOINT_ID").status(job["id"])
print(status)

# Get result
result = runpod.Endpoint("ENDPOINT_ID").result(job["id"])
print(result["output"]["download_urls"])
```

## Input Schema

```json
{
  "input": {
    "model_type": "flux | sdxl | wan13 | wan14b_t2v | wan14b_i2v | qwen",

    "image_urls": ["https://...", "https://..."],
    "video_urls": ["https://...", "https://..."],

    "image_caption_urls": ["https://...", "https://..."],
    "video_caption_urls": ["https://...", "https://..."],

    "caption_mode": "images | videos | both | skip",
    "trigger_word": "optional_trigger_word",
    "caption_prompt": "Custom caption prompt...",

    "hf_token": "hf_xxxxx",
    "gemini_api_key": "AIzaSyxxxx",

    "training_params": {
      "epochs": 80,
      "micro_batch_size_per_gpu": 1,
      "gradient_accumulation_steps": 4,
      "save_every_n_epochs": 10,
      "lr": 2e-4,
      "rank": 32,
      "optimizer_type": "adamw_optimi",
      "weight_decay": 0.01,
      "resolution": 1024,
      "enable_ar_bucket": true,
      "min_ar": 0.5,
      "max_ar": 2.0,
      "num_ar_buckets": 7,
      "num_repeats": 1
    }
  }
}
```

### Required Fields

- `model_type`: Model type to train
- `image_urls` or `video_urls`: At least one required

### Conditionally Required Fields

- `hf_token`: Required when using Flux model
- `gemini_api_key`: Required when using video captioning

### Optional Fields

- `image_caption_urls`: Pre-made caption `.txt` files for images (skips auto-captioning)
- `video_caption_urls`: Pre-made caption `.txt` files for videos (skips auto-captioning)
- `caption_mode`: Default `"skip"`
- `trigger_word`: Word to prepend to image captions
- `training_params`: Filled with defaults (can override partially)

### Using Pre-made Caption Files

Instead of auto-generating captions, you can provide your own `.txt` caption files:

**Example:**

```json
{
  "input": {
    "model_type": "flux",
    "image_urls": [
      "https://example.com/photo1.jpg",
      "https://example.com/photo2.jpg"
    ],
    "image_caption_urls": [
      "https://example.com/caption1.txt",
      "https://example.com/caption2.txt"
    ],
    "caption_mode": "skip",
    "hf_token": "hf_xxxxx"
  }
}
```

**Important Notes:**

- Caption file order must match image/video URL order (first caption → first image, etc.)
- Caption files are automatically named as `image_0000.txt`, `image_0001.txt`, etc.
- When caption URLs are provided, automatic captioning is skipped for that dataset
- Each `.txt` file should contain plain text description
- You can mix: provide captions for images and auto-generate for videos (or vice versa)

## Output Schema

### On Success

```json
{
  "status": "success",
  "output": {
    "download_urls": [
      "https://runpod-storage.../epoch10.safetensors?presigned=...",
      "https://runpod-storage.../epoch20.safetensors?presigned=..."
    ],
    "metrics": {
      "final_loss": 0.023,
      "epochs_completed": 50
    },
    "files": [
      "epoch10/adapter_model.safetensors",
      "epoch20/adapter_model.safetensors"
    ],
    "caption_results": {
      "images_captioned": 10,
      "videos_captioned": 0,
      "errors": []
    },
    "model_type": "flux",
    "epochs_completed": 50
  },
  "execution_time": 3600.5
}
```

### On Failure

```json
{
  "status": "failed",
  "error": "Error message",
  "traceback": "Full Python traceback...",
  "execution_time": 123.45
}
```

## Local Testing

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Create test job
cat > test_job.json << 'EOF'
{
  "id": "test-job-001",
  "input": {
    "model_type": "sdxl",
    "image_urls": [
      "https://picsum.photos/512/512?random=1",
      "https://picsum.photos/512/512?random=2"
    ],
    "caption_mode": "skip",
    "training_params": {
      "epochs": 2,
      "save_every_n_epochs": 1
    }
  }
}
EOF

# Run handler
cd /Users/ghyeok/Desktop/runpod-diffusion_pipe/serverless
python -c "
import json
from handler import handler

with open('test_job.json') as f:
    job = json.load(f)

result = handler(job)
print(json.dumps(result, indent=2))
"
```

## Cost Optimization

### 1. Model Caching

Pre-download models to Network Volume:

```bash
# Run on RunPod Pod
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /runpod-volume/models/flux \
  --token YOUR_HF_TOKEN
```

Subsequent serverless jobs will use cache without downloading.

### 2. Minimize Cold Starts

- **Min Workers**: Set to 1 to maintain warm worker (increases cost)
- **Network Volume**: Reduce initialization time with model cache

### 3. GPU Selection

- **Development/Testing**: RTX A6000 (cheaper)
- **Production**: H100 (faster, full CUDA 12.8 support)

## Storage Configuration

This serverless handler supports two storage backends for uploading trained LoRA models:

### Option 1: Google Cloud Storage (Recommended for Production)

**Benefits:**

- Longer URL expiration (7 days vs 24-48 hours for RunPod)
- More control over storage lifecycle
- Better for multi-cloud deployments
- Parallel uploads (faster for multiple checkpoints)

**Setup:**

1. **Create GCS Bucket:**

   ```bash
   gsutil mb -l us-central1 gs://your-lora-models-bucket
   ```

2. **Create Service Account:**

   ```bash
   gcloud iam service-accounts create lora-uploader \
     --display-name="LoRA Training Uploader"

   # Grant Storage Object Creator permission
   gsutil iam ch serviceAccount:lora-uploader@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
     gs://your-lora-models-bucket

   # Generate JSON key
   gcloud iam service-accounts keys create service-account-key.json \
     --iam-account=lora-uploader@PROJECT_ID.iam.gserviceaccount.com
   ```

3. **Configure Environment Variables in RunPod Endpoint:**
   - `GCS_BUCKET_NAME`: `your-lora-models-bucket`
   - `GCP_SERVICE_ACCOUNT_JSON`: Base64-encoded service account JSON (recommended) or plain JSON

**Encoding credentials to Base64 (recommended):**

```bash
# Encode service account JSON to base64
cat service-account-key.json | base64
```

**Example Environment Variables:**

```bash
GCS_BUCKET_NAME=my-lora-models

# Option 1: Base64-encoded (recommended - safer for environment variables)
GCP_SERVICE_ACCOUNT_JSON=ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOi...

# Option 2: Plain JSON (may have issues with newlines in private key)
GCP_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"my-project",...}
```

**Note:** Base64 encoding is strongly recommended to avoid issues with special characters and newlines in the private key.

**File Organization:**
Uploaded files will be organized as:

```
gs://your-bucket/{job_id}/flux_lora.safetensors
gs://your-bucket/{job_id}/training.log
```

**URL Format:**
Signed URLs with 7-day expiration:

```
https://storage.googleapis.com/your-bucket/job-abc123/flux_lora.safetensors?X-Goog-Algorithm=...
```

### Option 2: RunPod Native S3 Storage (Default Fallback)

**Benefits:**

- Simple configuration (just 3 environment variables)
- Works with any S3-compatible storage (AWS S3, Backblaze B2, DigitalOcean Spaces, etc.)
- Built into RunPod SDK

**Setup:**

Configure these environment variables in your RunPod Endpoint:

- `BUCKET_ENDPOINT_URL`: Your S3 endpoint URL (must include bucket name)
  - Example (AWS): `https://your-bucket.s3.us-west-2.amazonaws.com`
  - Example (Backblaze): `https://your-bucket.s3.us-west-004.backblazeb2.com`
  - Example (DigitalOcean): `https://your-bucket.nyc3.digitaloceanspaces.com`
- `BUCKET_ACCESS_KEY_ID`: Your S3 access key ID
- `BUCKET_SECRET_ACCESS_KEY`: Your S3 secret access key

**Example Environment Variables:**

```bash
BUCKET_ENDPOINT_URL=https://my-lora-models.s3.us-west-2.amazonaws.com
BUCKET_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
BUCKET_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**File Organization:**
Files are uploaded with job ID prefix:

```
your-bucket/{job_id}/flux_lora.safetensors
your-bucket/{job_id}/training.log
```

**Documentation:** See [RunPod Environment Variables Guide](https://docs.runpod.io/serverless/development/environment-variables)

## Troubleshooting

### CUDA Error

```
RuntimeError: CUDA not available
```

**Solution**: Select CUDA 12.8 GPU when deploying on RunPod

### Model Download Failure

```
RuntimeError: Failed to download model
```

**Solution**:

1. Verify HF Token (for Flux)
2. Check Network Volume capacity
3. Check RunPod network status

### Training Failure

```
RuntimeError: Training failed
```

**Solution**:

1. Check logs: `result["traceback"]`
2. If GPU out of memory: Reduce `micro_batch_size_per_gpu`
3. Validate TOML parameters

### Timeout

```
asyncio.TimeoutError: Command timed out
```

**Solution**:

1. Increase RunPod Endpoint timeout (24 hours)
2. Reduce number of epochs
3. Check dataset size

## Development Guide

### Adding a New Model

1. Add to `MODEL_CONFIGS` in [config.py](config.py):

```python
"new_model": {
    "repo": "org/model-name",
    "requires_hf_token": False,
    "toml_file": "new_model.toml",
    "path_in_volume": "models/new_model"
}
```

2. Create `/toml_files/new_model.toml`

3. Test

### Adding Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Adding Retry Logic

```python
from utils import retry

@retry(max_attempts=3, backoff=2.0)
async def my_function():
    # Retries 3 times on failure (waits 2^n seconds)
    pass
```

## License

This project follows the original [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) license.

## Contributing

Issues and Pull Requests are welcome!

## References

- [RunPod Serverless Docs](https://docs.runpod.io/serverless/workers/handler-functions)
- [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe)
- [JoyCaption](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)
