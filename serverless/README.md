# RunPod Serverless LoRA Training

RunPod Serverless 환경에서 LoRA (Low-Rank Adaptation) 모델을 훈련하는 비동기 핸들러입니다.

## 특징

- **비동기 실행**: 긴 훈련 작업을 위한 async 핸들러
- **전체 커스터마이징**: 모든 TOML 파라미터를 JSON으로 제어
- **자동 캡셔닝**: JoyCaption (이미지) + Gemini API (비디오)
- **모델 캐싱**: `/runpod-volume`에 모델 저장하여 재사용
- **Retry 로직**: 네트워크 오류 시 3회 재시도
- **URL 입력**: 이미지/비디오를 URL로 다운로드
- **Pre-signed URL 출력**: `rp upload`로 결과 업로드

## 지원 모델

- **Flux** (black-forest-labs/FLUX.1-dev) - HF Token 필요
- **SDXL** (Stable Diffusion XL)
- **Wan 1.3B** (Text-to-Video)
- **Wan 14B T2V** (Text-to-Video 14B)
- **Wan 14B I2V** (Image-to-Video 14B)
- **Qwen Image**

## 아키텍처

```
handler.py (진입점)
    ↓
├── config.py (설정 검증)
├── downloader.py (모델 + 데이터셋 다운로드)
├── caption_manager.py (JoyCaption + Gemini)
├── training_manager.py (TOML 생성 + DeepSpeed 실행)
└── utils.py (Retry, CUDA, Upload)
```

## 배포 방법

### 1. Docker 이미지 빌드

```bash
cd /Users/ghyeok/Desktop/runpod-diffusion_pipe
docker build -f Dockerfile.serverless -t your-registry/runpod-lora-training:latest .
docker push your-registry/runpod-lora-training:latest
```

### 2. RunPod Serverless 생성

RunPod 웹사이트에서:

1. **Serverless** → **New Endpoint** 클릭
2. **Container Image**: `your-registry/runpod-lora-training:latest`
3. **GPU Type**: H100 또는 H200 권장
4. **Min Workers**: 0 (cold start 허용)
5. **Max Workers**: 필요한 동시 작업 수
6. **Timeout**: 24시간 (긴 훈련용)
7. **Network Volume**: 생성 및 마운트 (`/runpod-volume`)

### 3. API 호출

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Job 생성
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

# Job 상태 확인
status = runpod.Endpoint("ENDPOINT_ID").status(job["id"])
print(status)

# 결과 가져오기
result = runpod.Endpoint("ENDPOINT_ID").result(job["id"])
print(result["output"]["download_urls"])
```

## 입력 스키마

```json
{
  "input": {
    "model_type": "flux | sdxl | wan13 | wan14b_t2v | wan14b_i2v | qwen",

    "image_urls": ["https://...", "https://..."],
    "video_urls": ["https://...", "https://..."],

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

### 필수 필드

- `model_type`: 훈련할 모델 타입
- `image_urls` 또는 `video_urls`: 최소 하나 이상

### 조건부 필수 필드

- `hf_token`: Flux 모델 사용 시 필수
- `gemini_api_key`: 비디오 캡셔닝 사용 시 필수

### 선택 필드

- `caption_mode`: 기본값 `"skip"`
- `trigger_word`: 이미지 캡션 앞에 추가할 단어
- `training_params`: 기본값으로 채워짐 (일부만 override 가능)

## 출력 스키마

### 성공 시

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

### 실패 시

```json
{
  "status": "failed",
  "error": "Error message",
  "traceback": "Full Python traceback...",
  "execution_time": 123.45
}
```

## 로컬 테스트

```bash
# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0

# 테스트 job 생성
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

# 핸들러 실행
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

## 비용 최적화

### 1. 모델 캐싱

모델을 Network Volume에 미리 다운로드:

```bash
# RunPod Pod에서 실행
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /runpod-volume/models/flux \
  --token YOUR_HF_TOKEN
```

이후 serverless job에서는 다운로드 없이 캐시 사용.

### 2. Cold Start 최소화

- **Min Workers**: 1로 설정하여 warm worker 유지 (비용 증가)
- **Network Volume**: 모델 캐시로 초기화 시간 단축

### 3. GPU 선택

- **개발/테스트**: RTX A6000 (저렴)
- **프로덕션**: H100 (빠름, CUDA 12.8 완벽 지원)

## 문제 해결

### CUDA 오류

```
RuntimeError: CUDA not available
```

**해결**: RunPod 배포 시 CUDA 12.8 GPU 선택

### 모델 다운로드 실패

```
RuntimeError: Failed to download model
```

**해결**:
1. HF Token 확인 (Flux의 경우)
2. Network Volume 용량 확인
3. RunPod 네트워크 상태 확인

### 훈련 실패

```
RuntimeError: Training failed
```

**해결**:
1. 로그 확인: `result["traceback"]`
2. GPU 메모리 부족 시: `micro_batch_size_per_gpu` 감소
3. TOML 파라미터 검증

### Timeout

```
asyncio.TimeoutError: Command timed out
```

**해결**:
1. RunPod Endpoint timeout 증가 (24시간)
2. Epochs 수 감소
3. 데이터셋 크기 확인

## 개발 가이드

### 새 모델 추가

1. [config.py](config.py:21:0-52:1)의 `MODEL_CONFIGS`에 추가:

```python
"new_model": {
    "repo": "org/model-name",
    "requires_hf_token": False,
    "toml_file": "new_model.toml",
    "path_in_volume": "models/new_model"
}
```

2. `/toml_files/new_model.toml` 생성

3. 테스트

### 로깅 추가

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Retry 로직 추가

```python
from utils import retry

@retry(max_attempts=3, backoff=2.0)
async def my_function():
    # 실패 시 3회 재시도 (2^n초 대기)
    pass
```

## 라이센스

이 프로젝트는 원본 [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) 라이센스를 따릅니다.

## 기여

Issues와 Pull Requests 환영합니다!

## 참고 문서

- [RunPod Serverless Docs](https://docs.runpod.io/serverless/workers/handler-functions)
- [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe)
- [JoyCaption](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)
