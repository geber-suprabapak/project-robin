# 🤖 Face Recognition API

High-performance REST API for face recognition processing using **ArcFace** ONNX model with **GPU acceleration** (NVIDIA CUDA) and **Supabase** database integration.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Model Setup](#-model-setup-important)
- [Configuration](#️-configuration)
- [Database Setup](#-database-setup)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **GPU-Accelerated Inference**: CUDA support with automatic CPU fallback
- **Singleton Pattern**: Model loaded once, efficient memory usage
- **Thread-Safe**: Concurrent request handling with thread locks
- **Clean Architecture**: Separated layers (endpoints, core, services, schemas)
- **Supabase Integration**: PostgreSQL with pgvector for similarity search
- **Docker Ready**: NVIDIA runtime support for containerized deployment
- **Type-Safe**: Full mypy compliance with Pydantic models
- **Production Ready**: Health checks, logging, error handling

## 🏗 Architecture

```
project-robin/
├── main.py                      # FastAPI app with lifespan management
├── config.py                    # Pydantic Settings (env-based config)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # NVIDIA CUDA optimized container
├── docker-compose.yml           # Docker orchestration with GPU support
│
├── core/
│   └── inference_engine.py      # Singleton FaceInferenceEngine (ONNX Runtime)
│
├── services/
│   ├── image_decoder.py         # Base64 → numpy, preprocessing, cosine similarity
│   └── supabase_client.py       # Database operations, vector search
│
├── schemas/
│   └── api_models.py            # Pydantic request/response models
│
└── sql/
    ├── schema_latest_latest.sql # Main database schema
    └── face_embeddings_schema.sql # Face embeddings table + pgvector
```

## 🔧 Prerequisites

### Hardware
- **NVIDIA GPU** (GTX 1650 or better) with CUDA 11.8 support
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for models and dependencies

### Software
- **Python** 3.11+
- **CUDA Toolkit** 11.8+ (for local GPU support)
- **Docker** with NVIDIA Container Toolkit (for containerized deployment)
- **PostgreSQL** with pgvector extension (via Supabase)

## 📦 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd project-robin
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Model Setup (IMPORTANT)

### Download ArcFace ONNX Model

You need to download a pre-trained **ArcFace** ONNX model optimized for 224x224 Hi-Res input:

#### Option 1: Official InsightFace Models (Recommended)

1. Visit [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
2. Download **ArcFace ResNet100** model:
   - Model Name: `glint360k_cosface_r100_fp16_0.1` or `ms1mv3_arcface_r100_fp16`
   - Input Size: **224×224** (Hi-Res)
   - Output: **512-dimensional** embedding

3. Convert to ONNX format (if not already):
```bash
python -m insightface.model_zoo.model_export \
    --model arcface_r100 \
    --output ./models/arcface_r100_224x224.onnx
```

#### Option 2: ONNX Model Hub

```bash
# Create models directory
mkdir -p models

# Download from ONNX Model Zoo (example URL - verify latest)
wget -O models/arcface_r100_224x224.onnx \
    https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/arcface_r100_v1.onnx
```

#### Option 3: Custom Training

If you have your own trained ArcFace model:
- Ensure input shape: `(1, 3, 224, 224)`
- Ensure output shape: `(1, 512)`
- Export to ONNX format with opset 11+

### Model Specifications

| Property | Value |
|----------|-------|
| **File Type** | `.onnx` |
| **Input Shape** | `(1, 3, 224, 224)` |
| **Input Format** | RGB, normalized to [-1, 1] |
| **Output Shape** | `(1, 512)` |
| **Model Size** | ~200-500 MB |

### Verify Model

```bash
python -c "import onnx; model = onnx.load('models/arcface_r100_224x224.onnx'); print(onnx.checker.check_model(model))"
```

## ⚙️ Configuration

### Create `.env` File

```bash
cp .env.example .env
```

### Edit Configuration

```ini
# Model Configuration
MODEL_PATH=./models/arcface_r100_224x224.onnx
MODEL_INPUT_SIZE=224
EMBEDDING_DIM=512

# GPU Configuration
GPU_DEVICE_ID=0                # 0 for first GPU, -1 for CPU only
GPU_MEM_LIMIT=2147483648       # 2GB memory limit

# API Server
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Supabase (Get from Supabase Dashboard)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Face Recognition Thresholds
FACE_MATCH_THRESHOLD=0.6       # 0.0 to 1.0 (higher = stricter)
MAX_COSINE_DISTANCE=0.4        # 0.0 to 2.0 (lower = stricter)
```

## 🗄 Database Setup

### 1. Enable pgvector Extension

In your Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Run Main Schema

```bash
# Execute schema_latest_latest.sql in Supabase SQL Editor
# (Already exists in your project)
```

### 3. Run Face Embeddings Schema

```bash
# Execute sql/face_embeddings_schema.sql in Supabase SQL Editor
```

This creates:
- `face_embeddings` table with vector(512) column
- HNSW index for fast similarity search
- RPC functions: `find_face_match()`, `upsert_face_embedding()`
- Row Level Security policies

## 🚀 Usage

### Local Development

```bash
# Ensure .env is configured and model is downloaded
python main.py
```

Server starts at: `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Test Request

```bash
curl -X POST "http://localhost:8000/v1/identify" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
    "camera_id": "kiosk_001"
  }'
```

### Response

```json
{
  "status": "ok",
  "student_id": "12345678",
  "student_name": "Ahmad Rizki",
  "confidence": 0.95,
  "process_time_ms": 45,
  "message": "Face identified successfully"
}
```

## 📚 API Documentation

### `POST /v1/identify`

Identify a person from a face image.

**Request:**
```json
{
  "image_base64": "string (base64-encoded image)",
  "camera_id": "string (camera/kiosk identifier)"
}
```

**Response (Success):**
```json
{
  "status": "ok",
  "student_id": "12345678",
  "student_name": "Ahmad Rizki",
  "confidence": 0.95,
  "process_time_ms": 45,
  "message": "Face identified successfully"
}
```

**Response (Not Found):**
```json
{
  "status": "not_found",
  "student_id": null,
  "confidence": null,
  "process_time_ms": 52,
  "message": "No matching face found in database"
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "supabase_connected": true
}
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t face-recognition-api:latest .
```

### Run with Docker Compose (Recommended)

```bash
# Ensure .env file is configured
docker-compose up -d
```

### Run with Docker CLI

```bash
docker run -d \
  --name face-recognition-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/.env:/app/.env:ro \
  face-recognition-api:latest
```

### Verify GPU Access

```bash
docker exec face-recognition-api nvidia-smi
```

## ⚡ Performance

### Typical Latency (with GPU)

| Operation | Time |
|-----------|------|
| Image Decode | ~5ms |
| Preprocessing | ~3ms |
| GPU Inference | ~15-30ms |
| DB Search (pgvector) | ~10-20ms |
| **Total** | **~35-60ms** |

### Optimization Tips

1. **Batch Processing**: Increase workers for concurrent requests
2. **Model Quantization**: Use FP16 instead of FP32 (2x faster)
3. **Index Tuning**: Adjust HNSW parameters in SQL schema
4. **Caching**: Cache frequent queries

## 🔧 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should include 'CUDAExecutionProvider'
```

### Model Loading Error

```
FileNotFoundError: Model file not found
```

**Solution**: Ensure model is at `./models/arcface_r100_224x224.onnx`

### Supabase Connection Failed

```
WARNING: Supabase not connected (simulation mode)
```

**Solution**: Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`

### Low Confidence Scores

**Solution**: Lower `FACE_MATCH_THRESHOLD` in `.env` (e.g., from 0.6 to 0.5)

## 📝 License

[Your License Here]

## 🤝 Contributing

[Your Contributing Guidelines]

## 📧 Support

For issues and questions, please open a GitHub issue.

---

**Built with ❤️ using FastAPI, ONNX Runtime, and Supabase**
