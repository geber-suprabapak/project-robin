# 🤖 Face Recognition API (FaceVector-Core)

High-performance REST API for face recognition processing using **ArcFace** ONNX model with **GPU acceleration** (NVIDIA CUDA) and **Supabase** database integration.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start (Local Testing)](#-quick-start-local-testing)
- [Installation (Manual)](#-installation-manual)
- [Model Setup](#-model-setup-important)
- [Configuration](#️-configuration)
- [Database Setup](#-database-setup)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **GPU-Accelerated Inference**: CUDA support with automatic CPU fallback
- **Face Enrollment**: DNN-based face detection + embedding extraction
- **Face Identification**: Real-time face matching against database
- **Singleton Pattern**: Model loaded once, efficient memory usage
- **Thread-Safe**: Concurrent request handling with thread locks
- **Clean Architecture**: Separated layers (endpoints, core, services, schemas)
- **Supabase Integration**: PostgreSQL with pgvector for similarity search
- **Docker Ready**: NVIDIA runtime support for containerized deployment
- **Type-Safe**: Full mypy compliance with Pydantic models
- **Admin Protection**: API key authentication for sensitive endpoints

## 🏗 Architecture

```
project-robin/
├── main.py                      # FastAPI app with lifespan management
├── config.py                    # Pydantic Settings (env-based config)
├── dependencies.py              # FastAPI dependencies (admin auth)
├── pyproject.toml               # uv package configuration
├── run-local.ps1                # Windows launcher script
│
├── core/
│   └── inference_engine.py      # Singleton FaceInferenceEngine (ONNX Runtime)
│
├── services/
│   ├── image_decoder.py         # Base64 → numpy, preprocessing
│   ├── face_detector.py         # DNN face detection (ResNet SSD)
│   └── supabase_client.py       # Database operations, vector search
│
├── schemas/
│   └── api_models.py            # Pydantic request/response models
│
└── sql/
    ├── schema_latest_latest.sql # Main database schema
    └── face_embeddings_schema.sql # Face embeddings + pgvector
```

## 🚀 Quick Start (Local Testing)

### Prerequisites
- Windows 10/11
- NVIDIA GPU with CUDA support (optional, will fallback to CPU)

### One-Command Setup

```powershell
# Run the launcher script (installs uv automatically if needed)
.\run-local.ps1
```

The script will:
1. ✅ Check/install **uv** package manager
2. ✅ Install **Python 3.12** (compatible with all dependencies)
3. ✅ Sync all dependencies
4. ✅ Validate configuration
5. ✅ Start the API server

### Manual uv Installation

If you prefer to install uv manually first:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then run:

```powershell
uv sync --python 3.12
uv run python main.py
```

## 📦 Installation (Manual)

### Using uv (Recommended)

```bash
# Install dependencies
uv sync --python 3.12

# Run the server
uv run python main.py
```

### Using pip (Traditional)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

## 🎯 Model Setup (IMPORTANT)

### Download ArcFace ONNX Model

Create `models/` directory and download an ArcFace model:

```bash
mkdir models
```

#### Option 1: InsightFace Model Zoo (Recommended)

1. Visit [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
2. Download **ArcFace ResNet100** (224×224 input, 512-dim output)
3. Place at: `models/arcface_r100_224x224.onnx`

### Model Specifications

| Property | Value |
|----------|-------|
| **File Type** | `.onnx` |
| **Input Shape** | `(1, 3, 224, 224)` |
| **Input Format** | RGB, normalized to [-1, 1] |
| **Output Shape** | `(1, 512)` |
| **Model Size** | ~200-500 MB |

## ⚙️ Configuration

### Create `.env` File

```bash
cp .env.example .env
```

### Edit Configuration

```ini
# ===== Model Configuration =====
MODEL_PATH=./models/arcface_r100_224x224.onnx
MODEL_INPUT_SIZE=224
EMBEDDING_DIM=512

# ===== GPU Configuration =====
GPU_DEVICE_ID=0                # 0 for first GPU, -1 for CPU only
GPU_MEM_LIMIT=2147483648       # 2GB memory limit

# ===== API Server =====
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# ===== Supabase (Staging/Production) =====
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# ===== Face Recognition Thresholds =====
FACE_MATCH_THRESHOLD=0.6       # 0.0 to 1.0 (higher = stricter)
MAX_COSINE_DISTANCE=0.4        # 0.0 to 2.0 (lower = stricter)

# ===== Admin Security =====
ADMIN_SECRET_KEY=your-secret-admin-key-here
```

### Supabase Staging Database

For local testing with staging database:

1. Go to your [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your staging project
3. Go to **Settings** → **API**
4. Copy:
   - **URL**: `SUPABASE_URL`
   - **anon public**: `SUPABASE_KEY`
   - **service_role**: `SUPABASE_SERVICE_ROLE_KEY`

## 🗄 Database Setup

### 1. Enable pgvector Extension

In Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Run Face Embeddings Schema

Execute `sql/face_embeddings_schema.sql` in Supabase SQL Editor.

This creates:
- `face_embeddings` table with vector(512) column
- HNSW index for fast similarity search
- RPC functions: `find_face_match()`, `upsert_face_embedding()`
- Row Level Security policies

## 📚 API Documentation

Interactive docs available at: **http://localhost:8000/docs**

---

### `POST /v1/enroll` - Enroll Student Face

Register a student's face for recognition. **Requires admin API key.**

**Headers:**
```
X-Admin-Key: your-admin-secret-key
```

**Request (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Face image (JPG/PNG) |
| `name` | string | ✅ | Student's full name |
| `nisn` | string | ✅ | Student's unique ID (NIS/NISN) |
| `class_name` | string | ❌ | Class name (optional) |

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/v1/enroll" \
  -H "X-Admin-Key: your-admin-secret-key" \
  -F "file=@student_photo.jpg" \
  -F "name=Ahmad Rizki" \
  -F "nisn=12345678" \
  -F "class_name=XII IPA 1"
```

**Response (Success):**
```json
{
  "status": "success",
  "student_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Student enrolled successfully"
}
```

**Response (Error - Multiple Faces):**
```json
{
  "status": "error",
  "error": "HTTPException",
  "message": "Multiple faces detected (3). Please use a solo photo."
}
```

---

### `POST /v1/identify` - Identify Person

Identify a person from a face image.

**Request (JSON):**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "camera_id": "kiosk_001"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/v1/identify" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded-image>",
    "camera_id": "kiosk_001"
  }'
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

---

### `GET /health` - Health Check

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

### Build & Run

```bash
docker-compose up -d
```

### Run with Docker CLI (GPU)

```bash
docker run -d \
  --name face-recognition-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/.env:/app/.env:ro \
  face-recognition-api:latest
```

## ⚡ Performance

### Typical Latency (with GPU)

| Operation | Time |
|-----------|------|
| Image Decode | ~5ms |
| Face Detection (DNN) | ~10-20ms |
| Preprocessing | ~3ms |
| GPU Inference | ~15-30ms |
| DB Search (pgvector) | ~10-20ms |
| **Total** | **~45-80ms** |

## 🔧 Troubleshooting

### uv Not Found

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### GPU Not Detected

```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

### Model Loading Error

Ensure model is at `./models/arcface_r100_224x224.onnx`

### Supabase Connection Failed

Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`

### Invalid Admin Key (401)

Ensure `X-Admin-Key` header matches `ADMIN_SECRET_KEY` in `.env`

---

**Built with ❤️ using FastAPI, ONNX Runtime, uv, and Supabase**

