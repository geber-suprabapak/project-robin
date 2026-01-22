# 🤖 Face Recognition API (FaceVector-Core)

High-performance REST API for face recognition processing using **ArcFace** ONNX model with **GPU acceleration** (NVIDIA CUDA) and **Supabase** database integration.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Database Setup](#-database-setup)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **GPU-Accelerated Inference**: CUDA support with automatic CPU fallback
- **Multi-Image Enrollment**: Support for 10-20 training images per user
- **Auto Face Cropping**: Automatic face detection and cropping
- **Face Identification**: Real-time face matching against database
- **Singleton Pattern**: Model loaded once, efficient memory usage
- **Thread-Safe**: Concurrent request handling with thread locks
- **Clean Architecture**: Modular layers (api, core, services, schemas)
- **Supabase Integration**: PostgreSQL with pgvector for similarity search
- **Docker Ready**: NVIDIA runtime support for containerized deployment
- **Admin Protection**: API key authentication for sensitive endpoints

## 🏗 Architecture

```
project-robin/
├── src/                              # Application source code
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # Pydantic Settings
│   ├── dependencies.py               # FastAPI dependencies (auth)
│   │
│   ├── api/                          # API layer
│   │   ├── __init__.py
│   │   ├── exceptions.py             # Exception handlers
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py             # GET /, GET /health
│   │       ├── identification.py     # POST /v1/identify
│   │       └── enrollment.py         # POST /v1/enroll
│   │
│   ├── core/                         # ML inference
│   │   ├── __init__.py
│   │   └── inference_engine.py       # Singleton ONNX Runtime engine
│   │
│   ├── services/                     # Business logic
│   │   ├── __init__.py
│   │   ├── image_decoder.py          # Base64 → numpy, preprocessing
│   │   ├── face_detector.py          # DNN face detection + cropping
│   │   └── supabase_client.py        # Database operations
│   │
│   └── schemas/                      # Pydantic models
│       ├── __init__.py
│       └── api_models.py
│
├── models/                           # ONNX models (gitignored)
├── sql/                              # Database schemas
│   ├── schema_latest_latest.sql
│   ├── face_embeddings_user_id_schema.sql
│   └── MIGRATION_INSTRUCTIONS.md
│
├── .env                              # Environment config
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── run-local.ps1                     # Windows launcher
└── setup.ps1
```

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 or Linux
- NVIDIA GPU with CUDA support (optional, falls back to CPU)

### One-Command Setup (Windows)

```powershell
.\run-local.ps1
```

### Manual Setup

```bash
# Install dependencies
uv sync --python 3.12

# Run the server
uv run python -m src.main
```

## ⚙️ Configuration

### Create `.env` File

```bash
cp .env.example .env
```

### Environment Variables

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

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Face Recognition
FACE_MATCH_THRESHOLD=0.6
MAX_COSINE_DISTANCE=0.4

# Admin Security
ADMIN_SECRET_KEY=your-secret-admin-key
```

## 🗄 Database Setup

### 1. Enable pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Run Schema

Execute `sql/face_embeddings_user_id_schema.sql` in Supabase SQL Editor.

## 📚 API Documentation

Interactive docs: **http://localhost:8000/docs**

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/v1/identify` | Identify face |
| POST | `/v1/enroll` | Enroll student (10-20 images) |

### `POST /v1/enroll`

**Headers**: `X-Admin-Key: your-admin-secret-key`

**Request (multipart/form-data)**:
- `files`: 10-20 face images (JPG/PNG)
- `name`: Student's full name
- `nisn`: Student ID (NIS/NISN)
- `class_name`: Class name (optional)

### `POST /v1/identify`

**Request (JSON)**:
```json
{
  "image_base64": "<base64-encoded-image>",
  "camera_id": "kiosk_001"
}
```

**Response**:
```json
{
  "status": "ok",
  "student_id": "12345678",
  "student_name": "Ahmad Rizki",
  "confidence": 0.95,
  "process_time_ms": 45
}
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## ⚡ Performance

| Operation | Time |
|-----------|------|
| Image Decode | ~5ms |
| Face Detection | ~10-20ms |
| GPU Inference | ~15-30ms |
| DB Search | ~10-20ms |
| **Total** | **~45-80ms** |

## 🔧 Troubleshooting

### GPU Not Detected
```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Model Loading Error
Ensure model is at `./models/arcface_r100_224x224.onnx`

### Invalid Admin Key (401)
Ensure `X-Admin-Key` header matches `ADMIN_SECRET_KEY` in `.env`

---

**Built with ❤️ using FastAPI, ONNX Runtime, uv, and Supabase**
