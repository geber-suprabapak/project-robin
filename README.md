# Face Recognition API (FaceVector-Core)

A REST API for face recognition, powered by the ArcFace ONNX model. It supports GPU acceleration (CUDA) and uses Supabase for vector storage.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Features

- **GPU-Accelerated Inference**: Uses CUDA if available, otherwise falls back to CPU.
- **Multi-Image Enrollment**: Accepts 10-20 training images per user to improve accuracy.
- **Auto Face Cropping**: Automatically detects and crops faces from input images.
- **Face Identification**: Matches faces against the database in real-time.
- **Singleton Pattern**: Loads the model once to save memory.
- **Thread-Safe**: Handles concurrent requests safely.
- **Clean Architecture**: Code is organized into modular layers (api, core, services, schemas).
- **Supabase Integration**: Uses PostgreSQL and pgvector for similarity search.
- **Docker Ready**: Supports NVIDIA runtime for containerized deployment.
- **JWT Authentication**: Secure session-based auth using Supabase JWT tokens.

## Architecture

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
├── scripts/                          # Build scripts
│   ├── build-local.ps1               # Local Docker build (Windows)
│   ├── build-local.sh                # Local Docker build (Linux/macOS)
│   ├── build-prod.ps1                # GHCR build & push (Windows)
│   └── build-prod.sh                 # GHCR build & push (Linux/macOS)
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

## Quick Start

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

## Configuration

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
SUPABASE_JWT_SECRET=your-jwt-secret       # Dashboard → Settings → API → JWT Secret

# GHCR (GitHub Container Registry) - for Docker builds
GHCR_USERNAME=your-github-username
GHCR_TOKEN=ghp_xxxxxxxxxxxxx  # Personal Access Token with write:packages scope

# Face Recognition
FACE_MATCH_THRESHOLD=0.6
MAX_COSINE_DISTANCE=0.4

# Security
ADMIN_SECRET_KEY=your-secret-admin-key    # For /v1/enroll (admin-only)
```

## Database Setup

### 1. Enable pgvector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Run Schema

Execute `sql/face_embeddings_user_id_schema.sql` in Supabase SQL Editor.

## API Documentation

Interactive docs: `http://localhost:8000/docs`

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

**Headers**: `Authorization: Bearer <supabase_session_token>`

Requires a valid Supabase JWT session token. The server verifies the token signature and checks that the user exists in the database.

**Request (JSON)**:
```json
{
  "image_base64": "<base64-encoded-image>"
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

## Docker Deployment

### Quick Start with Docker Compose

```bash
docker-compose up -d
```

### Building Docker Images

#### Local Build (Testing)

Build Docker image locally without pushing to registry:

**Windows (PowerShell):**
```powershell
.\scripts\build-local.ps1              # Build with 'latest' tag
.\scripts\build-local.ps1 -Tag "dev"   # Build with custom tag
.\scripts\build-local.ps1 -NoCache     # Build without cache
```

**Linux/macOS (Bash):**
```bash
./scripts/build-local.sh               # Build with 'latest' tag
./scripts/build-local.sh -t "dev"      # Build with custom tag
./scripts/build-local.sh --no-cache    # Build without cache
```

#### Production Build (GHCR)

Build and push to GitHub Container Registry:

**Prerequisites:**
1. Add GHCR credentials to `.env`:
   ```ini
   GHCR_USERNAME=your-github-username
   GHCR_TOKEN=ghp_xxxxxxxxxxxxx  # GitHub Personal Access Token
   ```
2. Ensure your PAT has `write:packages` scope

**Windows (PowerShell):**
```powershell
.\scripts\build-prod.ps1               # Build and push with 'latest' tag
.\scripts\build-prod.ps1 -Tag "v1.0.0" # Build and push with version tag
```

**Linux/macOS (Bash):**
```bash
./scripts/build-prod.sh                # Build and push with 'latest' tag
./scripts/build-prod.sh -t "v1.0.0"    # Build and push with version tag
```

**Image Registry:** `ghcr.io/geber-suprabapak/project-robin`

### Running Docker Image

**With GPU support:**
```bash
docker run --gpus all -p 8000:8000 --env-file .env project-robin:latest
```

**CPU only:**
```bash
docker run -p 8000:8000 --env-file .env project-robin:latest
```

**From GHCR:**
```bash
docker pull ghcr.io/geber-suprabapak/project-robin:latest
docker run --gpus all -p 8000:8000 --env-file .env ghcr.io/geber-suprabapak/project-robin:latest
```

## Performance

| Operation | Time |
|-----------|------|
| Image Decode | ~5ms |
| Face Detection | ~10-20ms |
| GPU Inference | ~15-30ms |
| DB Search | ~10-20ms |
| **Total** | **~45-80ms** |

## Troubleshooting

### GPU Not Detected
```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Model Loading Error
Ensure model is at `./models/arcface_r100_224x224.onnx`

### Authentication Errors (401)
- **For `/v1/enroll`**: Ensure `X-Admin-Key` header matches `ADMIN_SECRET_KEY` in `.env`
- **For `/v1/identify`**: Ensure valid Supabase JWT token in `Authorization: Bearer <token>` header. Check that `SUPABASE_JWT_SECRET` is correctly configured.

---

**made with ❤️ by fizrayy**