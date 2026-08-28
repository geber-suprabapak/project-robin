# Project Robin (FaceVector-Core)

## Purpose
High-performance internal REST API for face recognition processing using ArcFace ONNX model with GPU acceleration (NVIDIA CUDA), called by Astra.

## Tech Stack
- **Framework**: FastAPI (Python 3.10+)
- **ML Runtime**: ONNX Runtime GPU (CUDA support)
- **Vector store**: Qdrant (user-scoped face embeddings)
- **Package Manager**: uv (preferred)
- **Image Processing**: OpenCV, Pillow, NumPy

## Architecture
```
project-robin/
├── main.py              # FastAPI app with endpoints
├── config.py            # Pydantic Settings
├── dependencies.py      # FastAPI dependencies (admin auth)
├── core/                # Inference engine (singleton)
├── services/            # Business logic (image, face detection, Qdrant)
├── schemas/             # Pydantic models
└── sql/                 # Database schemas
```

## Key Features
- GPU-accelerated face inference
- Face enrollment with quality check (DNN face detection)
- Face identification with pgvector similarity search
- Role-based API key protection

## Security Architecture
- **Astra service boundary**:
  - `Authorization: Bearer <ROBIN_SERVICE_TOKEN>` authenticates Astra
  - `X-Astra-User-Id` supplies the already-authenticated user context
  - Robin has no direct identity-provider or domain-database dependency
- **Dependencies** (`src/dependencies.py`):
    - `verify_jwt_bearer`: Validates the Astra service credential and user context
