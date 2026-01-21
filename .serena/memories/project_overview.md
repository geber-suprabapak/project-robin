# Project Robin (FaceVector-Core)

## Purpose
High-performance REST API for face recognition processing using ArcFace ONNX model with GPU acceleration (NVIDIA CUDA) and Supabase database integration.

## Tech Stack
- **Framework**: FastAPI (Python 3.10+)
- **ML Runtime**: ONNX Runtime GPU (CUDA support)
- **Database**: Supabase (PostgreSQL with pgvector)
- **Package Manager**: uv (preferred)
- **Image Processing**: OpenCV, Pillow, NumPy

## Architecture
```
project-robin/
├── main.py              # FastAPI app with endpoints
├── config.py            # Pydantic Settings
├── dependencies.py      # FastAPI dependencies (admin auth)
├── core/                # Inference engine (singleton)
├── services/            # Business logic (image, face detection, supabase)
├── schemas/             # Pydantic models
└── sql/                 # Database schemas
```

## Key Features
- GPU-accelerated face inference
- Face enrollment with quality check (DNN face detection)
- Face identification with pgvector similarity search
- Admin API key protection (`X-Admin-Key` header)
