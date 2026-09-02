# Modules

## Application runtime

**Purpose:** Start FastAPI, register routes/middleware, and manage model lifecycle.
**Entry point:** `src/main.py`.

## API routes and schemas

**Purpose:** Validate enrollment/identification/health requests and shape responses.
**Entry points:** `src/api/routes/`, `src/schemas/api_models.py`.

## Inference core

**Purpose:** Decode images, detect faces, and generate embeddings with ONNX/OpenCV.
**Entry points:** `src/core/`, `src/services/image_decoder.py`.

## Qdrant and model services

**Purpose:** Store/search embeddings and obtain verified model assets.
**Entry points:** `src/services/qdrant_client.py`, `src/services/model_downloader.py`.
