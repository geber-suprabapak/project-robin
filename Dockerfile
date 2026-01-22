# ============================================================================
# Face Recognition API - Dockerfile (with uv)
# Optimized for NVIDIA GPU Runtime with CUDA 11.8 + uv package manager
# ============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="LunaraDev"
LABEL description="Face Recognition API with ONNX Runtime GPU support (uv-powered)"
LABEL version="2.0.0"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (minimal Python, we'll let uv handle the rest)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install Python 3.12 and sync dependencies with uv
# This creates a .venv in /app/.venv
RUN uv python install 3.12 && \
    uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using uv
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
