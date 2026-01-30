# ============================================================================
# Face Recognition API - Unified Dockerfile
# Supports both CPU and GPU runtimes via build arguments.
# Default: CPU Optimized (Smaller size, faster pull)
# ============================================================================

# Arguments for flexible build
ARG RUNTIME_TYPE=cpu
ARG BASE_IMAGE=python:3.12-slim-bookworm

# ----------------------------------------------------------------------------
# Base Stage
# ----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

# Metadata
LABEL maintainer="LunaraDev"
LABEL description="Face Recognition API"
LABEL version="2.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0: for OpenCV
# libsm6, libxext6, libxrender-dev: for graphical rendering support
# libgomp1: for OpenMP support
# curl: for healthcheck and uv download
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# ----------------------------------------------------------------------------
# Dependencies Stage
# ----------------------------------------------------------------------------
# Copy dependency files
# We copy all potential config files, then select the right one based on RUNTIME_TYPE
COPY pyproject.toml pyproject.toml.cpu ./
COPY uv.lock ./

ARG RUNTIME_TYPE
RUN if [ "$RUNTIME_TYPE" = "cpu" ]; then \
    echo "🔧 Configuring for CPU runtime..."; \
    mv pyproject.toml.cpu pyproject.toml; \
    # Remove lock file for CPU to force fresh resolve (since lock is likely GPU)
    rm -f uv.lock; \
    else \
    echo "🔧 Configuring for GPU runtime..."; \
    # Keep pyproject.toml and uv.lock as is for GPU
    rm -f pyproject.toml.cpu; \
    fi

# Install Dependencies
# create virtual environment and install deps
RUN uv python install 3.12 && \
    if [ "$RUNTIME_TYPE" = "cpu" ]; then \
    uv sync --no-dev; \
    else \
    uv sync --frozen --no-dev; \
    fi

# ----------------------------------------------------------------------------
# Final Stage
# ----------------------------------------------------------------------------
# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "python", "-m", "src.main"]
