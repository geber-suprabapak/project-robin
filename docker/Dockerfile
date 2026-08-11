ARG RUNTIME_TYPE=cpu
ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG PYTHON_VERSION=3.12

FROM ${BASE_IMAGE} AS base

LABEL maintainer="LunaraDev"
LABEL description="Face Recognition API"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ARG RUNTIME_TYPE
ARG PYTHON_VERSION

COPY pyproject.toml uv.lock ./
RUN uv python install "${PYTHON_VERSION}" && \
    uv sync --frozen --no-dev --extra "${RUNTIME_TYPE}" --python "${PYTHON_VERSION}"

COPY src ./src

RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --create-home appuser && \
    mkdir -p /app/models/face_detector /app/runtime-models /app/logs && \
    curl -fsSL \
        https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt \
        -o /app/models/face_detector/deploy.prototxt && \
    echo "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9  /app/models/face_detector/deploy.prototxt" | sha256sum -c - && \
    curl -fsSL \
        https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel \
        -o /app/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel && \
    echo "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d  /app/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel" | sha256sum -c -

EXPOSE 8000

RUN chown -R appuser:appuser /app

USER appuser

# Fail the image build if the default runtime user cannot use the project venv.
RUN python -c "import sys, fastapi, uvicorn; assert sys.prefix == '/app/.venv', sys.prefix"

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["python", "-m", "src.main"]
