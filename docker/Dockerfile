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

RUN mkdir -p /app/models/face_detector /app/runtime-models /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["python", "-m", "src.main"]
