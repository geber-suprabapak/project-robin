"""Health, readiness, and root endpoints."""

import asyncio
import logging
from typing import Dict

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.face_detector import is_face_detector_ready
from src.services.qdrant_client import qdrant_service
from src.schemas.api_models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


async def _qdrant_connected_bounded() -> bool:
    """Check Qdrant readiness with a bounded wait; timeouts report not ready."""
    try:
        return await asyncio.wait_for(
            qdrant_service.is_connected(),
            timeout=settings.qdrant_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Qdrant readiness check exceeded %.1fs timeout; reporting not ready",
            settings.qdrant_timeout_seconds,
        )
        return False


async def _readiness_payload() -> HealthResponse:
    """Build readiness status from Robin's technical dependencies."""
    model_loaded = inference_engine.is_loaded()
    face_detector_ready = is_face_detector_ready()
    qdrant_connected = await _qdrant_connected_bounded()
    is_ready = all((model_loaded, face_detector_ready, qdrant_connected))

    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=model_loaded,
        face_detector_ready=face_detector_ready,
        gpu_available=inference_engine.is_gpu_enabled(),
        qdrant_connected=qdrant_connected,
    )


async def _readiness_response() -> HealthResponse | JSONResponse:
    payload = await _readiness_payload()
    if payload.status != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )
    return payload


@router.get("/live", summary="Liveness check endpoint")
async def liveness_check() -> Dict[str, str]:
    """Report that the API process is alive."""
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse, summary="Readiness check endpoint")
async def readiness_check() -> HealthResponse | JSONResponse:
    """Check whether Robin's technical dependencies are ready.

    Strictly read-only: a missing Qdrant collection is reported as not ready
    (HTTP 503), never auto-created from a health probe.
    """
    return await _readiness_response()


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check() -> HealthResponse | JSONResponse:
    """Backward-compatible health endpoint with readiness semantics."""
    return await _readiness_response()


@router.get("/", summary="Root endpoint")
async def root() -> Dict[str, str]:
    """Return API information without exposing domain dependencies."""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "gpu_enabled": str(inference_engine.is_gpu_enabled()),
        "docs": "/docs",
    }
