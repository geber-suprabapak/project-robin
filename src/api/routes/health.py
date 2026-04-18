"""Health, readiness, and root endpoints."""

from typing import Dict

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.core.inference_engine import inference_engine
from src.services.face_detector import is_face_detector_ready
from src.services.qdrant_client import qdrant_service
from src.services.supabase_client import supabase_service
from src.schemas.api_models import HealthResponse

router = APIRouter(tags=["System"])


async def _readiness_payload() -> HealthResponse:
    """Build readiness status from all runtime dependencies."""
    model_loaded = inference_engine.is_loaded()
    face_detector_ready = is_face_detector_ready()
    supabase_connected = await supabase_service.is_ready()
    qdrant_connected = await qdrant_service.is_connected()

    is_ready = all(
        (
            model_loaded,
            face_detector_ready,
            supabase_connected,
            qdrant_connected,
        )
    )

    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=model_loaded,
        face_detector_ready=face_detector_ready,
        gpu_available=inference_engine.is_gpu_enabled(),
        supabase_connected=supabase_connected,
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
    """Check whether API dependencies are ready to serve traffic."""
    return await _readiness_response()


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check() -> HealthResponse | JSONResponse:
    """Backward-compatible health endpoint with readiness semantics."""
    return await _readiness_response()


@router.get("/", summary="Root endpoint")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "gpu_enabled": str(inference_engine.is_gpu_enabled()),
        "docs": "/docs",
    }
