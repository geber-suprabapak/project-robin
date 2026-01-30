"""Health and root endpoints."""

from typing import Dict
from fastapi import APIRouter
from src.core.inference_engine import inference_engine
from src.services.qdrant_client import qdrant_service
from src.services.supabase_client import supabase_service
from src.schemas.api_models import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check() -> HealthResponse:
    """Check API health status and component availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine.is_loaded(),
        gpu_available=inference_engine.is_gpu_enabled(),
        supabase_connected=supabase_service.is_connected(),
        qdrant_connected=qdrant_service.is_connected()
    )


@router.get("/", summary="Root endpoint")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "gpu_enabled": str(inference_engine.is_gpu_enabled()),
        "docs": "/docs"
    }
