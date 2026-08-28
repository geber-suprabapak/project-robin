"""
Face Recognition API - FastAPI Application
High-performance REST API for face recognition processing with GPU acceleration.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.core.inference_engine import inference_engine
from src.api.routes import health, identification, enrollment
from src.api.exceptions import http_exception_handler, general_exception_handler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Face Recognition API")
    
    try:
        if settings.skip_model_load:
            logger.warning("SKIP_MODEL_LOAD=true; skipping face recognition model load")
        else:
            logger.info("📦 Loading face recognition model...")
            inference_engine.load_model()

            logger.info("🔥 Warming up inference engine...")
            inference_engine.warmup()
        
        logger.info("=" * 60)
        logger.info(f"✅ Server ready on {settings.api_host}:{settings.api_port}")
        logger.info(f"📊 GPU Enabled: {inference_engine.is_gpu_enabled()}")
        logger.info(f"🔧 Provider: {inference_engine.get_provider()}")
        logger.info(f"🌍 Environment: {settings.environment}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down Face Recognition API...")
    logger.info("✅ Cleanup complete")


# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="High-performance REST API for face recognition using ArcFace ONNX model with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def enforce_request_size(request: Request, call_next):
    """Reject oversized requests before body parsing when Content-Length is present."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            request_size = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": "InvalidContentLength",
                    "message": "Content-Length header must be a valid non-negative integer",
                },
            )

        if request_size < 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": "InvalidContentLength",
                    "message": "Content-Length header must be a valid non-negative integer",
                },
            )

        if request_size > settings.max_request_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "status": "error",
                    "error": "RequestTooLarge",
                    "message": f"Request exceeds {settings.max_request_bytes} byte limit",
                },
            )

    response = await call_next(request)
    response.headers["X-Robin-Contract-Version"] = "v1"
    return response


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origin_list,
    allow_headers=["Authorization", "Content-Type", "X-Astra-User-Id", "X-Request-ID"],
    expose_headers=["X-Robin-Contract-Version"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
)

# Register routers
app.include_router(health.router)
app.include_router(identification.router)
app.include_router(enrollment.router)

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


if __name__ == "__main__":
    import uvicorn
    
    is_dev = settings.environment == "development"
    
    # Uvicorn does not support using reload with multiple workers.
    # In development (reload=True), always run with a single process (workers=None).
    workers = settings.api_workers
    if is_dev:
        if workers not in (None, 1):
            logger.warning(
                "api_workers=%s is ignored in development because uvicorn does not "
                "support reload with multiple workers. Falling back to workers=None.",
                workers,
            )
        workers = None
    
    uvicorn.run(
        "src.main:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=is_dev,
        workers=workers,
        log_level="info"
    )
