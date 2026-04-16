"""
Face Recognition API - FastAPI Application
High-performance REST API for face recognition processing with GPU acceleration.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.supabase_client import supabase_service
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
        
        if supabase_service.is_connected():
            logger.info("✅ Supabase client connected")
        else:
            logger.warning("⚠️ Supabase client not configured (simulation mode)")
        
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
