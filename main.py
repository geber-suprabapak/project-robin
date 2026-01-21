"""
Face Recognition API - FastAPI Application
High-performance REST API for face recognition processing with GPU acceleration.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Optional, List

from config import settings
from core.inference_engine import inference_engine
from services.image_decoder import (
    decode_base64_image,
    preprocess_face_image,
    calculate_cosine_distance,
    ImageDecodeError
)
from services.supabase_client import supabase_service
from services.face_detector import validate_single_face, crop_face_from_image, FaceDetectionError
from dependencies import get_admin_api_key
from schemas.api_models import (
    IdentifyRequest,
    IdentifyResponse,
    ErrorResponse,
    HealthResponse,
    EnrollResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (model loading, GPU initialization) and shutdown (cleanup).
    """
    # ===== STARTUP =====
    logger.info("=" * 60)
    logger.info("🚀 Starting Face Recognition API")
    logger.info("=" * 60)
    
    try:
        # Load ONNX model
        logger.info("📦 Loading face recognition model...")
        inference_engine.load_model()
        
        # Warmup GPU/CUDA context
        logger.info("🔥 Warming up inference engine...")
        inference_engine.warmup()
        
        # Check Supabase connection
        if supabase_service.is_connected():
            logger.info("✅ Supabase client connected")
        else:
            logger.warning("⚠️  Supabase client not configured (simulation mode)")
        
        logger.info("=" * 60)
        logger.info(f"✅ Server ready on {settings.api_host}:{settings.api_port}")
        logger.info(f"📊 GPU Enabled: {inference_engine.is_gpu_enabled()}")
        logger.info(f"🔧 Provider: {inference_engine.get_provider()}")
        logger.info(f"🌍 Environment: {settings.environment}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # ===== SHUTDOWN =====
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
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """Check API health status and component availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine.is_loaded(),
        gpu_available=inference_engine.is_gpu_enabled(),
        supabase_connected=supabase_service.is_connected()
    )


@app.get(
    "/",
    tags=["System"],
    summary="Root endpoint"
)
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "gpu_enabled": str(inference_engine.is_gpu_enabled()),
        "docs": "/docs"
    }


# ============================================================================
# Face Identification Endpoint
# ============================================================================

@app.post(
    "/v1/identify",
    response_model=IdentifyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Face Recognition"],
    summary="Identify person from face image",
    responses={
        200: {"description": "Face identified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Face not found"},
        422: {"model": ErrorResponse, "description": "Processing error"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def identify_face(request: IdentifyRequest) -> IdentifyResponse:
    """
    Identify a person from a face image.
    
    Process:
    1. Decode base64 image
    2. Preprocess for model inference
    3. Generate face embedding using GPU
    4. Search for match in Supabase database
    5. Return identification result with confidence score
    
    Args:
        request: IdentifyRequest with image_base64 and camera_id
        
    Returns:
        IdentifyResponse with status, student_id, confidence, and processing time
    """
    start_time = time.perf_counter()
    
    try:
        # ===== Step 1: Decode Image =====
        try:
            image_np = decode_base64_image(request.image_base64)
            logger.info(f"📸 Image decoded - Shape: {image_np.shape}, Camera: {request.camera_id}")
        except ImageDecodeError as e:
            logger.error(f"Image decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # ===== Step 2: Detect and Crop Face =====
        try:
            # Validate single face
            is_valid, face_count, face_message = validate_single_face(image_np)
            
            if not is_valid:
                logger.warning(f"Face validation failed: {face_message}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=face_message
                )
            
            # Crop face region
            cropped_face = crop_face_from_image(image_np, margin=0.2)
            logger.debug(f"Face cropped: {cropped_face.shape}")
        except FaceDetectionError as e:
            logger.error(f"Face detection error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except HTTPException:
            raise
        
        # ===== Step 3: Preprocess Image =====
        try:
            preprocessed = preprocess_face_image(
                cropped_face,  # Use cropped face
                target_size=settings.model_input_size,
                normalize=True
            )
            logger.debug(f"Preprocessed shape: {preprocessed.shape}")
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image preprocessing failed: {str(e)}"
            )
        
        # ===== Step 4: Generate Embedding (GPU Inference) =====
        try:
            embedding = inference_engine.predict(preprocessed)
            logger.info(f"🔬 Embedding generated - Dim: {embedding.shape}")
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Face recognition inference failed: {str(e)}"
            )
        
        # ===== Step 5: Search for Match in Database =====
        try:
            match_result = await supabase_service.find_face_match(
                embedding,
                threshold=settings.face_match_threshold
            )
        except Exception as e:
            logger.error(f"Database search error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database search failed: {str(e)}"
            )
        
        # ===== Step 6: Prepare Response =====
        end_time = time.perf_counter()
        process_time_ms = int((end_time - start_time) * 1000)
        
        if match_result:
            logger.info(
                f"✅ Match found - NIS: {match_result['nis']}, "
                f"Name: {match_result['nama']}, "
                f"Confidence: {match_result['confidence']:.3f}"
            )
            
            return IdentifyResponse(
                status="ok",
                student_id=str(match_result["nis"]),
                student_name=match_result["nama"],
                confidence=match_result["confidence"],
                process_time_ms=process_time_ms,
                message="Face identified successfully"
            )
        else:
            logger.warning("❌ No matching face found in database")
            return IdentifyResponse(
                status="not_found",
                student_id=None,
                student_name=None,
                confidence=None,
                process_time_ms=process_time_ms,
                message="No matching face found in database"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error in identify_face: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Face Enrollment Endpoint (Multi-Image)
# ============================================================================

# Constants for enrollment constraints
MIN_ENROLLMENT_IMAGES = 10
MAX_ENROLLMENT_IMAGES = 20

@app.post(
    "/v1/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_200_OK,
    tags=["Face Recognition"],
    summary="Enroll a student with multiple face images",
    responses={
        200: {"description": "Student enrolled successfully"},
        400: {"model": ErrorResponse, "description": "Invalid images or count"},
        401: {"model": ErrorResponse, "description": "Invalid admin key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def enroll_student(
    files: List[UploadFile] = File(..., description="10-20 face images (JPG/PNG)"),
    name: str = Form(..., description="Student's full name"),
    nisn: str = Form(..., description="Student's unique ID (NIS/NISN)"),
    class_name: Optional[str] = Form(None, description="Class name (optional)"),
    admin_key: str = Depends(get_admin_api_key)
) -> EnrollResponse:
    """
    Enroll a student with multiple face images for improved accuracy.
    
    Process:
    1. Validate image count (10-20 images required)
    2. For each image: decode, validate single face, extract embedding
    3. Store all valid embeddings in Supabase
    
    Args:
        files: Multiple image files (10-20 JPG/PNG images)
        name: Student's full name
        nisn: Student's NIS for user_id lookup
        class_name: Optional class name
        admin_key: Admin API key (from X-Admin-Key header)
        
    Returns:
        EnrollResponse with status, counts, and message
    """
    try:
        # ===== Step 1: Validate Image Count =====
        file_count = len(files)
        
        if file_count < MIN_ENROLLMENT_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Minimum {MIN_ENROLLMENT_IMAGES} images required. Received: {file_count}"
            )
        
        if file_count > MAX_ENROLLMENT_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {MAX_ENROLLMENT_IMAGES} images allowed. Received: {file_count}"
            )
        
        logger.info(f"📸 Enrollment started - Student: {name}, NIS: {nisn}, Images: {file_count}")
        
        # ===== Step 2: Process Each Image =====
        embeddings = []
        failed_images = []
        
        for idx, file in enumerate(files):
            image_name = file.filename or f"image_{idx+1}"
            
            try:
                # Read and decode image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image_np is None:
                    failed_images.append({"index": idx+1, "name": image_name, "error": "Failed to decode"})
                    continue
                
                # Validate single face
                is_valid, face_count, face_message = validate_single_face(image_np)
                
                if not is_valid:
                    failed_images.append({"index": idx+1, "name": image_name, "error": face_message})
                    continue
                
                # Crop face region with margin
                cropped_face = crop_face_from_image(image_np, margin=0.2)
                
                # Preprocess and extract embedding
                preprocessed = preprocess_face_image(
                    cropped_face,  # Use cropped face instead of full image
                    target_size=settings.model_input_size,
                    normalize=True
                )
                embedding = inference_engine.predict(preprocessed)
                embeddings.append(embedding)
                
                logger.debug(f"✅ Image {idx+1}/{file_count} processed: {image_name}")
                
            except Exception as e:
                failed_images.append({"index": idx+1, "name": image_name, "error": str(e)})
                logger.warning(f"⚠️ Image {idx+1} failed: {str(e)}")
        
        # ===== Step 3: Check Minimum Valid Images =====
        if len(embeddings) < MIN_ENROLLMENT_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough valid images. Need {MIN_ENROLLMENT_IMAGES}, got {len(embeddings)}. "
                       f"Failed: {len(failed_images)} images"
            )
        
        logger.info(f"🔬 Extracted {len(embeddings)} embeddings from {file_count} images")
        
        # ===== Step 4: Store in Database =====
        try:
            result = await supabase_service.enroll_student_multi(
                nis=nisn,
                name=name,
                embeddings=embeddings,
                class_name=class_name
            )
            
            if result["success"]:
                logger.info(
                    f"✅ Enrollment complete - NIS: {nisn}, "
                    f"Embeddings: {result['total_embeddings']}"
                )
                return EnrollResponse(
                    status="success",
                    student_id=nisn,
                    images_processed=result["inserted_count"],
                    images_failed=len(failed_images),
                    total_embeddings=result["total_embeddings"],
                    message=f"Student enrolled successfully with {result['total_embeddings']} face images"
                )
            else:
                logger.warning(f"⚠️ Enrollment failed: {result['message']}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["message"]
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database operation failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in enroll_student: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.__class__.__name__,
            "message": exc.detail,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler for uncaught exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.environment == "development" else None
        }
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level="info"
    )
