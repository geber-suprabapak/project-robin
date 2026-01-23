"""Face enrollment endpoint."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Depends
import numpy as np
import cv2

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.image_decoder import preprocess_face_image
from src.services.supabase_client import supabase_service
from src.services.face_detector import validate_single_face, crop_face_from_image
from src.dependencies import verify_admin_key
from src.schemas.api_models import EnrollResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Face Recognition"])

MIN_ENROLLMENT_IMAGES = 10
MAX_ENROLLMENT_IMAGES = 20


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_200_OK,
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
    admin_key: str = Depends(verify_admin_key)
) -> EnrollResponse:
    """Enroll a student with multiple face images for improved accuracy."""
    try:
        # Validate Image Count
        file_count = len(files)
        if file_count < MIN_ENROLLMENT_IMAGES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Minimum {MIN_ENROLLMENT_IMAGES} images required. Received: {file_count}")
        if file_count > MAX_ENROLLMENT_IMAGES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Maximum {MAX_ENROLLMENT_IMAGES} images allowed. Received: {file_count}")
        
        logger.info(f"📸 Enrollment started - Student: {name}, NIS: {nisn}, Images: {file_count}")
        
        # Process Each Image
        embeddings, failed_images = [], []
        for idx, file in enumerate(files):
            image_name = file.filename or f"image_{idx+1}"
            try:
                contents = await file.read()
                image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
                if image_np is None:
                    failed_images.append({"index": idx+1, "name": image_name, "error": "Failed to decode"})
                    continue
                
                is_valid, _, face_message = validate_single_face(image_np)
                if not is_valid:
                    failed_images.append({"index": idx+1, "name": image_name, "error": face_message})
                    continue
                
                cropped_face = crop_face_from_image(image_np, margin=0.2)
                preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)
                embeddings.append(inference_engine.predict(preprocessed))
                logger.debug(f"✅ Image {idx+1}/{file_count} processed: {image_name}")
            except Exception as e:
                failed_images.append({"index": idx+1, "name": image_name, "error": str(e)})
                logger.warning(f"⚠️ Image {idx+1} failed: {e}")
        
        # Check Minimum Valid Images
        if len(embeddings) < MIN_ENROLLMENT_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough valid images. Need {MIN_ENROLLMENT_IMAGES}, got {len(embeddings)}. Failed: {len(failed_images)}"
            )
        
        logger.info(f"🔬 Extracted {len(embeddings)} embeddings from {file_count} images")
        
        # Store in Database
        result = await supabase_service.enroll_student_multi(nis=nisn, name=name, embeddings=embeddings, class_name=class_name)
        
        if result["success"]:
            logger.info(f"✅ Enrollment complete - NIS: {nisn}, Embeddings: {result['total_embeddings']}")
            return EnrollResponse(
                status="success", student_id=nisn, images_processed=result["inserted_count"],
                images_failed=len(failed_images), total_embeddings=result["total_embeddings"],
                message=f"Student enrolled successfully with {result['total_embeddings']} face images"
            )
        
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
