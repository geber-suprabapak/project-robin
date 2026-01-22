"""Face identification endpoint."""

import logging
import time
from fastapi import APIRouter, HTTPException, status

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.image_decoder import decode_base64_image, preprocess_face_image, ImageDecodeError
from src.services.supabase_client import supabase_service
from src.services.face_detector import validate_single_face, crop_face_from_image, FaceDetectionError
from src.schemas.api_models import IdentifyRequest, IdentifyResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Face Recognition"])


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Identify person from face image",
    responses={
        200: {"description": "Face identified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def identify_face(request: IdentifyRequest) -> IdentifyResponse:
    """Identify a person from a face image using GPU-accelerated inference."""
    start_time = time.perf_counter()
    
    try:
        # Decode Image
        try:
            image_np = decode_base64_image(request.image_base64)
            logger.info(f"📸 Image decoded - Shape: {image_np.shape}, Camera: {request.camera_id}")
        except ImageDecodeError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Detect and Crop Face
        try:
            is_valid, _, face_message = validate_single_face(image_np)
            if not is_valid:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=face_message)
            cropped_face = crop_face_from_image(image_np, margin=0.2)
        except FaceDetectionError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Preprocess and Generate Embedding
        try:
            preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)
            embedding = inference_engine.predict(preprocessed)
            logger.info(f"🔬 Embedding generated - Dim: {embedding.shape}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {e}")
        
        # Search for Match
        match_result = await supabase_service.find_face_match(embedding, threshold=settings.face_match_threshold)
        process_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        if match_result:
            logger.info(f"✅ Match: NIS={match_result['nis']}, Conf={match_result['confidence']:.3f}")
            return IdentifyResponse(
                status="ok", student_id=str(match_result["nis"]), student_name=match_result["nama"],
                confidence=match_result["confidence"], process_time_ms=process_time_ms, message="Face identified successfully"
            )
        
        logger.warning("❌ No matching face found")
        return IdentifyResponse(
            status="not_found", student_id=None, student_name=None, confidence=None,
            process_time_ms=process_time_ms, message="No matching face found in database"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
