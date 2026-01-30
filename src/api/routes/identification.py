"""Face identification endpoint."""

import logging
import time
from fastapi import APIRouter, HTTPException, status, Depends

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.image_decoder import decode_base64_image, preprocess_face_image, ImageDecodeError
from src.services.qdrant_client import qdrant_service
from src.services.supabase_client import supabase_service
from src.services.face_detector import validate_single_face, crop_face_from_image, FaceDetectionError
from src.schemas.api_models import IdentifyRequest, IdentifyResponse, ErrorResponse
from src.dependencies import verify_jwt_bearer

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
        401: {"model": ErrorResponse, "description": "Invalid or expired JWT token"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def identify_face(
    request: IdentifyRequest,
    user_id: str = Depends(verify_jwt_bearer)
) -> IdentifyResponse:
    """
    Identify a person from a face image using GPU-accelerated inference.
    
    Requires a valid Supabase JWT Bearer token in Authorization header.
    The user_id from the token's 'sub' claim is verified against the database.
    """
    start_time = time.perf_counter()
    
    try:
        # Step 1: Verify user exists in database
        user_profile = await supabase_service.get_user_profile_by_id(user_id)
        if not user_profile:
            logger.warning(f"🚫 JWT valid but user not found in database: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in database"
            )
        logger.info(f"✅ User verified: {user_id}")
        
        # Step 2: Decode Image
        try:
            image_np = decode_base64_image(request.image_base64)
            logger.info(f"📸 Image decoded - Shape: {image_np.shape}")
        except ImageDecodeError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Step 3: Detect and Crop Face
        try:
            is_valid, _, face_message = validate_single_face(image_np)
            if not is_valid:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=face_message)
            cropped_face = crop_face_from_image(image_np, margin=0.2)
        except FaceDetectionError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Step 4: Preprocess and Generate Embedding
        try:
            preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)
            embedding = inference_engine.predict(preprocessed)
            logger.info(f"🔬 Embedding generated - Dim: {embedding.shape}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {e}")
        
        # Step 5: 1:1 Face Verification using Qdrant (manual similarity, no ANN)
        verify_result = await qdrant_service.verify_face_1to1(
            user_id=user_id,
            query_embedding=embedding,
            threshold=settings.face_match_threshold
        )
        process_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        if verify_result is None:
            # No embeddings enrolled for this user
            logger.warning(f"❌ No face embeddings enrolled for user {user_id}")
            return IdentifyResponse(
                status="not_found", student_id=None, student_name=None, confidence=None,
                process_time_ms=process_time_ms, message="No face embeddings enrolled for this user"
            )
        
        if verify_result["verified"]:
            # Face verified - get student info
            student = await supabase_service.get_student_by_user_id(user_id)
            if student:
                logger.info(f"✅ Verified: NIS={student['nis']}, Conf={verify_result['confidence']:.3f}")
                return IdentifyResponse(
                    status="ok", student_id=str(student["nis"]), student_name=student["nama"],
                    confidence=verify_result["confidence"], process_time_ms=process_time_ms, 
                    message="Face verified successfully"
                )
            else:
                logger.warning(f"⚠️ Face verified but no student profile for user {user_id}")
                return IdentifyResponse(
                    status="ok", student_id=None, student_name=None,
                    confidence=verify_result["confidence"], process_time_ms=process_time_ms,
                    message="Face verified but student profile not found"
                )
        
        logger.warning(f"❌ Face not verified for user {user_id}, confidence: {verify_result['confidence']:.3f}")
        return IdentifyResponse(
            status="not_found", student_id=None, student_name=None, 
            confidence=verify_result["confidence"],
            process_time_ms=process_time_ms, message="Face does not match enrolled face"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")

