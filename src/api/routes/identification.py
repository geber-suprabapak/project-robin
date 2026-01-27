"""Face identification endpoint with anti-spoofing."""

import logging
import time
from fastapi import APIRouter, HTTPException, status, Depends

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.image_decoder import decode_base64_image, preprocess_face_image, ImageDecodeError
from src.services.supabase_client import supabase_service
from src.services.face_detector import detect_faces, crop_face_from_image, FaceDetectionError
from src.services.anti_spoof_detector import anti_spoof_detector
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
    
    Includes anti-spoofing liveness detection to prevent spoof attacks.
    """
    start_time = time.perf_counter()
    
    # Track liveness result for response
    liveness_result = {"is_live": None, "score": None}
    
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
        
        # Step 3: Detect Face and Get Bounding Box
        try:
            faces = detect_faces(image_np)
            if len(faces) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in the image. Please use a clear photo with a visible face."
                )
            elif len(faces) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Multiple faces detected ({len(faces)}). Please use a solo photo."
                )
            
            face_bbox = faces[0]  # (x, y, w, h)
            logger.info(f"👤 Face detected: bbox={face_bbox}")
            
        except FaceDetectionError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Step 4: Anti-Spoofing Check (Liveness Detection)
        if settings.anti_spoof_enabled:
            try:
                is_live, liveness_score = anti_spoof_detector.check_liveness(image_np, face_bbox)
                liveness_result = {"is_live": is_live, "score": liveness_score}
                
                logger.info(f"🛡️ Liveness check: is_live={is_live}, score={liveness_score:.3f}")
                
                if not is_live:
                    process_time_ms = int((time.perf_counter() - start_time) * 1000)
                    logger.warning(f"⚠️ Spoof detected for user {user_id}, score={liveness_score:.3f}")
                    return IdentifyResponse(
                        status="spoof_detected",
                        student_id=None,
                        student_name=None,
                        confidence=None,
                        is_live=False,
                        liveness_score=liveness_score,
                        process_time_ms=process_time_ms,
                        message="Liveness check failed - possible spoof attack detected"
                    )
                    
            except Exception as e:
                logger.error(f"Anti-spoof check failed: {e}")
                # For security, deny on error
                process_time_ms = int((time.perf_counter() - start_time) * 1000)
                return IdentifyResponse(
                    status="spoof_detected",
                    student_id=None,
                    student_name=None,
                    confidence=None,
                    is_live=False,
                    liveness_score=0.0,
                    process_time_ms=process_time_ms,
                    message=f"Liveness check error: {str(e)}"
                )
        else:
            # Anti-spoofing disabled
            liveness_result = {"is_live": True, "score": 1.0}
        
        # Step 5: Crop Face and Preprocess
        try:
            cropped_face = crop_face_from_image(image_np, margin=0.2)
            preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)
            embedding = inference_engine.predict(preprocessed)
            logger.info(f"🔬 Embedding generated - Dim: {embedding.shape}")
        except FaceDetectionError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {e}")
        
        # Step 6: 1:1 Face Verification (compare against THIS user's embeddings only)
        verify_result = await supabase_service.verify_face_1to1(
            user_id=user_id,
            embedding=embedding,
            threshold=settings.face_match_threshold
        )
        process_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        if verify_result is None:
            # No embeddings enrolled for this user
            logger.warning(f"❌ No face embeddings enrolled for user {user_id}")
            return IdentifyResponse(
                status="not_found",
                student_id=None,
                student_name=None,
                confidence=None,
                is_live=liveness_result["is_live"],
                liveness_score=liveness_result["score"],
                process_time_ms=process_time_ms,
                message="No face embeddings enrolled for this user"
            )
        
        if verify_result["verified"]:
            # Face verified - get student info
            student = await supabase_service.get_student_by_user_id(user_id)
            if student:
                logger.info(f"✅ Verified: NIS={student['nis']}, Conf={verify_result['confidence']:.3f}")
                return IdentifyResponse(
                    status="ok",
                    student_id=str(student["nis"]),
                    student_name=student["nama"],
                    confidence=verify_result["confidence"],
                    is_live=liveness_result["is_live"],
                    liveness_score=liveness_result["score"],
                    process_time_ms=process_time_ms,
                    message="Face verified successfully"
                )
            else:
                logger.warning(f"⚠️ Face verified but no student profile for user {user_id}")
                return IdentifyResponse(
                    status="ok",
                    student_id=None,
                    student_name=None,
                    confidence=verify_result["confidence"],
                    is_live=liveness_result["is_live"],
                    liveness_score=liveness_result["score"],
                    process_time_ms=process_time_ms,
                    message="Face verified but student profile not found"
                )
        
        logger.warning(f"❌ Face not verified for user {user_id}, confidence: {verify_result['confidence']:.3f}")
        return IdentifyResponse(
            status="not_found",
            student_id=None,
            student_name=None,
            confidence=verify_result["confidence"],
            is_live=liveness_result["is_live"],
            liveness_score=liveness_result["score"],
            process_time_ms=process_time_ms,
            message="Face does not match enrolled face"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
