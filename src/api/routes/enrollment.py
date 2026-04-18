"""Face enrollment endpoint - Self-serve with JWT authentication."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends

from src.config import settings
from src.core.inference_engine import inference_engine
from src.services.image_decoder import ImageDecodeError, decode_image_bytes, preprocess_face_image
from src.services.qdrant_client import QdrantServiceError, qdrant_service
from src.services.supabase_client import SupabaseServiceError, supabase_service
from src.services.face_detector import validate_single_face, crop_face_from_image
from src.dependencies import verify_jwt_bearer
from src.schemas.api_models import EnrollResponse, EnrollStatusResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Face Recognition"])

REQUIRED_IMAGES = 10
READ_CHUNK_SIZE = 1024 * 1024


async def _read_upload_file_limited(file: UploadFile) -> bytes:
    """Read an uploaded image without allowing unbounded memory use."""
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break

        total_size += len(chunk)
        if total_size > settings.max_image_bytes:
            raise ImageDecodeError(f"Image exceeds {settings.max_image_bytes} byte limit")
        chunks.append(chunk)

    return b"".join(chunks)


@router.get(
    "/enroll/status",
    response_model=EnrollStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check user enrollment status",
    responses={
        200: {"description": "Enrollment status retrieved"},
        401: {"model": ErrorResponse, "description": "Invalid or expired JWT token"},
        503: {"model": ErrorResponse, "description": "Dependency unavailable"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def check_enrollment_status(
    user_id: str = Depends(verify_jwt_bearer)
) -> EnrollStatusResponse:
    """
    Check if current user has face embeddings enrolled.
    
    Requires a valid Supabase JWT Bearer token in Authorization header.
    """
    try:
        count = await qdrant_service.get_user_embedding_count(user_id)
        return EnrollStatusResponse(
            is_enrolled=count > 0,
            embedding_count=count,
            user_id=user_id
        )
    except QdrantServiceError as e:
        logger.exception(f"Qdrant error checking enrollment status: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding store is unavailable"
        )
    except Exception as e:
        logger.exception(f"Error checking enrollment status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check enrollment status: {e}"
        )


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_200_OK,
    summary="Self-serve face enrollment with 10 images",
    responses={
        200: {"description": "User enrolled successfully"},
        400: {"model": ErrorResponse, "description": "Invalid images or count"},
        401: {"model": ErrorResponse, "description": "Invalid or expired JWT token"},
        503: {"model": ErrorResponse, "description": "Dependency unavailable"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def enroll_user(
    files: List[UploadFile] = File(..., description=f"Exactly {REQUIRED_IMAGES} face images (JPG/PNG)"),
    user_id: str = Depends(verify_jwt_bearer)
) -> EnrollResponse:
    """
    Self-serve face enrollment with exactly 10 images.
    
    Requires a valid Supabase JWT Bearer token in Authorization header.
    The user_id is extracted from the token's 'sub' claim.
    """
    try:
        # Step 1: Validate Image Count
        file_count = len(files)
        if file_count != REQUIRED_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Exactly {REQUIRED_IMAGES} images required. Received: {file_count}"
            )
        
        # Step 2: Verify user exists in database
        user_profile = await supabase_service.get_user_profile_by_id(user_id)
        if not user_profile:
            logger.warning(f"🚫 JWT valid but user not found in database: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in database"
            )
        
        logger.info(f"📸 Self-serve enrollment started - user_id: {user_id}, Images: {file_count}")
        
        # Step 3: Process Each Image
        embeddings, failed_images = [], []
        for idx, file in enumerate(files):
            image_name = file.filename or f"image_{idx+1}"
            try:
                contents = await _read_upload_file_limited(file)
                image_np = decode_image_bytes(contents)
                
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
        
        # Step 4: Check All Images Processed Successfully
        if len(embeddings) != REQUIRED_IMAGES:
            failed_count = len(failed_images)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All {REQUIRED_IMAGES} images must be valid. {failed_count} failed processing."
            )
        
        logger.info(f"🔬 Extracted {len(embeddings)} embeddings from {file_count} images")
        
        # Step 5: Store in Qdrant
        result = await qdrant_service.enroll_user_embeddings(
            user_id=user_id,
            embeddings=embeddings,
            model_name=settings.model_path.split("/")[-1].replace(".onnx", "")
        )
        
        logger.info(f"✅ Enrollment complete - user_id: {user_id}, Embeddings: {result['total_embeddings']}")
        return EnrollResponse(
            status="success",
            student_id=user_id,
            images_processed=result["inserted_count"],
            images_failed=len(failed_images),
            total_embeddings=result["total_embeddings"],
            message=f"Face enrolled successfully with {result['total_embeddings']} images"
        )
    except HTTPException:
        raise
    except (QdrantServiceError, SupabaseServiceError) as e:
        logger.exception(f"Dependency error during enrollment: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required dependency is unavailable"
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
