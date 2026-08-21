"""
Pydantic models for API request and response schemas.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.base64_utils import BASE64_RE, estimate_base64_decoded_size
from src.config import settings


class IdentifyRequest(BaseModel):
    """Request model for face identification endpoint."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image string",
        min_length=100
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Validate base64 shape and size without decoding the full payload."""
        if "," in value:
            _, value = value.split(",", 1)

        compact = "".join(value.split())
        if len(compact) % 4 != 0:
            raise ValueError("Invalid base64 string length")
        if not BASE64_RE.fullmatch(compact):
            raise ValueError("Invalid base64 characters")
        if estimate_base64_decoded_size(compact) > settings.max_image_bytes:
            raise ValueError(f"Image exceeds {settings.max_image_bytes} byte limit")
        return compact

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
            }
        }
    )


class IdentifyResponse(BaseModel):
    """Response model for face identification endpoint."""

    status: str = Field(
        ...,
        description="Status of the operation (ok, error, not_found)"
    )
    student_id: Optional[str] = Field(
        None,
        description="Identified student NIS (if found)"
    )
    student_name: Optional[str] = Field(
        None,
        description="Student's full name (if found)"
    )
    confidence: Optional[float] = Field(
        None,
        description="Match confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    process_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds",
        ge=0
    )
    message: Optional[str] = Field(
        None,
        description="Additional information or error message"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "student_id": "12345678",
                "student_name": "Ahmad Rizki",
                "confidence": 0.95,
                "process_time_ms": 45,
                "message": "Face identified successfully"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    status: str = Field(default="error", description="Status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "error": "InvalidImageError",
                "message": "Failed to decode image",
                "detail": "Corrupt or invalid image format"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether ONNX model is loaded")
    face_detector_ready: bool = Field(..., description="Whether the face detector is available")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    qdrant_connected: bool = Field(..., description="Whether Qdrant is connected")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "face_detector_ready": True,
                "gpu_available": True,
                "qdrant_connected": True
            }
        }
    )


class EnrollResponse(BaseModel):
    """Response model for multi-image student enrollment endpoint."""

    status: str = Field(
        ...,
        description="Status of the operation (success, partial, or error)"
    )
    student_id: Optional[str] = Field(
        None,
        description="Student's NIS (Nomor Induk Siswa)"
    )
    images_processed: int = Field(
        0,
        description="Number of images successfully processed"
    )
    images_failed: int = Field(
        0,
        description="Number of images that failed processing"
    )
    total_embeddings: int = Field(
        0,
        description="Total embeddings now stored for this student"
    )
    message: str = Field(
        ...,
        description="Human-readable result message"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "student_id": "12345678",
                "images_processed": 15,
                "images_failed": 0,
                "total_embeddings": 15,
                "message": "Student enrolled successfully with 15 face images"
            }
        }
    )


class EnrollStatusResponse(BaseModel):
    """Response model for enrollment status check endpoint."""

    is_enrolled: bool = Field(
        ...,
        description="Whether user has face embeddings enrolled"
    )
    embedding_count: int = Field(
        ...,
        description="Number of face embeddings stored for this user",
        ge=0
    )
    user_id: str = Field(
        ...,
        description="User ID from JWT token"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_enrolled": True,
                "embedding_count": 10,
                "user_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )
