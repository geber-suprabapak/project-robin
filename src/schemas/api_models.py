"""
Pydantic models for API request and response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
import base64


class IdentifyRequest(BaseModel):
    """Request model for face identification endpoint."""
    
    image_base64: str = Field(
        ...,
        description="Base64-encoded image string",
        min_length=100
    )
    
    @validator("image_base64")
    def validate_base64(cls, v: str) -> str:
        """Validate that the string is valid base64."""
        try:
            # Remove data URI prefix if present
            if "," in v:
                v = v.split(",")[1]
            
            # Try to decode to validate
            base64.b64decode(v, validate=True)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 string: {str(e)}")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
        }


class IdentifyResponse(BaseModel):
    """Response model for face identification endpoint."""
    
    status: str = Field(
        ...,
        description="Status of the operation (ok, error, not_found, spoof_detected)"
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
    is_live: Optional[bool] = Field(
        None,
        description="Liveness check result (True = real face, False = spoof detected)"
    )
    liveness_score: Optional[float] = Field(
        None,
        description="Liveness confidence score (0.0 = fake, 1.0 = real)",
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "student_id": "12345678",
                "student_name": "Ahmad Rizki",
                "confidence": 0.95,
                "is_live": True,
                "liveness_score": 0.89,
                "process_time_ms": 45,
                "message": "Face identified successfully"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = Field(default="error", description="Status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "InvalidImageError",
                "message": "Failed to decode image",
                "detail": "Corrupt or invalid image format"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether ONNX model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    supabase_connected: bool = Field(..., description="Whether Supabase is connected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": True,
                "supabase_connected": True
            }
        }


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "student_id": "12345678",
                "images_processed": 15,
                "images_failed": 0,
                "total_embeddings": 15,
                "message": "Student enrolled successfully with 15 face images"
            }
        }
