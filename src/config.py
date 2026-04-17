"""
Configuration module using Pydantic Settings.
Loads environment variables from .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Model Configuration
    model_path: str = Field(
        default="./models/arcface_r100_224x224.onnx",
        description="Path to ONNX model file"
    )
    model_input_size: int = Field(
        default=224,
        description="Model input size (224 for Hi-Res ArcFace)"
    )
    embedding_dim: int = Field(
        default=512,
        description="Face embedding vector dimension"
    )
    skip_model_load: bool = Field(
        default=False,
        description="Skip ONNX model loading for CI smoke tests"
    )
    
    # GPU Configuration
    gpu_device_id: int = Field(
        default=0,
        description="GPU device ID (0 for first GPU, -1 for CPU only)"
    )
    gpu_mem_limit: int = Field(
        default=2147483648,  # 2GB
        description="GPU memory limit in bytes"
    )
    
    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of uvicorn workers")
    environment: str = Field(default="production", description="Environment")
    
    # CORS Configuration
    cors_allowed_origins: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins. Use * for native/mobile clients."
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Whether browser credentials such as cookies are allowed in CORS requests"
    )
    
    # Request / Image Guardrails
    max_request_bytes: int = Field(
        default=62_914_560,
        description="Maximum HTTP request size accepted by the API"
    )
    max_image_bytes: int = Field(
        default=5_242_880,
        description="Maximum decoded image file size"
    )
    max_image_width: int = Field(default=4096, description="Maximum accepted image width")
    max_image_height: int = Field(default=4096, description="Maximum accepted image height")
    max_image_pixels: int = Field(default=16_777_216, description="Maximum accepted image pixel count")
    
    # Supabase Configuration
    supabase_url: str = Field(
        default="",
        description="Supabase project URL"
    )
    supabase_key: str = Field(
        default="",
        description="Supabase anon/public key"
    )
    supabase_service_role_key: str = Field(
        default="",
        description="Supabase service role key (admin access)"
    )
    supabase_timeout_seconds: float = Field(
        default=3.0,
        description="Timeout for Supabase operations"
    )
    supabase_max_retries: int = Field(
        default=2,
        ge=0,
        description="Retry count for transient Supabase operation failures"
    )
    
    # Security - JWT
    supabase_jwt_secret: str = Field(
        default="",
        description="Supabase JWT secret for verifying session tokens"
    )
    
    # Face Recognition Configuration
    face_match_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for face matching"
    )
    
    # Face Detector Assets
    face_detector_prototxt_path: str = Field(
        default="/app/models/face_detector/deploy.prototxt",
        description="Path to the OpenCV DNN face detector prototxt file"
    )
    face_detector_model_path: str = Field(
        default="/app/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        description="Path to the OpenCV DNN face detector model file"
    )
    
    # Qdrant Configuration
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant host (localhost for Docker, or Qdrant Cloud URL)"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant REST API port"
    )
    qdrant_collection_name: str = Field(
        default="face_embeddings",
        description="Qdrant collection name for face embeddings"
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API key (optional, for Qdrant Cloud)"
    )
    qdrant_https: bool = Field(
        default=False,
        description="Whether to use HTTPS for Qdrant connection"
    )
    qdrant_timeout_seconds: float = Field(
        default=3.0,
        description="Timeout for Qdrant operations"
    )

    @property
    def cors_allowed_origin_list(self) -> list[str]:
        """Configured CORS origins normalized as a list."""
        return [
            value.strip()
            for value in self.cors_allowed_origins.split(",")
            if value.strip()
        ] or ["*"]


# Global settings instance
settings = Settings()
