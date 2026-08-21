"""
Configuration module using Pydantic Settings.
Loads environment variables from .env file.
"""

from pydantic import AliasChoices, Field, model_validator
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
        default="/app/runtime-models/glintr100.onnx",
        description="Path to ONNX model file"
    )
    model_input_size: int = Field(
        default=112,
        description="Model input size (112 for default AuraFace glintr100)"
    )
    embedding_dim: int = Field(
        default=512,
        description="Face embedding vector dimension"
    )
    skip_model_load: bool = Field(
        default=False,
        description="Skip ONNX model loading for CI smoke tests"
    )
    auto_download_models: bool = Field(
        default=False,
        description="Download missing runtime model assets only when explicitly enabled"
    )
    model_download_url: str = Field(
        default=(
            "https://huggingface.co/fal/AuraFace-v1/resolve/"
            "686774cad65e40933022896195e07e01ee06bee9/glintr100.onnx"
        ),
        description="URL used to download the default face recognition ONNX model"
    )
    model_download_checksum: str = Field(
        default="sha256:a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
        description="Checksum for the default face recognition ONNX model"
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

    # Security - JWT Authentication
    jwt_secret: str = Field(
        default="",
        validation_alias=AliasChoices("jwt_secret", "auth_jwt_secret"),
        description="JWT secret for verifying symmetric OIDC access tokens",
    )
    jwt_jwks_url: str = Field(
        default="",
        validation_alias=AliasChoices("jwt_jwks_url", "auth_jwks_url"),
        description="JWKS endpoint for verifying OIDC access tokens",
    )
    jwt_issuer: str = Field(
        default="",
        validation_alias=AliasChoices("jwt_issuer", "auth_jwt_issuer"),
        description="Expected OIDC issuer in access tokens",
    )
    jwt_audience: str = Field(
        default="astra-api",
        validation_alias=AliasChoices("jwt_audience", "auth_jwt_audience"),
        description="Expected OIDC audience in access tokens",
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
    face_detector_prototxt_url: str = Field(
        default="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        description="URL used to download the OpenCV DNN face detector prototxt"
    )
    face_detector_prototxt_checksum: str = Field(
        default="sha256:dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9",
        description="Checksum for the OpenCV DNN face detector prototxt"
    )
    face_detector_model_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
            "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        ),
        description="URL used to download the OpenCV DNN face detector Caffe model"
    )
    face_detector_model_checksum: str = Field(
        default="sha256:2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d",
        description="Checksum for the OpenCV DNN face detector Caffe model"
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

    @model_validator(mode="after")
    def validate_cors_credentials(self) -> "Settings":
        """Reject invalid wildcard CORS config when credentials are enabled."""
        if self.cors_allow_credentials and "*" in self.cors_allowed_origin_list:
            raise ValueError(
                "CORS_ALLOW_CREDENTIALS=true requires explicit CORS_ALLOWED_ORIGINS; wildcard '*' is invalid"
            )
        return self


# Global settings instance
settings = Settings()
