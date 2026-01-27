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
    environment: str = Field(default="development", description="Environment")
    
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
    max_cosine_distance: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Maximum cosine distance for face matching"
    )
    
    # Admin Security Configuration
    admin_secret_key: str = Field(
        default="",
        description="Secret key for admin-only endpoints (e.g., enrollment)"
    )
    
    # Anti-Spoofing Configuration
    anti_spoof_enabled: bool = Field(
        default=True,
        description="Enable anti-spoofing liveness detection"
    )
    anti_spoof_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum liveness score to pass anti-spoofing check (0.0-1.0)"
    )
    anti_spoof_model_path_v2_scale27: str = Field(
        default="./models/anti_spoof_2.7_80x80.onnx",
        description="Path to MiniFASNetV2 ONNX model for scale 2.7"
    )
    anti_spoof_model_path_v2_scale40: str = Field(
        default="./models/anti_spoof_4.0_80x80.onnx",
        description="Path to MiniFASNetV2 ONNX model for scale 4.0"
    )


# Global settings instance
settings = Settings()
