import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_rejects_wildcard_cors_with_credentials():
    with pytest.raises(ValidationError, match="requires explicit CORS_ALLOWED_ORIGINS"):
        Settings(cors_allowed_origins="*", cors_allow_credentials=True)


def test_settings_allows_explicit_cors_with_credentials():
    settings = Settings(
        cors_allowed_origins="https://app.example.com, https://admin.example.com",
        cors_allow_credentials=True,
    )

    assert settings.cors_allowed_origin_list == [
        "https://app.example.com",
        "https://admin.example.com",
    ]
