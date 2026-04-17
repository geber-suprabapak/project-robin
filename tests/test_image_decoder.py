import base64
from io import BytesIO

import pytest
from PIL import Image

from src.config import settings
from src.services.image_decoder import ImageDecodeError, decode_base64_image, decode_image_bytes, validate_image_bytes


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_bytes_accepts_valid_png():
    image = decode_image_bytes(_png_bytes())

    assert image.shape == (8, 8, 3)


def test_validate_image_bytes_rejects_corrupt_image():
    with pytest.raises(ImageDecodeError, match="corrupt or unsupported format"):
        validate_image_bytes(b"not-a-real-image")


def test_validate_image_bytes_rejects_oversized_image(monkeypatch):
    monkeypatch.setattr(settings, "max_image_bytes", 4)

    with pytest.raises(ImageDecodeError, match="byte limit"):
        validate_image_bytes(_png_bytes())


def test_validate_image_bytes_rejects_large_dimensions(monkeypatch):
    monkeypatch.setattr(settings, "max_image_width", 4)

    with pytest.raises(ImageDecodeError, match="dimensions exceed"):
        validate_image_bytes(_png_bytes(width=8))


def test_decode_base64_ignores_data_uri_mime_metadata():
    payload = base64.b64encode(_png_bytes()).decode("ascii")

    image = decode_base64_image(f"data:text/plain;base64,{payload}")

    assert image.shape == (8, 8, 3)
