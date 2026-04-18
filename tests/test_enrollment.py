import numpy as np

from src.api.routes import enrollment
from src.services.qdrant_client import QdrantUnavailableError


def _files(count: int, content_type: str = "image/png"):
    return [
        ("files", (f"image_{idx}.png", b"not-a-real-image", content_type))
        for idx in range(count)
    ]


def _patch_enrollment_pipeline(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    preprocessed = np.zeros((1, 3, 224, 224), dtype=np.float32)
    embedding = np.ones(512, dtype=np.float32)

    monkeypatch.setattr(enrollment, "decode_image_bytes", lambda *args, **kwargs: image)
    monkeypatch.setattr(enrollment, "validate_single_face", lambda _: (True, 1, "ok"))
    monkeypatch.setattr(enrollment, "crop_face_from_image", lambda image_np, margin=0.2: image_np)
    monkeypatch.setattr(enrollment, "preprocess_face_image", lambda *args, **kwargs: preprocessed)
    monkeypatch.setattr(enrollment.inference_engine, "predict", lambda _: embedding)


async def _user_profile(user_id: str):
    return {"user_id": user_id}


async def _qdrant_down(*args, **kwargs):
    raise QdrantUnavailableError("Qdrant is down")


def test_enroll_requires_exactly_ten_files(client, auth_headers):
    response = client.post(
        "/v1/enroll",
        files=_files(1),
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Exactly 10 images required" in response.json()["message"]


def test_enroll_rejects_corrupt_image_even_without_mime_validation(client, monkeypatch, auth_headers):
    monkeypatch.setattr(enrollment.supabase_service, "get_user_profile_by_id", _user_profile)

    response = client.post(
        "/v1/enroll",
        files=_files(10, content_type="text/plain"),
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "All 10 images must be valid" in response.json()["message"]


def test_enroll_returns_503_when_qdrant_write_fails(client, monkeypatch, auth_headers):
    _patch_enrollment_pipeline(monkeypatch)
    monkeypatch.setattr(enrollment.supabase_service, "get_user_profile_by_id", _user_profile)
    monkeypatch.setattr(enrollment.qdrant_service, "enroll_user_embeddings", _qdrant_down)

    response = client.post(
        "/v1/enroll",
        files=_files(10),
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["message"] == "Required dependency is unavailable"
