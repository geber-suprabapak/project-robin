import numpy as np

from src.api.routes import identification
from src.services.qdrant_client import QdrantUnavailableError

VALID_BASE64 = "A" * 104


def _patch_identify_pipeline(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    preprocessed = np.zeros((1, 3, 224, 224), dtype=np.float32)
    embedding = np.ones(512, dtype=np.float32)

    monkeypatch.setattr(identification, "decode_base64_image", lambda _: image)
    monkeypatch.setattr(identification, "validate_single_face", lambda _: (True, 1, "ok"))
    monkeypatch.setattr(identification, "crop_face_from_image", lambda image_np, margin=0.2: image_np)
    monkeypatch.setattr(identification, "preprocess_face_image", lambda *args, **kwargs: preprocessed)
    monkeypatch.setattr(identification.inference_engine, "predict", lambda _: embedding)


async def _qdrant_down(*args, **kwargs):
    raise QdrantUnavailableError("Qdrant is down")


async def _no_embeddings(*args, **kwargs):
    return None


def test_identify_returns_503_when_qdrant_is_down(client, monkeypatch, auth_headers):
    _patch_identify_pipeline(monkeypatch)
    monkeypatch.setattr(identification.qdrant_service, "verify_face_1to1", _qdrant_down)

    response = client.post(
        "/v1/identify",
        json={"image_base64": VALID_BASE64},
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["message"] == "Required dependency is unavailable"


def test_identify_reports_not_enrolled_only_when_qdrant_query_succeeds(client, monkeypatch, auth_headers):
    _patch_identify_pipeline(monkeypatch)
    monkeypatch.setattr(identification.qdrant_service, "verify_face_1to1", _no_embeddings)

    response = client.post(
        "/v1/identify",
        json={"image_base64": VALID_BASE64},
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "not_found"
    assert payload["message"] == "No face embeddings enrolled for this user"


def test_identify_rejects_oversized_base64_before_inference(client, monkeypatch, auth_headers):
    monkeypatch.setattr(identification, "decode_base64_image", lambda _: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(identification.settings, "max_image_bytes", 10)

    response = client.post(
        "/v1/identify",
        json={"image_base64": VALID_BASE64},
        headers=auth_headers,
    )

    assert response.status_code == 422
