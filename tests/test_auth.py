import pytest

from src.api.routes import enrollment
from src.api.routes import identification
from tests.helpers import TEST_USER_ID


async def _zero_embeddings(user_id: str) -> int:
    return 0


async def _unexpected_qdrant_call(*args, **kwargs):
    raise AssertionError("Invalid requests must not reach Qdrant")


def _unexpected_inference_call(*args, **kwargs):
    raise AssertionError("Invalid requests must not reach inference")


def test_missing_bearer_token_returns_401(client):
    response = client.get(
        "/v1/enroll/status",
        headers={"X-Request-ID": "test-request-id", "X-Astra-Contract-Version": "v1"},
    )
    assert response.status_code == 401
    assert response.json()["message"] == "Missing authorization header"


def test_malformed_bearer_token_returns_401(client):
    response = client.get(
        "/v1/enroll/status",
        headers={"Authorization": "Token abc", "X-Request-ID": "test-request-id", "X-Astra-Contract-Version": "v1"},
    )
    assert response.status_code == 401
    assert "Invalid authorization header format" in response.json()["message"]


def test_invalid_service_credential_returns_401(client, auth_headers):
    headers = auth_headers.copy()
    headers["Authorization"] = "Bearer user-token"
    response = client.get(
        "/v1/enroll/status",
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json()["message"] == "Invalid Robin service credential"


def test_missing_user_context_returns_401(client, auth_headers):
    headers = auth_headers.copy()
    headers.pop("X-Astra-User-Id")
    response = client.get(
        "/v1/enroll/status",
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json()["message"] == "Missing Astra user context"


def test_valid_service_credential_reaches_protected_endpoint(client, monkeypatch, auth_headers):
    monkeypatch.setattr(enrollment.qdrant_service, "get_user_embedding_count", _zero_embeddings)
    response = client.get("/v1/enroll/status", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {
        "is_enrolled": False,
        "embedding_count": 0,
        "user_id": TEST_USER_ID,
    }
    assert response.headers["X-Robin-Contract-Version"] == "v1"


@pytest.mark.parametrize(
    ("header", "value", "expected_error"),
    [
        ("X-Request-ID", None, "MissingRequestId"),
        ("X-Request-ID", " ", "MissingRequestId"),
        ("X-Astra-Contract-Version", None, "MissingAstraContractVersion"),
        ("X-Astra-Contract-Version", "v2", "UnsupportedAstraContractVersion"),
    ],
)
def test_invalid_contract_headers_reject_before_qdrant(client, monkeypatch, auth_headers, header, value, expected_error):
    monkeypatch.setattr(enrollment.qdrant_service, "get_user_embedding_count", _unexpected_qdrant_call)
    headers = auth_headers.copy()
    if value is None:
        headers.pop(header)
    else:
        headers[header] = value

    response = client.get("/v1/enroll/status", headers=headers)

    assert response.status_code == 400
    assert response.json()["error"] == expected_error


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/enroll/status"),
        ("POST", "/v1/identify"),
        ("POST", "/v1/enroll"),
        ("DELETE", "/v1/enroll"),
    ],
)
def test_missing_contract_version_rejects_all_private_operations_before_processing(
    client, monkeypatch, auth_headers, method, path
):
    for name in ("get_user_embedding_count", "verify_face_1to1", "enroll_user_embeddings", "delete_user_embeddings"):
        monkeypatch.setattr(enrollment.qdrant_service, name, _unexpected_qdrant_call)
    monkeypatch.setattr(identification, "decode_base64_image", _unexpected_inference_call)
    monkeypatch.setattr(enrollment, "decode_image_bytes", _unexpected_inference_call)
    monkeypatch.setattr(identification.inference_engine, "predict", _unexpected_inference_call)

    headers = auth_headers.copy()
    headers.pop("X-Astra-Contract-Version")
    request_data = {"json": {"image_base64": "A" * 104}} if path == "/v1/identify" else {}
    response = client.request(method, path, headers=headers, **request_data)

    assert response.status_code == 400
    assert response.json()["error"] == "MissingAstraContractVersion"
