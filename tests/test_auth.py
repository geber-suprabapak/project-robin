from src.api.routes import enrollment
from tests.helpers import TEST_USER_ID


async def _zero_embeddings(user_id: str) -> int:
    return 0


def test_missing_bearer_token_returns_401(client):
    response = client.get("/v1/enroll/status")
    assert response.status_code == 401
    assert response.json()["message"] == "Missing authorization header"


def test_malformed_bearer_token_returns_401(client):
    response = client.get("/v1/enroll/status", headers={"Authorization": "Token abc"})
    assert response.status_code == 401
    assert "Invalid authorization header format" in response.json()["message"]


def test_invalid_service_credential_returns_401(client):
    response = client.get(
        "/v1/enroll/status",
        headers={
            "Authorization": "Bearer user-token",
            "X-Astra-User-Id": TEST_USER_ID,
        },
    )
    assert response.status_code == 401
    assert response.json()["message"] == "Invalid Robin service credential"


def test_missing_user_context_returns_401(client):
    response = client.get(
        "/v1/enroll/status",
        headers={"Authorization": "Bearer dev-robin-service-token"},
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
