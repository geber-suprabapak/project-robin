from datetime import datetime, timedelta, timezone

import jwt

from src.api.routes import enrollment
from src.config import settings
from tests.helpers import TEST_USER_ID, make_token


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


def test_jwt_secret_missing_returns_500(client, monkeypatch):
    monkeypatch.setattr(settings, "supabase_jwt_secret", "")

    response = client.get("/v1/enroll/status", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["message"] == "JWT authentication not configured"


def test_expired_jwt_returns_401(client):
    response = client.get(
        "/v1/enroll/status",
        headers={"Authorization": f"Bearer {make_token(expired=True)}"},
    )

    assert response.status_code == 401
    assert response.json()["message"] == "Token has expired"


def test_invalid_jwt_returns_401(client):
    token = jwt.encode(
        {
            "sub": TEST_USER_ID,
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        },
        "wrong-secret",
        algorithm="HS256",
    )

    response = client.get("/v1/enroll/status", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert "Invalid token" in response.json()["message"]


def test_valid_jwt_reaches_protected_endpoint(client, monkeypatch, auth_headers):
    monkeypatch.setattr(enrollment.qdrant_service, "get_user_embedding_count", _zero_embeddings)

    response = client.get("/v1/enroll/status", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {
        "is_enrolled": False,
        "embedding_count": 0,
        "user_id": TEST_USER_ID,
    }
