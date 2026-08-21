from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

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
    monkeypatch.setattr(settings, "jwt_secret", "")

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
            "aud": "astra-api",
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


def test_valid_es256_jwks_token_reaches_protected_endpoint(client, monkeypatch):
    issuer = "https://auth.example.test"
    kid = "test-signing-key"
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_jwk = jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    public_jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    token = jwt.encode(
        {
            "sub": TEST_USER_ID,
            "aud": "astra-api",
            "iss": issuer,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        },
        private_key,
        algorithm="ES256",
        headers={"kid": kid},
    )

    class JwksHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("User-Agent") != "project-robin/1.0":
                self.send_response(403)
                self.end_headers()
                return
            body = json.dumps({"keys": [public_jwk]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), JwksHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(settings, "jwt_jwks_url", f"http://127.0.0.1:{server.server_port}/jwks.json")
        monkeypatch.setattr(settings, "jwt_issuer", issuer)
        monkeypatch.setattr(settings, "jwt_secret", "legacy-secret-must-not-win")
        monkeypatch.setattr(enrollment.qdrant_service, "get_user_embedding_count", _zero_embeddings)

        response = client.get(
            "/v1/enroll/status",
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json()["user_id"] == TEST_USER_ID
