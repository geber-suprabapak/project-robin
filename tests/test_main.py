from src.config import settings


async def _ok_response(request):
    return None


def test_invalid_content_length_returns_400(client):
    response = client.get("/live", headers={"Content-Length": "not-a-number"})

    assert response.status_code == 400
    assert response.json()["error"] == "InvalidContentLength"


def test_negative_content_length_returns_400(client):
    response = client.get("/live", headers={"Content-Length": "-1"})

    assert response.status_code == 400
    assert response.json()["error"] == "InvalidContentLength"


def test_oversized_content_length_returns_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_request_bytes", 10)

    response = client.post("/v1/identify", headers={"Content-Length": "11"}, content=b"{}")

    assert response.status_code == 413
    assert response.json()["error"] == "RequestTooLarge"


def test_cors_preflight_allows_delete(client):
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "DELETE",
        "Access-Control-Request-Headers": "Authorization, X-Astra-User-Id",
    }
    response = client.options("/v1/enroll", headers=headers)
    assert response.status_code == 200
    allow_methods = response.headers.get("access-control-allow-methods", "")
    assert "DELETE" in allow_methods

