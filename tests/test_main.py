import asyncio
import time

from src.api.routes import health
from src.config import settings


async def _ok_response(request):
    return None


async def _qdrant_ready() -> bool:
    return True


async def _qdrant_not_ready() -> bool:
    return False


async def _qdrant_slow() -> bool:
    await asyncio.sleep(1.0)
    return True


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


def _spy_qdrant_mutations(monkeypatch, service):
    """Replace every setup/mutating Qdrant surface with a failing recorder."""
    calls = []

    def forbidden(name):
        async def _forbidden(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"/ready must not call Qdrant {name}")

        return _forbidden

    for name in ("get_client", "_ensure_collection", "enroll_user_embeddings", "delete_user_embeddings"):
        monkeypatch.setattr(service, name, forbidden(name))
    return calls


def test_ready_and_health_never_mutate_qdrant(client, monkeypatch):
    calls = _spy_qdrant_mutations(monkeypatch, health.qdrant_service)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _qdrant_ready)
    monkeypatch.setattr(health.inference_engine, "is_loaded", lambda: True)
    monkeypatch.setattr(health.inference_engine, "is_gpu_enabled", lambda: False)
    monkeypatch.setattr(health, "is_face_detector_ready", lambda: True)

    for path in ("/ready", "/health"):
        response = client.get(path)

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["qdrant_connected"] is True

    assert calls == []


def test_ready_missing_collection_reports_unhealthy_without_setup(client, monkeypatch):
    """A missing collection must be reported not-ready, never auto-created."""
    calls = _spy_qdrant_mutations(monkeypatch, health.qdrant_service)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _qdrant_not_ready)

    response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["qdrant_connected"] is False
    assert calls == []


def test_live_is_process_only_and_never_touches_qdrant(client, monkeypatch):
    qdrant_calls = []

    async def forbidden(*args, **kwargs):
        qdrant_calls.append("is_connected")
        raise AssertionError("/live must not call Qdrant")

    monkeypatch.setattr(health.qdrant_service, "is_connected", forbidden)

    start = time.monotonic()
    response = client.get("/live")
    elapsed = time.monotonic() - start

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert qdrant_calls == []
    assert elapsed < 1.0


def test_ready_dependency_timeout_is_bounded(client, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_timeout_seconds", 0.05)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _qdrant_slow)

    start = time.monotonic()
    response = client.get("/ready")
    elapsed = time.monotonic() - start

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["qdrant_connected"] is False
    assert elapsed < 1.0

