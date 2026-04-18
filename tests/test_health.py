from src.api.routes import health


async def _ready() -> bool:
    return True


async def _not_ready() -> bool:
    return False


def test_liveness_does_not_require_dependencies(client):
    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_returns_503_when_dependency_is_down(client, monkeypatch):
    monkeypatch.setattr(health.inference_engine, "is_loaded", lambda: True)
    monkeypatch.setattr(health.inference_engine, "is_gpu_enabled", lambda: False)
    monkeypatch.setattr(health, "is_face_detector_ready", lambda: True)
    monkeypatch.setattr(health.supabase_service, "is_ready", _ready)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _not_ready)

    response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["model_loaded"] is True
    assert payload["qdrant_connected"] is False


def test_health_uses_readiness_semantics(client, monkeypatch):
    monkeypatch.setattr(health.inference_engine, "is_loaded", lambda: False)
    monkeypatch.setattr(health.inference_engine, "is_gpu_enabled", lambda: False)
    monkeypatch.setattr(health, "is_face_detector_ready", lambda: True)
    monkeypatch.setattr(health.supabase_service, "is_ready", _ready)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _ready)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_readiness_returns_200_when_all_dependencies_are_ready(client, monkeypatch):
    monkeypatch.setattr(health.inference_engine, "is_loaded", lambda: True)
    monkeypatch.setattr(health.inference_engine, "is_gpu_enabled", lambda: False)
    monkeypatch.setattr(health, "is_face_detector_ready", lambda: True)
    monkeypatch.setattr(health.supabase_service, "is_ready", _ready)
    monkeypatch.setattr(health.qdrant_service, "is_connected", _ready)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["face_detector_ready"] is True
    assert payload["supabase_connected"] is True
    assert payload["qdrant_connected"] is True
