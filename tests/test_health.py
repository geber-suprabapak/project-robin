from fastapi.testclient import TestClient

from src.api.routes import health
from src.config import settings
from src.main import app


async def _qdrant_disconnected() -> bool:
    return False


def test_root_and_health_without_model(monkeypatch):
    settings.skip_model_load = True
    monkeypatch.setattr(health.qdrant_service, "is_connected", _qdrant_disconnected)

    with TestClient(app) as client:
        root_response = client.get("/")
        health_response = client.get("/health")

    assert root_response.status_code == 200
    assert root_response.json()["status"] == "running"

    assert health_response.status_code == 200
    payload = health_response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is False
    assert payload["qdrant_connected"] is False
