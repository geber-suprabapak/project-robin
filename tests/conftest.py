import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from tests.helpers import TEST_USER_ID


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "skip_model_load", True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {
        "Authorization": "Bearer dev-robin-service-token",
        "X-Astra-User-Id": TEST_USER_ID,
    }
