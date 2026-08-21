import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from tests.helpers import TEST_JWT_SECRET, make_token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "skip_model_load", True)
    monkeypatch.setattr(settings, "jwt_secret", TEST_JWT_SECRET)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {make_token()}"}
