from types import SimpleNamespace

import numpy as np
import pytest

from src.services import qdrant_client
from src.services.qdrant_client import QdrantOperationError, QdrantService, QdrantUnavailableError


class FakeQdrantClient:
    def __init__(self, *, collections=None, fail_collections=False, **kwargs):
        self.kwargs = kwargs
        self.collections = collections or []
        self.fail_collections = fail_collections
        self.created_collection = False
        self.created_payload_index = False
        self.upserted_points = []
        self.deleted = False

    async def get_collections(self):
        if self.fail_collections:
            raise RuntimeError("qdrant down")
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    async def create_collection(self, **kwargs):
        self.created_collection = True

    async def create_payload_index(self, **kwargs):
        self.created_payload_index = True

    async def upsert(self, **kwargs):
        self.upserted_points = kwargs["points"]

    async def scroll(self, **kwargs):
        return [
            SimpleNamespace(
                id="p1",
                vector=[1.0, 0.0],
                payload={"user_id": "u1", "embedding_index": 1},
            )
        ], None

    async def count(self, **kwargs):
        return SimpleNamespace(count=2)

    async def delete(self, **kwargs):
        self.deleted = True


@pytest.mark.anyio
async def test_qdrant_get_client_creates_missing_collection(monkeypatch):
    created = {}

    def fake_client(**kwargs):
        created["client"] = FakeQdrantClient(**kwargs)
        return created["client"]

    monkeypatch.setattr(qdrant_client.settings, "qdrant_host", "localhost")
    monkeypatch.setattr(qdrant_client.settings, "qdrant_collection_name", "faces")
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", fake_client)

    service = QdrantService()
    client = await service.get_client()

    assert client.created_collection is True
    assert client.created_payload_index is True
    assert created["client"].kwargs["host"] == "localhost"


@pytest.mark.anyio
async def test_qdrant_get_client_wraps_initialization_failure(monkeypatch):
    monkeypatch.setattr(
        qdrant_client,
        "AsyncQdrantClient",
        lambda **kwargs: FakeQdrantClient(collections=["faces"], fail_collections=True, **kwargs),
    )

    service = QdrantService()

    with pytest.raises(QdrantUnavailableError, match="initialization failed"):
        await service.get_client()


@pytest.mark.anyio
async def test_qdrant_enroll_replaces_existing_embeddings(monkeypatch):
    client = FakeQdrantClient(collections=["face_embeddings"])
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda **kwargs: client)

    service = QdrantService()
    result = await service.enroll_user_embeddings("u1", [np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    assert result["success"] is True
    assert result["inserted_count"] == 2
    assert client.deleted is True
    assert len(client.upserted_points) == 2


@pytest.mark.anyio
async def test_qdrant_retrieve_and_verify_embeddings(monkeypatch):
    client = FakeQdrantClient(collections=["face_embeddings"])
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda **kwargs: client)

    service = QdrantService()
    embeddings = await service.retrieve_user_embeddings("u1")
    verification = await service.verify_face_1to1("u1", np.array([1.0, 0.0]), threshold=0.6)

    assert embeddings[0]["id"] == "p1"
    assert verification["verified"] is True
    assert verification["confidence"] == 1.0


@pytest.mark.anyio
async def test_qdrant_operation_errors_are_wrapped(monkeypatch):
    service = QdrantService()

    async def failing_get_client():
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "get_client", failing_get_client)

    with pytest.raises(QdrantOperationError, match="Failed to count embeddings"):
        await service.get_user_embedding_count("u1")
