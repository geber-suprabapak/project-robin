import pytest

from src.services import supabase_client
from src.services.supabase_client import SupabaseUnavailableError


def test_supabase_client_uses_configured_http_timeouts(monkeypatch):
    captured = {}

    def fake_create_client(supabase_url, supabase_key, options):
        captured["url"] = supabase_url
        captured["key"] = supabase_key
        captured["options"] = options
        return object()

    monkeypatch.setattr(supabase_client.settings, "supabase_url", "https://project.supabase.co")
    monkeypatch.setattr(supabase_client.settings, "supabase_key", "anon-key")
    monkeypatch.setattr(supabase_client.settings, "supabase_service_role_key", "service-key")
    monkeypatch.setattr(supabase_client.settings, "supabase_timeout_seconds", 7)
    monkeypatch.setattr(supabase_client, "create_client", fake_create_client)

    service = supabase_client.SupabaseService()

    assert service.is_connected() is True
    assert captured["key"] == "service-key"
    assert captured["options"].postgrest_client_timeout == 7
    assert captured["options"].storage_client_timeout == 7
    assert captured["options"].function_client_timeout == 7


@pytest.mark.anyio
async def test_supabase_execute_retries_then_succeeds(monkeypatch):
    calls = 0

    async def fake_run_in_threadpool(func):
        return func()

    def flaky_operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return "ok"

    service = supabase_client.SupabaseService()
    service.client = object()
    monkeypatch.setattr(supabase_client.settings, "supabase_max_retries", 1)
    monkeypatch.setattr(supabase_client, "run_in_threadpool", fake_run_in_threadpool)

    assert await service._execute_with_retry("test operation", flaky_operation) == "ok"
    assert calls == 2


@pytest.mark.anyio
async def test_supabase_execute_requires_configured_client(monkeypatch):
    service = supabase_client.SupabaseService()
    service.client = None

    with pytest.raises(SupabaseUnavailableError, match="not configured"):
        await service._execute_with_retry("test operation", lambda: None)


@pytest.mark.anyio
async def test_supabase_profile_lookup_returns_first_row(monkeypatch):
    async def fake_execute(operation, func):
        return type("Response", (), {"data": [{"user_id": "u1", "nis": "123"}]})()

    service = supabase_client.SupabaseService()
    service.client = object()
    monkeypatch.setattr(service, "_execute_with_retry", fake_execute)

    assert await service.get_user_profile_by_nis("123") == {"user_id": "u1", "nis": "123"}
    assert await service.get_user_profile_by_id("u1") == {"user_id": "u1", "nis": "123"}
    assert await service.get_student_by_user_id("u1") == {"user_id": "u1", "nis": "123"}


@pytest.mark.anyio
async def test_supabase_readiness_reports_false_on_failure(monkeypatch):
    async def fake_execute(operation, func):
        raise SupabaseUnavailableError("down")

    service = supabase_client.SupabaseService()
    service.client = object()
    monkeypatch.setattr(service, "_execute_with_retry", fake_execute)

    assert await service.is_ready() is False
