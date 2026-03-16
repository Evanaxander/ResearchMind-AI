import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
import io


@pytest.fixture
async def client():
    """Async test client — no real server needed."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Health ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data


# ── Upload ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_txt_file(client, tmp_path):
    content = b"This is a test research document about transformer architectures."
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("test_paper.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["document"]["filename"] == "test_paper.txt"
    assert "doc_id" in data["document"]


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_type(client):
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("virus.exe", io.BytesIO(b"bad"), "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(client, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_SIZE_MB", 0)  # force 0 MB limit
    content = b"A" * 1024  # 1 KB > 0 MB limit
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("big.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 413


# ── Query ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_returns_stub_response(client):
    response = await client.post(
        "/api/v1/query",
        json={"question": "What is the main thesis of the paper?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the main thesis of the paper?"
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_query_validates_empty_question(client):
    response = await client.post(
        "/api/v1/query",
        json={"question": "  "},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_documents(client):
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)