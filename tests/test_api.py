"""
API tests — fully offline.

The generation LLM is replaced with a fake, the reranker is disabled (no
model download), and an index built with fake embeddings is injected into
the registry. This exercises /health, /chat, and /chat/stream without any
API key or quota.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake import FakeStreamingListLLM

import rag.chain as chain_mod
from api.main import _INDEXES, app
from rag.config import settings

ANSWER = "The Skylark X2 has a flight time of 47 minutes."


@pytest.fixture
def client(index, monkeypatch):
    monkeypatch.setattr(settings, "use_reranker", False)  # skip model download
    monkeypatch.setattr(chain_mod, "make_llm", lambda *a, **k: FakeStreamingListLLM(responses=[ANSWER]))
    _INDEXES["test"] = index
    yield TestClient(app)
    _INDEXES.pop("test", None)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_model"]


def test_chat_returns_answer_and_sources(client):
    r = client.post("/chat", json={"index_id": "test", "question": "flight time of X2?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == ANSWER
    assert len(body["sources"]) >= 1
    assert "source" in body["sources"][0]


def test_chat_unknown_index_404(client):
    r = client.post("/chat", json={"index_id": "nope", "question": "hi"})
    assert r.status_code == 404


def test_chat_stream_emits_tokens(client):
    r = client.post("/chat/stream", json={"index_id": "test", "question": "flight time?"})
    assert r.status_code == 200
    assert "event: sources" in r.text
    assert "event: token" in r.text
    assert "event: done" in r.text
