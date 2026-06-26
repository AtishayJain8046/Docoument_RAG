"""
Shared pytest fixtures.

All tests run fully offline — no Gemini API key required — by using
deterministic fake embeddings and a fake LLM, and by running Qdrant in
embedded mode. This keeps CI fast, free, and quota-independent.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.pipeline import RagIndex
from rag.vectorstore import build_vectorstore


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def chunks() -> list[Document]:
    return [
        Document(page_content="The Skylark X2 has a flight time of 47 minutes.",
                 metadata={"source": "spec.pdf", "page": 1, "chunk_index": 0}),
        Document(page_content="The Skylark X3 has a flight time of 53 minutes.",
                 metadata={"source": "spec.pdf", "page": 2, "chunk_index": 1}),
        Document(page_content="Invoice INV-2025-0042 totals 1,250 euros.",
                 metadata={"source": "fin.pdf", "page": 1, "chunk_index": 2}),
    ]


@pytest.fixture
def index(chunks, fake_embeddings) -> RagIndex:
    vs = build_vectorstore(chunks, embedding=fake_embeddings)
    return RagIndex(vectorstore=vs, chunks=chunks)
