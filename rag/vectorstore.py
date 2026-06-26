"""
vectorstore.py
--------------
Builds and connects to the Qdrant vector store.

Two modes (controlled by config.qdrant_embedded):
  - embedded: an in-process Qdrant (no Docker) — used for tests/offline dev
  - server:   a real Qdrant instance at QDRANT_URL — used in docker-compose
              and in production

The collection schema (vector size + distance) is inferred automatically by
QdrantVectorStore from the embedding model, so swapping embedding models
needs no manual schema changes.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import settings
from rag.ingestor import get_embeddings


def _connection_kwargs() -> dict:
    """Connection kwargs selecting embedded (in-process) vs server mode."""
    if settings.qdrant_embedded:
        # In-process store; no Docker needed.
        return {"location": ":memory:"}
    return {
        "url": settings.qdrant_url,
        "api_key": settings.qdrant_api_key or None,
    }


def build_vectorstore(chunks: List[Document], embedding=None) -> QdrantVectorStore:
    """Embed `chunks` and index them into a fresh Qdrant collection.

    `embedding` defaults to the configured Gemini model; pass a different
    embedder (e.g. a local model) to compare retrieval quality across
    embedders in the eval harness.
    """
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding or get_embeddings(),
        collection_name=settings.qdrant_collection,
        force_recreate=True,
        **_connection_kwargs(),
    )


def connect_vectorstore() -> QdrantVectorStore:
    """Connect to an existing collection (server mode) without re-indexing."""
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )
