"""
pipeline.py
-----------
Top-level orchestration tying the RAG stages together so the UI and the
(future) API share one entry point:

    PDFs → chunks → Qdrant index → hybrid+rerank retriever → RAG chain

Keeps the chunk list alongside the vector store because BM25 (the sparse
half of hybrid search) indexes the raw chunks in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from rag.chain import build_chain
from rag.ingestor import files_to_chunks, pdfs_to_chunks
from rag.retriever import build_retriever
from rag.vectorstore import build_vectorstore


@dataclass
class RagIndex:
    """An ingested document set: the vector store plus its raw chunks."""
    vectorstore: object
    chunks: List[Document]


def _index_from_chunks(chunks: List[Document]) -> RagIndex:
    return RagIndex(vectorstore=build_vectorstore(chunks), chunks=chunks)


def ingest(uploaded_files) -> RagIndex:
    """Ingest Streamlit-uploaded PDFs into a queryable index."""
    return _index_from_chunks(pdfs_to_chunks(uploaded_files))


def ingest_files(files: list[tuple[str, bytes]]) -> RagIndex:
    """Ingest (filename, bytes) pairs into a queryable index (API entry point)."""
    return _index_from_chunks(files_to_chunks(files))


def make_retriever(index: RagIndex, k: int | None = None):
    """Build the hybrid+rerank retriever over an ingested index."""
    return build_retriever(index.vectorstore, index.chunks, k=k)


def make_chain(index: RagIndex, k: int | None = None):
    """Build the conversational RAG chain over an ingested index."""
    return build_chain(make_retriever(index, k))
