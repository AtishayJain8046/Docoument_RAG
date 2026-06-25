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
from rag.ingestor import pdfs_to_chunks
from rag.retriever import build_retriever
from rag.vectorstore import build_vectorstore


@dataclass
class RagIndex:
    """An ingested document set: the vector store plus its raw chunks."""
    vectorstore: object
    chunks: List[Document]


def ingest(uploaded_files) -> RagIndex:
    """Ingest uploaded PDFs into a queryable index."""
    chunks = pdfs_to_chunks(uploaded_files)
    vectorstore = build_vectorstore(chunks)
    return RagIndex(vectorstore=vectorstore, chunks=chunks)


def make_chain(index: RagIndex, k: int | None = None):
    """Build the conversational RAG chain over an ingested index."""
    retriever = build_retriever(index.vectorstore, index.chunks, k=k)
    return build_chain(retriever)
