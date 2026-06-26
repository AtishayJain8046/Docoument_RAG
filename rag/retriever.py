"""
retriever.py
------------
Builds the retrieval stack:

    query
      │
      ├── dense (Qdrant vector similarity)  ┐
      │                                     ├─ EnsembleRetriever (hybrid)
      └── sparse (BM25 keyword)             ┘
                          │
                          └── cross-encoder reranker (BGE)  ──▶ top-N chunks

Hybrid search catches both semantic matches (dense) and exact terms /
acronyms / IDs (BM25). The reranker then re-scores the merged candidates
with a cross-encoder for sharper ordering than either retriever alone.
"""

from __future__ import annotations

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.config import settings


def get_dense_retriever(vectorstore, k: int | None = None) -> BaseRetriever:
    """Vector-similarity retriever over the Qdrant collection."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.retrieval_fetch_k},
    )


def get_hybrid_retriever(
    vectorstore, chunks: list[Document], k: int | None = None
) -> BaseRetriever:
    """Ensemble of dense (Qdrant) + sparse (BM25) retrievers."""
    fetch_k = k or settings.retrieval_fetch_k
    dense = get_dense_retriever(vectorstore, fetch_k)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = fetch_k

    dense_w = settings.hybrid_dense_weight
    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[dense_w, 1.0 - dense_w],
    )


def _wrap_with_reranker(base: BaseRetriever) -> BaseRetriever:
    """Re-score base-retriever results with a local BGE cross-encoder.

    Imported lazily so the module loads without torch/sentence-transformers
    when reranking is disabled.
    """
    from langchain_classic.retrievers.document_compressors import (
        CrossEncoderReranker,
    )
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    compressor = CrossEncoderReranker(model=encoder, top_n=settings.rerank_top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )


def build_retriever(
    vectorstore, chunks: list[Document], k: int | None = None
) -> BaseRetriever:
    """Full retrieval stack: hybrid search, optionally reranked.

    `k` overrides the final number of chunks returned (top_n when reranking,
    else the ensemble fetch size).
    """
    if settings.use_reranker:
        base = get_hybrid_retriever(vectorstore, chunks)
        retriever = _wrap_with_reranker(base)
        if k:
            retriever.base_compressor.top_n = k
        return retriever

    return get_hybrid_retriever(vectorstore, chunks, k or settings.retrieval_top_k)
