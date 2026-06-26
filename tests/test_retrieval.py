"""
Retrieval tests — run offline with fake embeddings + embedded Qdrant.

Dense scores are meaningless under fake embeddings, but BM25 (the sparse
half of the hybrid ensemble) is lexical, so exact-term queries must still
surface the right chunk. This verifies the hybrid wiring end to end.
"""

from rag.retriever import get_hybrid_retriever


def test_hybrid_retrieves_exact_match(index):
    retriever = get_hybrid_retriever(index.vectorstore, index.chunks, k=3)
    docs = retriever.invoke("INV-2025-0042")
    assert any("INV-2025-0042" in d.page_content for d in docs)


def test_hybrid_disambiguates_model_by_exact_token(index):
    retriever = get_hybrid_retriever(index.vectorstore, index.chunks, k=3)
    docs = retriever.invoke("Skylark X3 flight time")
    # The X3 chunk should be retrieved (BM25 matches the exact token "X3").
    assert any("X3" in d.page_content for d in docs)
