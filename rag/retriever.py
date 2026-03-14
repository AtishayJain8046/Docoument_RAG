"""
retriever.py
------------
Wraps the FAISS vectorstore in a LangChain retriever.

RAG pipeline step: Query → Embedding → FAISS similarity search → Top-K chunks
"""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(vectorstore: FAISS, k: int = 4) -> VectorStoreRetriever:
    """
    Returns a retriever that fetches the top-k most similar chunks
    for a given query using cosine similarity in FAISS.

    Args:
        vectorstore: The populated FAISS vectorstore
        k:           Number of chunks to return per query

    Returns:
        A LangChain VectorStoreRetriever
    """
    return vectorstore.as_retriever(
        search_type="similarity",   # cosine similarity (default)
        search_kwargs={"k": k},
    )
