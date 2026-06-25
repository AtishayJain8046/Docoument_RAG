"""
ingestor.py
-----------
Loads PDFs, splits them into chunks, and embeds them into a vector store.

RAG pipeline step: PDF → Text → Chunks → Embeddings → Vector store

The ingestion is split into small, testable functions and tags every chunk
with rich metadata (source file, page, chunk index) so the UI can render
precise, verifiable citations.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Single place that constructs the embedding model from config."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key or None,
    )


def load_pdf(file_path: str, source_name: str) -> List[Document]:
    """Load one PDF into page-level Documents, tagged with the source name."""
    docs = PyPDFLoader(file_path).load()
    for doc in docs:
        doc.metadata["source"] = source_name
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split page Documents into overlapping, semantically coherent chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Tag each chunk with a stable index for citation references.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def ingest_pdfs(uploaded_files) -> FAISS:
    """
    Takes Streamlit UploadedFile objects → temp files → loaded pages →
    chunks → FAISS index.

    Returns a FAISS vectorstore populated with embedded chunks.
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            all_docs.extend(load_pdf(tmp_path, uploaded_file.name))
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise ValueError("No content could be extracted from the uploaded PDFs.")

    chunks = split_documents(all_docs)
    if not chunks:
        raise ValueError("Text splitting produced no chunks. Check your PDF content.")

    return FAISS.from_documents(chunks, get_embeddings())


def save_vectorstore(vectorstore: FAISS, path: str = "faiss_index") -> None:
    """Persist a FAISS index to disk so it survives app restarts."""
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    """Load a previously saved FAISS index from disk."""
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )
