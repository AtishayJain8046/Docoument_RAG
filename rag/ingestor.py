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


def load_pdf(file_path: str, source_name: str) -> list[Document]:
    """Load one PDF into page-level Documents, tagged with the source name."""
    docs = PyPDFLoader(file_path).load()
    for doc in docs:
        doc.metadata["source"] = source_name
    return docs


def load_pdf_bytes(data: bytes, source_name: str) -> list[Document]:
    """Load a PDF from raw bytes (PyPDFLoader needs a real path → temp file)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return load_pdf(tmp_path, source_name)
    finally:
        os.unlink(tmp_path)


def files_to_chunks(files: list[tuple[str, bytes]]) -> list[Document]:
    """Core ingestion: (filename, bytes) pairs → page docs → chunks.

    Shared by the Streamlit UI and the API so both produce identical chunks.
    """
    all_docs: list[Document] = []
    for name, data in files:
        all_docs.extend(load_pdf_bytes(data, name))

    if not all_docs:
        raise ValueError("No content could be extracted from the uploaded PDFs.")

    chunks = split_documents(all_docs)
    if not chunks:
        raise ValueError("Text splitting produced no chunks. Check your PDF content.")
    return chunks


def split_documents(docs: list[Document]) -> list[Document]:
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


def pdfs_to_chunks(uploaded_files) -> list[Document]:
    """Streamlit entry point: UploadedFile objects → chunks."""
    return files_to_chunks([(f.name, f.read()) for f in uploaded_files])


def ingest_pdfs(uploaded_files) -> FAISS:
    """Legacy FAISS path — kept for tests/fallback. Returns a FAISS store."""
    return FAISS.from_documents(pdfs_to_chunks(uploaded_files), get_embeddings())


def save_vectorstore(vectorstore: FAISS, path: str = "faiss_index") -> None:
    """Persist a FAISS index to disk so it survives app restarts."""
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    """Load a previously saved FAISS index from disk."""
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )
