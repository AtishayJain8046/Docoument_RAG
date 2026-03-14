"""
ingestor.py
-----------
Handles loading PDFs, splitting them into chunks, and storing
the chunks as vector embeddings in a local FAISS index.

RAG pipeline step: PDF → Text → Chunks → Embeddings → FAISS
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_pdfs(uploaded_files, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    Takes a list of Streamlit UploadedFile objects, saves them to temp files,
    loads them with PyPDFLoader, splits into chunks, and indexes with FAISS.

    Args:
        uploaded_files: List of st.UploadedFile objects from st.file_uploader
        chunk_size:     Number of characters per chunk (default 1000)
        chunk_overlap:  Overlap between consecutive chunks (default 200)

    Returns:
        FAISS vectorstore populated with embedded chunks
    """
    all_docs = []

    for uploaded_file in uploaded_files:
        # Save to a temp file because PyPDFLoader needs a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Tag each document with original filename (for source display)
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)  # clean up temp file

    if not all_docs:
        raise ValueError("No content could be extracted from the uploaded PDFs.")

    # ── Split into overlapping chunks ────────────────────────────────────────
    # RecursiveCharacterTextSplitter splits by paragraph → sentence → word
    # to keep chunks semantically coherent.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("Text splitting produced no chunks. Check your PDF content.")

    # ── Embed and store in FAISS ─────────────────────────────────────────────
    # OpenAIEmbeddings calls text-embedding-ada-002 (cheap and accurate)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # newest, cheaper, better than ada-002
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = "faiss_index") -> None:
    """Persist FAISS index to disk so it survives app restarts."""
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
