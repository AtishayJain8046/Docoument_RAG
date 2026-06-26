"""Unit tests for chunking — pure, no external services."""

from langchain_core.documents import Document

from rag.config import settings
from rag.ingestor import split_documents


def test_split_respects_chunk_size():
    long_text = "sentence. " * 500  # ~5000 chars
    chunks = split_documents([Document(page_content=long_text, metadata={"source": "a.pdf"})])
    assert len(chunks) > 1
    # allow a little slack for separator boundaries
    assert all(len(c.page_content) <= settings.chunk_size + 50 for c in chunks)


def test_chunks_get_sequential_index_and_keep_metadata():
    long_text = "paragraph.\n\n" * 300
    chunks = split_documents([Document(page_content=long_text, metadata={"source": "a.pdf"})])
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)


def test_short_doc_stays_single_chunk():
    chunks = split_documents([Document(page_content="tiny", metadata={"source": "a.pdf"})])
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
