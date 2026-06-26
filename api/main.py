"""
main.py
-------
FastAPI service exposing the RAG pipeline:

    POST /ingest       upload PDFs → build an index → returns index_id
    POST /chat         ask a question against an index (JSON answer + sources)
    POST /chat/stream  same, but streams answer tokens over SSE
    GET  /health       liveness + active model/index info

Indexes are held in an in-process registry (fine for a single-instance demo;
swap for a shared Qdrant server + reconnect-by-collection to scale out).
Interactive docs at /docs.
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    Source,
)
from rag.chain import stream_answer
from rag.config import settings
from rag.pipeline import RagIndex, ingest_files, make_chain, make_retriever

app = FastAPI(title="PDF RAG API", version="1.0.0")

# index_id -> RagIndex. In-process; see module docstring for scaling note.
_INDEXES: dict[str, RagIndex] = {}


def _get_index(index_id: str) -> RagIndex:
    index = _INDEXES.get(index_id)
    if index is None:
        raise HTTPException(status_code=404, detail=f"Unknown index_id: {index_id}")
    return index


def _to_sources(docs) -> list[Source]:
    return [
        Source(
            source=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page"),
            snippet=d.page_content[:220].replace("\n", " "),
        )
        for d in docs
    ]


def _history(req: ChatRequest) -> list[tuple[str, str]]:
    return [(t.role, t.content) for t in req.chat_history]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        indexes_loaded=len(_INDEXES),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    pdfs = [(f.filename or "upload.pdf", await f.read()) for f in files]
    if not pdfs:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    try:
        index = ingest_files(pdfs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    index_id = uuid.uuid4().hex[:12]
    _INDEXES[index_id] = index
    return IngestResponse(
        index_id=index_id,
        files=[name for name, _ in pdfs],
        num_chunks=len(index.chunks),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    index = _get_index(req.index_id)
    chain = make_chain(index, k=req.top_k)
    result = chain.invoke({"input": req.question, "chat_history": _history(req)})
    return ChatResponse(
        answer=result["answer"], sources=_to_sources(result.get("context", []))
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> EventSourceResponse:
    index = _get_index(req.index_id)
    retriever = make_retriever(index, k=req.top_k)
    docs, tokens = stream_answer(
        retriever, {"input": req.question, "chat_history": _history(req)}
    )

    def events():
        # First frame: the sources, so the client can render citations early.
        yield {"event": "sources",
               "data": json.dumps([s.model_dump() for s in _to_sources(docs)])}
        for token in tokens:
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(events())
