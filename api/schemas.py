"""
schemas.py
----------
Pydantic request/response models for the API. Typed contracts give us
automatic validation and OpenAPI docs at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    index_id: str = Field(..., description="Handle to use in subsequent /chat calls")
    files: list[str]
    num_chunks: int


class Turn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    index_id: str
    question: str
    chat_history: list[Turn] = Field(default_factory=list)
    top_k: int | None = Field(None, description="Override number of chunks retrieved")


class Source(BaseModel):
    source: str
    page: int | str | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
    indexes_loaded: int
