"""
config.py
---------
Single source of truth for all tunable settings, loaded from environment
variables (and a local .env) via pydantic-settings.

Every other module imports `settings` from here instead of calling
os.getenv directly — so models, retrieval params, and infra endpoints are
configured in one place and are easy to override per environment.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Credentials ───────────────────────────────────────────────────
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")

    # ── Models ────────────────────────────────────────────────────────
    llm_model: str = Field("gemini-2.5-flash", alias="LLM_MODEL")
    # GA embedding model verified against the live API (preview ids can be
    # deprecated, so we default to the stable GA one).
    embedding_model: str = Field(
        "models/gemini-embedding-001", alias="EMBEDDING_MODEL"
    )
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")

    # ── Vector store (Qdrant) ─────────────────────────────────────────
    # Embedded (in-process) mode needs no Docker — used for tests/offline.
    # Set QDRANT_EMBEDDED=false to talk to a Qdrant server at QDRANT_URL.
    qdrant_embedded: bool = Field(True, alias="QDRANT_EMBEDDED")
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("pdf_chunks", alias="QDRANT_COLLECTION")

    # ── Retrieval ─────────────────────────────────────────────────────
    retrieval_top_k: int = Field(4, alias="RETRIEVAL_TOP_K")
    # candidates pulled before reranking (wider net → reranker picks best)
    retrieval_fetch_k: int = Field(20, alias="RETRIEVAL_FETCH_K")
    rerank_top_n: int = Field(4, alias="RERANK_TOP_N")
    # Lightweight (~90MB) cross-encoder — fast, CPU-friendly, clone/deploy
    # friendly. Swap to "BAAI/bge-reranker-base" (~1.1GB) for higher quality.
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    use_reranker: bool = Field(True, alias="USE_RERANKER")
    # Ensemble weight for dense (vs BM25) in hybrid search; 0..1.
    hybrid_dense_weight: float = Field(0.5, alias="HYBRID_DENSE_WEIGHT")

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key)


settings = Settings()
