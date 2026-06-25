# 🗺 Roadmap — PDF RAG → Production-Grade Product

Tracking the evolution from a basic RAG demo into a measurable, deployable, agentic RAG product.

**Architecture decisions (locked):**
- Vector store: **Qdrant** (Docker, persistent, metadata filtering)
- Reranker: **Local cross-encoder** (BAAI `bge-reranker`, no API cost)
- Deploy target: **Hugging Face Spaces** (Docker)
- Multimodal: **designed for now, built last** (capstone with agentic retrieval)
- LLM: `gemini-2.5-flash` · Embeddings: `models/gemini-embedding-001`

---

## Phase 0 — Repo hygiene (credibility) ✅ DONE
- [x] Verify real Gemini model ids against live API
- [x] Accurate `requirements.txt` (real stack, no phantom OpenAI deps)
- [x] Fix `.env.example` (Gemini, not OpenAI comments)
- [x] Remove dead code in ingestor
- [x] Central config via `pydantic-settings`
- [ ] README rewrite (done at the end, once features land)

## Phase 1 — Core refactor ✅ DONE
- [x] `rag/config.py` — typed settings
- [x] Ingestion: richer metadata (chunk index), split into testable functions
- [x] Chain reads model ids from config

## Phase 2 — Qdrant vector store
- [ ] Qdrant client + collection-per-document-set
- [ ] Persistence (survives restarts), metadata filtering
- [ ] `docker-compose` with Qdrant service

## Phase 3 — Retrieval quality
- [ ] Hybrid search: BM25 (keyword) + dense (Ensemble)
- [ ] BGE cross-encoder reranker after retrieval
- [ ] Inline citations mapped to claims ([1][2])

## Phase 4 — Evaluation harness ⭐ key differentiator
- [ ] Eval dataset (Q/A over sample docs)
- [ ] RAGAS metrics: faithfulness, answer relevance, context precision/recall
- [ ] Before/after retrieval-quality table in README

## Phase 5 — API layer
- [ ] FastAPI service: `/ingest`, `/chat`, `/health`
- [ ] SSE token streaming
- [ ] Pydantic request/response models

## Phase 6 — UI + observability
- [ ] Streamlit talks to the API (thin client)
- [ ] LangSmith / Langfuse tracing (latency, cost, retrieval traces)

## Phase 7 — Quality engineering
- [ ] pytest: chunking, retrieval, mocked-LLM integration
- [ ] GitHub Actions CI: ruff + tests
- [ ] Live deploy to Hugging Face Spaces

## Phase 8 — Capstone (the "wow")
- [ ] Agentic retrieval: query decomposition, multi-hop, retrieval routing
- [ ] Multimodal: page-image rendering + vision path for figures/tables
- [ ] "No answer in docs → web search" fallback tool

---

### Status log
- 2026-06-25: Project audit complete; decisions locked; Phase 0 started.
