# 📄 Document RAG — a measurable, production-shaped RAG system

Ask questions about your PDFs in natural language. Built as a **proper RAG
product**, not a notebook demo: a typed core, a Qdrant vector store, hybrid
retrieval with reranking, a **FastAPI** service with streaming, a Streamlit
UI, an **evaluation harness with real numbers**, tests, and CI.

> Stack: **Gemini** (LLM + embeddings) · **Qdrant** · **LangChain** ·
> **FastAPI** · **Streamlit** · **RAGAS** · **pytest/ruff/GitHub Actions**

---

## Why this isn't just another PDF chatbot

| Most demos | This project |
|---|---|
| One `app.py`, everything inline | Typed core (`rag/`), API (`api/`), eval (`eval/`), tests (`tests/`) |
| Naive top-k vector search | **Hybrid** (BM25 + dense) + **cross-encoder reranking** |
| In-memory FAISS, lost on restart | **Qdrant** (embedded for dev, server for prod) |
| "It works on my machine" | **Eval harness** that measures retrieval quality with numbers |
| Streamlit only | **FastAPI** service (`/ingest`, `/chat`, SSE streaming) + UI |
| No tests | Offline **pytest** suite + **CI** (ruff + tests) |

---

## Architecture

```
                     ┌──────────────┐      ┌──────────────┐
   PDFs ──▶ ingest ─▶│  chunk +     │─────▶│   Qdrant     │
                     │  embed       │      │ (vectors)    │
                     └──────────────┘      └──────┬───────┘
                                                  │
   question ──▶ ┌─────────────────────────────────▼─────────────┐
                │  Hybrid retrieval: BM25 (keyword) + dense       │
                │  → cross-encoder rerank → top-k chunks          │
                └─────────────────────────┬───────────────────────┘
                                          ▼
                          Gemini LLM (grounded answer + sources)

   Surfaces:  FastAPI  /ingest · /chat · /chat/stream (SSE) · /health
              Streamlit chat UI
```

| Module | Responsibility |
|---|---|
| [`rag/config.py`](rag/config.py) | Typed settings (pydantic-settings) — models, chunking, retrieval, Qdrant |
| [`rag/ingestor.py`](rag/ingestor.py) | PDF → pages → chunks (+ citation metadata) |
| [`rag/vectorstore.py`](rag/vectorstore.py) | Qdrant — embedded (no Docker) or server mode |
| [`rag/retriever.py`](rag/retriever.py) | Hybrid (BM25 + dense) ensemble → cross-encoder reranker |
| [`rag/chain.py`](rag/chain.py) | Conversational RAG chain + streaming |
| [`rag/pipeline.py`](rag/pipeline.py) | One entry point: ingest → index → retriever → chain |
| [`api/main.py`](api/main.py) | FastAPI service with SSE streaming |
| [`eval/`](eval/) | RAGAS + quota-free retrieval evaluation |

---

## 📊 Evaluation — with real numbers

Retrieval quality is measured, not assumed. The
[quota-free retrieval eval](eval/retrieval_eval.py) scores dense vs
hybrid+rerank against a gold set built with confusable near-duplicates
(a 40-SKU catalog) — see [`eval/retrieval_results.md`](eval/retrieval_results.md):

| Embedder | Dense (MRR) | Hybrid + rerank (MRR) |
|---|:--:|:--:|
| Gemini `gemini-embedding-001` (3072-d) | **1.000** | 0.962 |
| Local `all-MiniLM-L6-v2` (384-d) | 0.923 | **0.962** |

**Honest finding:** a strong embedder already ranks answers first, so extra
machinery adds little — but with a cheap/weak embedder, dense degrades and
**hybrid + reranking recovers the lost ranking quality**. The eval lets you
pick the retrieval stack to match the embedder instead of cargo-culting
complexity. (A RAGAS *generation* eval — faithfulness/relevancy/grounding —
is wired up in [`eval/run_eval.py`](eval/run_eval.py); it needs LLM quota
beyond the Gemini free tier to run.)

```bash
python -m eval.retrieval_eval     # regenerate the table above
```

---

## 🚀 Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add your GOOGLE_API_KEY
```

Get a free Gemini key at <https://aistudio.google.com/app/apikey>.

**Streamlit UI** (embedded Qdrant, no Docker needed):

```bash
streamlit run app.py
```

**API** (interactive docs at <http://localhost:8000/docs>):

```bash
uvicorn api.main:app --reload
```

```bash
# ingest a PDF, then chat against it
curl -F "files=@paper.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"index_id":"<id from ingest>","question":"What is this about?"}'
```

**Qdrant server mode** (instead of embedded):

```bash
docker compose up -d              # starts Qdrant on :6333
# then set QDRANT_EMBEDDED=false in .env
```

---

## 🔧 Configuration

All settings live in [`rag/config.py`](rag/config.py) and are env-overridable
(see [`.env.example`](.env.example)). Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini key (required) |
| `LLM_MODEL` | `gemini-2.5-flash` | Answer model |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embeddings |
| `QDRANT_EMBEDDED` | `true` | In-process Qdrant (set `false` for server) |
| `USE_RERANKER` | `true` | Cross-encoder reranking |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Reranker (BGE optional) |
| `RETRIEVAL_TOP_K` | `4` | Chunks fed to the LLM |

---

## 🧪 Development

```bash
ruff check .          # lint
python -m pytest -q   # offline tests (no API key needed)
```

CI runs both on every push/PR ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## 🗺 Roadmap

See [ROADMAP.md](ROADMAP.md). Shipped: typed core, Qdrant, hybrid+rerank,
eval harness, FastAPI + streaming, tests + CI. Next: observability tracing,
Hugging Face Spaces deploy, and an agentic + multimodal capstone.

## License

MIT
