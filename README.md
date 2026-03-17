# 📄 PDF ChatBot — RAG System with LangChain, FAISS & Streamlit

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDFs and ask questions about them in natural language. Built with LangChain, FAISS, OpenAI, and Streamlit.

> ✅ Great for resumes: demonstrates LLMs, vector databases, embeddings, and full-stack Python.

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique where an LLM answers questions using **retrieved context** from your own documents — not just its training data.

```
PDF → Extract Text → Split into Chunks → Embed Chunks → Store in FAISS
                                                              ↓
User Question → Embed Question → Similarity Search → Top-K Chunks → LLM → Answer
```

---

## 🏗 Architecture

```
pdf_chatbot/
├── app.py              # Streamlit UI + session management
├── rag/
│   ├── ingestor.py     # PDF loading, text splitting, FAISS indexing
│   ├── retriever.py    # FAISS similarity search wrapper
│   └── chain.py        # Conversational RAG chain (LangChain)
├── requirements.txt
├── .env.example
└── README.md
```

### Component Breakdown

| File | Responsibility | Key Concepts |
|---|---|---|
| `ingestor.py` | Load PDFs → split → embed → FAISS | `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `OpenAIEmbeddings` |
| `retriever.py` | Query → top-K similar chunks | `FAISS.as_retriever`, cosine similarity |
| `chain.py` | Conversational Q&A with history | `create_retrieval_chain`, `history_aware_retriever` |
| `app.py` | UI, file upload, session state | Streamlit, session management |

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/pdf-chatbot-rag.git
cd pdf-chatbot-rag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your API_KEY
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| ` ANY API KEY ` | ✅ Yes | api-keys |
| `LANGCHAIN_TRACING_V2` | ❌ No | Set `true` to enable LangSmith debugging |
| `LANGCHAIN_API_KEY` | ❌ No | LangSmith key (free) for tracing |

---

## 🛠 How to Use

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files (research papers, books, contracts, etc.)
2. **Ingest** — Click "⚡ Ingest" to extract text, split into chunks, and build the FAISS index
3. **Chat** — Ask questions in natural language. The system retrieves relevant chunks and generates answers
4. **View Sources** — Expand the "Sources used" section under each answer to see which document chunks were retrieved

---

## 🧩 Key Technical Decisions

### Why FAISS?
- **Local** — no external database needed, runs entirely on your machine
- **Fast** — Facebook's optimized similarity search library
- **Persistent** — index can be saved/loaded from disk

### Why `RecursiveCharacterTextSplitter`?
Splits by paragraph → sentence → word in order, keeping semantically coherent chunks. The `chunk_overlap` ensures context isn't lost at boundaries.

### Why `history_aware_retriever`?
Without it, follow-up questions like "What did it say about that?" lose context. This component reformulates the question using chat history before retrieval.


## 📈 Resume Talking Points

- Built an end-to-end RAG pipeline using **LangChain** and **FAISS** vector database
- Implemented **document chunking** with configurable overlap to preserve context at chunk boundaries
- Used **OpenAI embeddings** (`text-embedding-3-small`) to represent document semantics in vector space
- Built **conversational memory** with `create_history_aware_retriever` for multi-turn Q&A
- Deployed an interactive UI with **Streamlit** supporting multi-file upload and real-time chat

---

## 🔧 Extending the Project

Ideas to make it even more impressive:

- **Persistent index**: call `vectorstore.save_local("faiss_index")` to persist across restarts
- **Multiple namespaces**: separate FAISS indices per user/document set
- **Reranking**: add a cross-encoder reranker after retrieval for better accuracy
- **Evaluation**: use RAGAS to benchmark answer quality
- **Deploy**: host on Streamlit Cloud (free) or Hugging Face Spaces

---

## 📦 Dependencies

- `streamlit` — UI framework
- `langchain` + `langchain_google_genai` — LLM orchestration
- `faiss-cpu` — vector similarity search
- `pypdf` — PDF text extraction
- `python-dotenv` — environment variable management
- `tiktoken` — token counting for chunking

---

## License

MIT
