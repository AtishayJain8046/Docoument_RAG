import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag.ingestor import ingest_pdfs
from rag.retriever import get_retriever
from rag.chain import build_chain

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark slate background */
.stApp {
    background: #0d0f14;
    color: #e8e6df;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
}

[data-testid="stSidebar"] * {
    color: #c9c6bd !important;
}

/* Title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f5c842 0%, #f5a623 60%, #e8835a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Chat messages */
.user-msg {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 20%;
    color: #e8e6df;
    font-size: 0.92rem;
    line-height: 1.6;
}

.bot-msg {
    background: #111318;
    border: 1px solid #1f2330;
    border-left: 3px solid #f5c842;
    border-radius: 2px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 20%;
    color: #d4d2cb;
    font-size: 0.92rem;
    line-height: 1.6;
}

.role-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #f5c842;
    margin-bottom: 4px;
}

.role-label-user {
    text-align: right;
    color: #6b7280;
}

/* Sources box */
.sources-box {
    background: #0d0f14;
    border: 1px solid #1f2330;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
}

.sources-box .source-item {
    padding: 4px 0;
    border-bottom: 1px solid #1a1d27;
    color: #9ca3af;
}

/* Status badges */
.status-ok {
    display: inline-block;
    background: #14201a;
    border: 1px solid #1a3a28;
    color: #4ade80;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}

.status-pending {
    display: inline-block;
    background: #1e1a12;
    border: 1px solid #3a3020;
    color: #f5c842;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}

/* Section headers in sidebar */
.sidebar-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563 !important;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1f2330;
}

/* Input box styling */
.stTextInput input, .stTextArea textarea {
    background: #13161e !important;
    border: 1px solid #252836 !important;
    color: #e8e6df !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f5c842, #f5a623) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 0.03em;
}

.stButton > button:hover {
    opacity: 0.88 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13161e;
    border: 1px dashed #252836;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #f5c842 !important;
}

/* Divider */
hr {
    border-color: #1f2330 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #252836; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">PDF Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">RAG · FAISS · LangChain</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        ingest_btn = st.button("⚡ Ingest", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑 Clear", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.ingested_files = []
        st.session_state.chain = None
        st.rerun()

    if ingest_btn and uploaded_files:
        with st.spinner("Chunking & embedding PDFs…"):
            try:
                vs = ingest_pdfs(uploaded_files)
                st.session_state.vectorstore = vs
                st.session_state.ingested_files = [f.name for f in uploaded_files]
                retriever = get_retriever(vs)
                st.session_state.chain = build_chain(retriever)
                st.success(f"✓ {len(uploaded_files)} file(s) indexed")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown('<div class="sidebar-section">Indexed Files</div>', unsafe_allow_html=True)
    if st.session_state.ingested_files:
        for fname in st.session_state.ingested_files:
            st.markdown(f'<span class="status-ok">● {fname}</span>', unsafe_allow_html=True)
            st.markdown("")
    else:
        st.markdown('<span class="status-pending">○ No files yet</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Settings</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash"],
        label_visibility="collapsed",
    )
    top_k = st.slider("Source chunks (k)", 2, 8, 4)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#3b4255;text-align:center;">Built with LangChain · FAISS · Streamlit</div>',
        unsafe_allow_html=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title" style="font-size:1.6rem;margin-bottom:4px;">Ask your documents</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload PDFs → Ingest → Chat</div>', unsafe_allow_html=True)
st.markdown("---")

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="role-label role-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="role-label">Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        source = src.metadata.get("source", "unknown")
                        snippet = src.page_content[:220].replace("\n", " ")
                        st.markdown(f"""
<div class="sources-box">
  <div style="color:#f5c842;font-family:'DM Mono',monospace;font-size:0.72rem;">
    [{i}] {Path(source).name} · page {page}
  </div>
  <div class="source-item" style="margin-top:6px;">{snippet}…</div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask a question",
            placeholder="What does this document say about…?",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_input.strip():
    if st.session_state.chain is None:
        st.warning("⚠️ Please upload and ingest PDFs first.")
    else:
        # Rebuild chain with current settings if changed
        retriever = get_retriever(st.session_state.vectorstore, k=top_k)
        chain = build_chain(retriever, model_name=model)

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking…"):
            try:
                result = chain.invoke({
                    "input": user_input,
                    "chat_history": [
                        (m["role"], m["content"])
                        for m in st.session_state.messages[:-1]
                        if m["role"] in ("user", "assistant")
                    ],
                })
                answer = result.get("answer", result.get("output", str(result)))
                sources = result.get("context", [])
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
