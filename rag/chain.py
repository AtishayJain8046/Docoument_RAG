"""
chain.py
--------
Builds a conversational RAG (Retrieval-Augmented Generation) chain.

RAG pipeline step: Query + History → Retriever → LLM → Answer

Built entirely from langchain_core runnables — no deprecated langchain.chains needed.
"""

from typing import Iterator, List, Tuple

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from rag.config import settings


# ── Prompts ───────────────────────────────────────────────────────────────────

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite it as a "
     "standalone question that makes sense without the history. "
     "If it's already standalone, return it as-is. Do NOT answer the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_SYSTEM_PROMPT = """\
You are an expert document assistant. Use ONLY the retrieved context below to \
answer the user's question. Be concise, accurate, and cite page numbers when possible.

If the answer is not contained in the context, say:
"I don't have enough information in the provided documents to answer that."

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def format_docs(docs: List[Document]) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')} | {doc.metadata.get('source', '')}]\n{doc.page_content}"
        for doc in docs
    )


def make_llm(model_name: str | None = None) -> GoogleGenerativeAI:
    """Construct the Gemini LLM from config (overridable model name)."""
    return GoogleGenerativeAI(
        model=model_name or settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key or None,
    )


def standalone_question(llm, inputs: dict) -> str:
    """Rewrite a follow-up into a self-contained question using chat history."""
    if inputs.get("chat_history"):
        return (CONTEXTUALIZE_PROMPT | llm | StrOutputParser()).invoke(inputs)
    return inputs["input"]


def _qa_payload(inputs: dict, docs: List[Document]) -> dict:
    return {
        "context": format_docs(docs),
        "input": inputs["input"],
        "chat_history": inputs.get("chat_history", []),
    }


def build_chain(retriever, model_name: str | None = None):
    """
    Constructs the full conversational RAG chain using pure langchain_core runnables.

    Returns a chain that accepts:
        {"input": "user question", "chat_history": [("user", "..."), ("assistant", "...")]}
    And returns:
        {"answer": "...", "context": [Document, ...]}
    """
    llm = make_llm(model_name)
    answer_chain = QA_PROMPT | llm | StrOutputParser()

    def full_chain(inputs: dict) -> dict:
        question = standalone_question(llm, inputs)
        docs = retriever.invoke(question)
        answer = answer_chain.invoke(_qa_payload(inputs, docs))
        return {"answer": answer, "context": docs}

    return RunnableLambda(full_chain)


def stream_answer(
    retriever, inputs: dict, model_name: str | None = None
) -> Tuple[List[Document], Iterator[str]]:
    """
    Streaming variant for the API / SSE.

    Retrieves first (so sources are known up front) and returns:
        (source_documents, token_iterator)
    where token_iterator yields answer chunks as they are generated.
    """
    llm = make_llm(model_name)
    question = standalone_question(llm, inputs)
    docs = retriever.invoke(question)
    answer_chain = QA_PROMPT | llm | StrOutputParser()
    return docs, answer_chain.stream(_qa_payload(inputs, docs))
