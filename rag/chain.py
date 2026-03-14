"""
chain.py
--------
Builds a conversational RAG (Retrieval-Augmented Generation) chain.

RAG pipeline step: Query + History → Retriever → LLM → Answer

Built entirely from langchain_core runnables — no deprecated langchain.chains needed.
"""

import os
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


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


def build_chain(retriever: VectorStoreRetriever, model_name: str = "gpt-4o-mini"):
    """
    Constructs the full conversational RAG chain using pure langchain_core runnables.

    Returns a chain that accepts:
        {"input": "user question", "chat_history": [("user", "..."), ("assistant", "...")]}
    And returns:
        {"answer": "...", "context": [Document, ...]}
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 1: Rewrite the question to be standalone (handles follow-ups)
    contextualize_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    def get_standalone_question(inputs: dict) -> str:
        if inputs.get("chat_history"):
            return contextualize_chain.invoke(inputs)
        return inputs["input"]

    # Step 2: Retrieve docs based on (possibly rewritten) question
    def retrieve_docs(inputs: dict):
        question = get_standalone_question(inputs)
        docs = retriever.invoke(question)
        return {"context": docs, "input": inputs["input"], "chat_history": inputs.get("chat_history", [])}

    # Step 3: Generate answer from context
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    # Full chain: retrieve then answer, returning both answer and source docs
    def full_chain(inputs: dict) -> dict:
        retrieved = retrieve_docs(inputs)
        answer = answer_chain.invoke(retrieved)
        return {"answer": answer, "context": retrieved["context"]}

    return RunnableLambda(full_chain)
