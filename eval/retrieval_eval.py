"""
retrieval_eval.py
-----------------
Quota-free evaluation of *retrieval* quality (no generation LLM calls — only
query embeddings + local BM25 + the local cross-encoder reranker).

It answers the real engineering question with numbers: **when** does hybrid
search + reranking beat naive dense retrieval?

We run the same gold set under two embedders:

  - Strong embedder (Gemini `gemini-embedding-001`, 3072-dim): a top-tier
    model that already ranks answers well on its own.
  - Weak embedder (local `all-MiniLM-L6-v2`, 384-dim): a cheap, offline
    model — representative of cost-constrained deployments.

The finding (see eval/retrieval_results.md): with a strong embedder, dense
is already near-perfect and extra machinery adds little; with a weak
embedder, dense degrades and **hybrid + reranking recovers most of the loss**.
That is exactly the tradeoff these techniques exist for.

Metrics (higher is better), scored against the gold `expected_substring`:
  - mrr@k      : mean reciprocal rank of the correct chunk (headline)
  - hit_rate@k : fraction of questions whose correct chunk is in the top-k

Usage:
    ./venv/bin/python -m eval.retrieval_eval
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings

from eval.sample_corpus import CORPUS, EVAL_SET
from rag.ingestor import get_embeddings
from rag.retriever import build_retriever, get_dense_retriever
from rag.vectorstore import build_vectorstore

K = 4
WEAK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _first_hit_rank(docs, expected_substring: str) -> int | None:
    """1-based rank (within top-K) of the first chunk containing the gold
    phrase, or None if the correct chunk is not retrieved."""
    needle = expected_substring.lower()
    for i, d in enumerate(docs[:K], start=1):
        if needle in d.page_content.lower():
            return i
    return None


def score_retriever(retriever) -> dict[str, float]:
    hits, rr = [], []
    for item in EVAL_SET:
        rank = _first_hit_rank(retriever.invoke(item["question"]), item["expected_substring"])
        hits.append(1.0 if rank else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
    n = len(EVAL_SET)
    return {"mrr@k": sum(rr) / n, "hit_rate@k": sum(hits) / n}


def eval_embedder(label: str, embedding) -> pd.DataFrame:
    """Score dense vs hybrid+rerank for one embedding model."""
    print(f"\n### Embedder: {label}")
    vs = build_vectorstore(CORPUS, embedding=embedding)
    configs = {
        "Dense (vector only)": get_dense_retriever(vs, k=K),
        "Hybrid + rerank": build_retriever(vs, CORPUS, k=K),
    }
    rows = {}
    for name, retriever in configs.items():
        rows[name] = score_retriever(retriever)
        print(f"   {name:22s} mrr={rows[name]['mrr@k']:.3f} "
              f"hit={rows[name]['hit_rate@k']:.3f}")
    return pd.DataFrame(rows).T.round(3)


def main() -> None:
    experiments = {
        "Strong embedder — Gemini gemini-embedding-001 (3072-dim)": get_embeddings(),
        f"Weak embedder — local {WEAK_MODEL.split('/')[-1]} (384-dim)":
            HuggingFaceEmbeddings(model_name=WEAK_MODEL),
    }
    results = {label: eval_embedder(label, emb) for label, emb in experiments.items()}
    _write_report(results)
    print("\nWrote eval/retrieval_results.md")


def _write_report(results: dict[str, pd.DataFrame]) -> None:
    md = [
        "# Retrieval Evaluation (quota-free)",
        "",
        f"Gold set: {len(EVAL_SET)} questions over a synthetic corpus with "
        f"confusable near-duplicates and exact identifiers "
        f"(`eval/sample_corpus.py`), k={K}. No generation LLM is used — only "
        "query embeddings + local BM25 + a local cross-encoder reranker.",
        "",
        "**Finding:** a strong embedder already ranks answers well on its own, "
        "so hybrid + reranking adds little; a weak/cheap embedder degrades, and "
        "hybrid + reranking recovers most of the lost ranking quality. Pick the "
        "retrieval stack to match the embedder — don't add machinery by reflex.",
        "",
    ]
    for label, df in results.items():
        md += [f"## {label}", "", df.to_markdown(), ""]
    md += [
        "- **mrr@k** — mean reciprocal rank of the correct chunk "
        "(1.0 = always first); the headline metric",
        "- **hit_rate@k** — correct chunk appears anywhere in the top-k",
        "",
        "_Regenerate with `./venv/bin/python -m eval.retrieval_eval`._",
    ]
    Path("eval/retrieval_results.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
