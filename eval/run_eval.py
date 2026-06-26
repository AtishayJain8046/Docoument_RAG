"""
run_eval.py
-----------
RAG evaluation harness.

Runs the gold Q/A set through two retrieval configurations and scores each
with RAGAS (LLM-as-judge), so we can show — with numbers — that hybrid
search + reranking beats naive dense top-k:

    Baseline   : dense (vector) similarity, top-4
    Improved   : hybrid (dense + BM25) + cross-encoder reranking

Outputs a markdown comparison table to eval/results.md.

Usage:
    ./venv/bin/python -m eval.run_eval
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from eval.sample_corpus import CORPUS, EVAL_SET
from rag.chain import build_chain
from rag.config import settings
from rag.ingestor import get_embeddings
from rag.retriever import build_retriever, get_dense_retriever
from rag.vectorstore import build_vectorstore

INPUT_COLS = {"user_input", "retrieved_contexts", "response", "reference"}


def gold_questions() -> list[dict]:
    """Gold set, optionally truncated to fit free-tier quotas."""
    limit = settings.eval_sample_limit
    return EVAL_SET[:limit] if limit else EVAL_SET


def collect_samples(chain) -> list[dict]:
    """Run every gold question through a chain and gather RAGAS samples."""
    samples = []
    for item in gold_questions():
        result = chain.invoke({"input": item["question"], "chat_history": []})
        contexts = [d.page_content for d in result.get("context", [])]
        samples.append(
            {
                "user_input": item["question"],
                "response": result.get("answer", ""),
                "retrieved_contexts": contexts or ["(no context retrieved)"],
                "reference": item["reference"],
            }
        )
    return samples


def score(samples: list[dict], judge_llm, judge_emb) -> dict[str, float]:
    """Score a batch of samples with RAGAS and return mean per metric."""
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        # Serialize calls + generous timeout to stay under Gemini free-tier
        # rate limits (bursts otherwise 429 → backoff → job timeouts).
        run_config=RunConfig(max_workers=1, timeout=600, max_retries=8),
        show_progress=False,
    )
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in INPUT_COLS]
    return {c: float(df[c].mean()) for c in metric_cols}


def main() -> None:
    if not settings.has_api_key:
        raise SystemExit("GOOGLE_API_KEY not set — eval needs it for the judge LLM.")

    print("Building index from sample corpus…")
    vs = build_vectorstore(CORPUS)

    answer_model = settings.eval_judge_model  # cheap model, separate quota
    configs = {
        "Baseline (dense top-4)": build_chain(
            get_dense_retriever(vs, k=4), model_name=answer_model
        ),
        "Hybrid + rerank": build_chain(
            build_retriever(vs, CORPUS, k=4), model_name=answer_model
        ),
    }

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.eval_judge_model,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings())

    results: dict[str, dict[str, float]] = {}
    for name, chain in configs.items():
        print(f"\n=== Evaluating: {name} ===")
        samples = collect_samples(chain)
        results[name] = score(samples, judge_llm, judge_emb)
        for metric, val in results[name].items():
            print(f"   {metric:38s} {val:.3f}")

    _write_report(results)
    print("\nWrote eval/results.md")


def _write_report(results: dict[str, dict[str, float]]) -> None:
    table = pd.DataFrame(results).T  # configs as rows, metrics as columns
    md = [
        "# RAG Evaluation Results",
        "",
        f"Gold set: {len(gold_questions())} questions over a synthetic corpus "
        f"(`eval/sample_corpus.py`). Answer + judge LLM: "
        f"`{settings.eval_judge_model}`. "
        "Metrics via RAGAS (higher is better, 0–1).",
        "",
        table.round(3).to_markdown(),
        "",
        "- **faithfulness** — answer is grounded in retrieved context (no hallucination)",
        "- **answer_relevancy** — answer addresses the question",
        "- **context_precision** — retrieved chunks are relevant (signal vs noise)",
        "- **context_recall** — retrieval found the chunks needed for the answer",
        "",
        "_Regenerate with `./venv/bin/python -m eval.run_eval`._",
    ]
    Path("eval/results.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
