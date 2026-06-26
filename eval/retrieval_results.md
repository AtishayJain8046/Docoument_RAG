# Retrieval Evaluation (quota-free)

Gold set: 13 questions over a synthetic corpus with confusable near-duplicates and exact identifiers (`eval/sample_corpus.py`), k=4. No generation LLM is used — only query embeddings + local BM25 + a local cross-encoder reranker.

**Finding:** a strong embedder already ranks answers well on its own, so hybrid + reranking adds little; a weak/cheap embedder degrades, and hybrid + reranking recovers most of the lost ranking quality. Pick the retrieval stack to match the embedder — don't add machinery by reflex.

## Strong embedder — Gemini gemini-embedding-001 (3072-dim)

|                     |   mrr@k |   hit_rate@k |
|:--------------------|--------:|-------------:|
| Dense (vector only) |   1     |            1 |
| Hybrid + rerank     |   0.962 |            1 |

## Weak embedder — local all-MiniLM-L6-v2 (384-dim)

|                     |   mrr@k |   hit_rate@k |
|:--------------------|--------:|-------------:|
| Dense (vector only) |   0.923 |            1 |
| Hybrid + rerank     |   0.962 |            1 |

- **mrr@k** — mean reciprocal rank of the correct chunk (1.0 = always first); the headline metric
- **hit_rate@k** — correct chunk appears anywhere in the top-k

_Regenerate with `./venv/bin/python -m eval.retrieval_eval`._