"""
sample_corpus.py
----------------
A small, fully synthetic corpus + gold Q/A set for evaluation.

It is deliberately about a *fictional* company so that correct answers must
come from retrieval — the LLM cannot fall back on memorized world knowledge.
This makes faithfulness / context-recall numbers meaningful rather than
flattering.
"""

from langchain_core.documents import Document

# ── Knowledge base (each item becomes one or more chunks) ──────────────
CORPUS = [
    Document(
        page_content=(
            "Aurelia Dynamics was founded in 2017 in Tallinn, Estonia, by "
            "Mara Voss and Idris Kaine. The company builds industrial drone "
            "inspection systems for offshore wind farms."
        ),
        metadata={"source": "aurelia_overview.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "The flagship product, the Aurelia Skylark X2, has a maximum "
            "flight time of 47 minutes and a wind tolerance of 22 meters per "
            "second. It carries a 48-megapixel thermal-RGB dual sensor."
        ),
        metadata={"source": "skylark_x2_spec.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "The Skylark X2 transmits inspection data over an encrypted "
            "5.8 GHz link with a range of 9 kilometers. Data is stored in "
            "the Aurelia Vault cloud platform under SOC 2 Type II controls."
        ),
        metadata={"source": "skylark_x2_spec.pdf", "page": 2},
    ),
    Document(
        page_content=(
            "Aurelia's pricing model is subscription-based. The Standard tier "
            "costs 1,900 euros per drone per month and includes 500 "
            "inspection flights. The Enterprise tier is custom-priced and "
            "adds unlimited flights and a dedicated support engineer."
        ),
        metadata={"source": "aurelia_pricing.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "In 2023 Aurelia Dynamics reported annual recurring revenue of "
            "14.2 million euros, up 63 percent year over year, and served "
            "31 offshore wind operators across the North Sea and Baltic Sea."
        ),
        metadata={"source": "aurelia_2023_report.pdf", "page": 4},
    ),
    Document(
        page_content=(
            "Regulatory note: all Skylark X2 deployments in EU waters operate "
            "under EASA Specific category authorization SORA-2019, requiring "
            "a remote pilot certificate and a pre-flight risk assessment."
        ),
        metadata={"source": "aurelia_compliance.pdf", "page": 2},
    ),
]

# ── Gold question / reference-answer pairs ─────────────────────────────
EVAL_SET = [
    {
        "question": "Who founded Aurelia Dynamics and when?",
        "reference": "Aurelia Dynamics was founded in 2017 by Mara Voss and Idris Kaine.",
    },
    {
        "question": "What is the maximum flight time of the Skylark X2?",
        "reference": "The Skylark X2 has a maximum flight time of 47 minutes.",
    },
    {
        "question": "How much does the Standard pricing tier cost and what does it include?",
        "reference": "The Standard tier costs 1,900 euros per drone per month and includes 500 inspection flights.",
    },
    {
        "question": "What was Aurelia's 2023 annual recurring revenue?",
        "reference": "In 2023 Aurelia reported annual recurring revenue of 14.2 million euros, up 63 percent year over year.",
    },
    {
        "question": "What authorization is required to fly the Skylark X2 in EU waters?",
        "reference": "EASA Specific category authorization SORA-2019, requiring a remote pilot certificate and a pre-flight risk assessment.",
    },
]
