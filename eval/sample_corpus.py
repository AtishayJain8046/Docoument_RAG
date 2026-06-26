"""
sample_corpus.py
----------------
A synthetic corpus + gold Q/A set for evaluation, deliberately designed to
*separate* retrieval strategies rather than flatter them.

It is about a fictional company (so answers must come from retrieval, not the
LLM's memory) and is built with:
  - confusable near-duplicates: three drone models with near-identical spec
    phrasing, so pure vector similarity struggles to pick the right one;
  - exact identifiers / part numbers / acronyms, where keyword (BM25) search
    beats dense search;
  - distractor entities (a similarly-named competitor).

This makes hybrid search + reranking measurably better than naive dense
top-k, which is the whole point of the eval.

Each gold item carries `expected_substring`: a phrase unique to the single
correct chunk, so the retrieval eval can score the *right chunk* (not just
the right document).
"""

from langchain_core.documents import Document


def _doc(text: str, source: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


# ── Large near-duplicate catalog ──────────────────────────────────────
# A realistic stress test: dozens of products with near-identical phrasing
# that differ only by an exact model id and a few numbers. Dense embeddings
# blur "F37" vs "F38"; exact-token BM25 + a cross-encoder reranker
# disambiguate. This is where hybrid retrieval genuinely earns its keep.
_CATALOG_N = 40


def _catalog() -> list[Document]:
    docs = []
    for n in range(1, _CATALOG_N + 1):
        flight = 30 + (n % 25)          # 30–54 minutes
        rng = 4 + (n % 12)              # 4–15 km
        payload = round(1.0 + (n % 7) * 0.3, 1)
        docs.append(
            _doc(
                f"The Falcon F{n} inspection drone has a maximum flight time "
                f"of {flight} minutes, a transmission range of {rng} "
                f"kilometers, and a payload capacity of {payload} kilograms. "
                f"Part number FL-F{n}-STD.",
                "falcon_catalog.pdf",
                page=n,
            )
        )
    return docs


def _catalog_questions() -> list[dict]:
    # Spread across the catalog so ranking pressure is real.
    qs = []
    for n in (7, 13, 22, 31, 38):
        flight = 30 + (n % 25)
        qs.append(
            {
                "question": f"What is the maximum flight time of the Falcon F{n}?",
                "reference": f"The Falcon F{n} has a maximum flight time of {flight} minutes.",
                "expected_substring": f"Falcon F{n} inspection drone has a maximum flight time of {flight} minutes",
            }
        )
    return qs


CORPUS = [
    # ── Company overview ──────────────────────────────────────────────
    _doc(
        "Aurelia Dynamics was founded in 2017 in Tallinn, Estonia, by Mara "
        "Voss and Idris Kaine. It builds drone inspection systems for "
        "offshore wind farms.",
        "aurelia_overview.pdf",
    ),
    # ── Distractor competitor (name-similar) ──────────────────────────
    _doc(
        "Auriga Robotics, a separate company founded in 2015 in Riga, also "
        "builds inspection drones but focuses on onshore solar farms, not "
        "offshore wind.",
        "market_landscape.pdf",
    ),
    # ── Confusable model specs (near-identical phrasing) ──────────────
    _doc(
        "The Aurelia Skylark X1 has a maximum flight time of 38 minutes, a "
        "wind tolerance of 16 meters per second, and a 24-megapixel RGB "
        "sensor. Part number SK-X1-A.",
        "skylark_specs.pdf",
        1,
    ),
    _doc(
        "The Aurelia Skylark X2 has a maximum flight time of 47 minutes, a "
        "wind tolerance of 22 meters per second, and a 48-megapixel "
        "thermal-RGB dual sensor. Part number SK-X2-B.",
        "skylark_specs.pdf",
        2,
    ),
    _doc(
        "The Aurelia Skylark X3 has a maximum flight time of 53 minutes, a "
        "wind tolerance of 27 meters per second, and a 61-megapixel "
        "thermal-RGB-LiDAR sensor. Part number SK-X3-C.",
        "skylark_specs.pdf",
        3,
    ),
    # ── Confusable connectivity specs ─────────────────────────────────
    _doc(
        "The Skylark X1 transmits over a 2.4 GHz link with a range of 5 "
        "kilometers.",
        "skylark_connectivity.pdf",
        1,
    ),
    _doc(
        "The Skylark X2 transmits over an encrypted 5.8 GHz link with a "
        "range of 9 kilometers, stored in Aurelia Vault under SOC 2 Type II.",
        "skylark_connectivity.pdf",
        2,
    ),
    _doc(
        "The Skylark X3 transmits over a dual-band 5.8/3.5 GHz link with a "
        "range of 14 kilometers and optional satellite uplink.",
        "skylark_connectivity.pdf",
        3,
    ),
    # ── Pricing (confusable tiers) ────────────────────────────────────
    _doc(
        "Aurelia Standard tier costs 1,900 euros per drone per month and "
        "includes 500 inspection flights.",
        "aurelia_pricing.pdf",
        1,
    ),
    _doc(
        "Aurelia Professional tier costs 3,400 euros per drone per month and "
        "includes 1,500 inspection flights plus priority support.",
        "aurelia_pricing.pdf",
        2,
    ),
    _doc(
        "Aurelia Enterprise tier is custom-priced and adds unlimited flights "
        "and a dedicated support engineer.",
        "aurelia_pricing.pdf",
        3,
    ),
    # ── Financials across years (confusable) ──────────────────────────
    _doc(
        "In 2021, Aurelia Dynamics reported annual recurring revenue of 5.1 "
        "million euros and served 11 offshore wind operators.",
        "aurelia_financials.pdf",
        1,
    ),
    _doc(
        "In 2022, Aurelia Dynamics reported annual recurring revenue of 8.7 "
        "million euros and served 19 offshore wind operators.",
        "aurelia_financials.pdf",
        2,
    ),
    _doc(
        "In 2023, Aurelia Dynamics reported annual recurring revenue of 14.2 "
        "million euros, up 63 percent year over year, and served 31 offshore "
        "wind operators across the North Sea and Baltic Sea.",
        "aurelia_financials.pdf",
        3,
    ),
    # ── Compliance / acronyms (BM25 territory) ────────────────────────
    _doc(
        "All Skylark deployments in EU waters operate under EASA Specific "
        "category authorization SORA-2019, requiring a remote pilot "
        "certificate and a pre-flight risk assessment.",
        "aurelia_compliance.pdf",
        1,
    ),
    _doc(
        "US deployments require FAA Part 107 certification and, for beyond "
        "visual line of sight, a BVLOS waiver under 14 CFR 107.31.",
        "aurelia_compliance.pdf",
        2,
    ),
    # ── Support / misc distractors ────────────────────────────────────
    _doc(
        "Aurelia Vault retains inspection imagery for 7 years by default; "
        "Enterprise customers may configure a 10-year retention policy.",
        "aurelia_vault.pdf",
    ),
    _doc(
        "The Aurelia warranty covers manufacturing defects for 24 months "
        "from the date of delivery, excluding propeller wear.",
        "aurelia_warranty.pdf",
    ),
] + _catalog()

# `expected_substring` is a phrase unique to the single correct chunk.
EVAL_SET = [
    {
        "question": "What is the maximum flight time of the Skylark X2?",
        "reference": "The Skylark X2 has a maximum flight time of 47 minutes.",
        "expected_substring": "flight time of 47 minutes",
    },
    {
        "question": "What sensor does the Skylark X3 carry?",
        "reference": "The Skylark X3 carries a 61-megapixel thermal-RGB-LiDAR sensor.",
        "expected_substring": "61-megapixel thermal-RGB-LiDAR",
    },
    {
        "question": "What is the part number of the Skylark X1?",
        "reference": "The Skylark X1 part number is SK-X1-A.",
        "expected_substring": "SK-X1-A",
    },
    {
        "question": "What is the transmission range of the Skylark X3?",
        "reference": "The Skylark X3 has a range of 14 kilometers.",
        "expected_substring": "range of 14 kilometers",
    },
    {
        "question": "How much does the Aurelia Professional tier cost?",
        "reference": "The Professional tier costs 3,400 euros per drone per month.",
        "expected_substring": "3,400 euros per drone per month",
    },
    {
        "question": "What was Aurelia's annual recurring revenue in 2022?",
        "reference": "In 2022 Aurelia reported annual recurring revenue of 8.7 million euros.",
        "expected_substring": "8.7 million euros",
    },
    {
        "question": "What authorization is required to fly in EU waters?",
        "reference": "EASA Specific category authorization SORA-2019.",
        "expected_substring": "SORA-2019",
    },
    {
        "question": "What waiver is needed for BVLOS flights in the US?",
        "reference": "A BVLOS waiver under 14 CFR 107.31.",
        "expected_substring": "14 CFR 107.31",
    },
] + _catalog_questions()
