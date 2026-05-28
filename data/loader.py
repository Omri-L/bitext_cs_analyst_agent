"""
data/loader.py

Loads the Bitext Customer Service dataset from HuggingFace and exposes
raw data-access functions used by both the LangChain tools and the MCP server.

Dataset columns:
    flags       - language-generation tags (single-letter codes, e.g. 'BIQ')
    instruction - the customer's utterance
    category    - top-level group (e.g. ORDER, ACCOUNT, REFUND)
    intent      - fine-grained label (e.g. cancel_order, get_refund)
    response    - the agent's reply
"""

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

_NULLISH = {"", "null", "none", "n/a", "na"}

def _clean_filter(value):
    """Treat model-supplied stringified nulls ('null', 'none', '') as a real None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULLISH else text

# Fixed seed for *analytical* sampling (summaries, flag examples) so those
# results are reproducible. get_examples() deliberately stays random unless a
# seed is passed, so "show me N more" returns fresh rows.
ANALYSIS_SEED = 42


@lru_cache(maxsize=1)
def load_bitext() -> pd.DataFrame:
    """Load and cache the Bitext dataset as a pandas DataFrame.
    Uses lru_cache so the dataset is downloaded once per process.
    """
    print("[data] Loading Bitext dataset from HuggingFace (first run may take a moment)...")
    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
    df = dataset["train"].to_pandas()
    # Normalise casing so queries are case-insensitive
    df["category"] = df["category"].str.upper()
    df["intent"] = df["intent"].str.lower()
    print(
        f"[data] Dataset loaded: {len(df):,} rows, "
        f"{df['category'].nunique()} categories, {df['intent'].nunique()} intents."
    )
    return df


# ---------------------------------------------------------------------------
# TF-IDF index for semantic search (built once, lazily)
# ---------------------------------------------------------------------------

_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None


def _get_tfidf_index():
    """Build (or return cached) TF-IDF index over the instruction column.

    Return global variables:
        _vectorizer   - learns a vocabulary from all customer instructions.
        _tfidf_matrix - all instructions translated into TF-IDF vectors.
    """
    global _vectorizer, _tfidf_matrix
    if _vectorizer is None:
        df = load_bitext()
        _vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000)
        _tfidf_matrix = _vectorizer.fit_transform(df["instruction"].fillna(""))
    return _vectorizer, _tfidf_matrix


# ---------------------------------------------------------------------------
# Basic data operations
# ---------------------------------------------------------------------------

def get_categories_impl() -> list[str]:
    """Return all unique category names, sorted alphabetically."""
    return sorted(load_bitext()["category"].unique().tolist())


def get_intents_impl(category: str | None = None) -> list[str]:
    """Return all unique intents, optionally filtered by category."""
    df = load_bitext()
    category = _clean_filter(category)
    if category:
        df = df[df["category"] == category.upper()]
        if df.empty:
            return []
    return sorted(df["intent"].unique().tolist())


def count_rows_impl(category: str | None = None, intent: str | None = None) -> int:
    """Count rows matching the optional category and/or intent filters."""
    df = load_bitext()
    category = _clean_filter(category)
    intent = _clean_filter(intent)

    if category:
        df = df[df["category"] == category.upper()]
    if intent:
        df = df[df["intent"] == intent.lower()]
    return len(df)


def get_distribution_impl(category: str) -> dict[str, int]:
    """Return the intent-count distribution within a given category."""
    df = load_bitext()
    category = _clean_filter(category)
    if not category:
        return {}
    filtered = df[df["category"] == category.upper()]
    if filtered.empty:
        return {}
    return filtered["intent"].value_counts().to_dict()


def get_examples_impl(
    n: int,
    category: str | None = None,
    intent: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Sample N rows (instruction + response + metadata) matching the filters.

    Sampling is random by default so repeated calls (e.g. "show me 3 more")
    return fresh rows. Pass an explicit ``seed`` for a reproducible sample.
    """
    df = load_bitext()
    category = _clean_filter(category)
    intent = _clean_filter(intent)
    if category:
        df = df[df["category"] == category.upper()]
    if intent:
        df = df[df["intent"] == intent.lower()]

    if df.empty:
        return []

    random_state = seed if seed is not None else random.randint(0, 999_999)
    sample = df.sample(min(n, len(df)), random_state=random_state)
    return sample[["instruction", "response", "category", "intent"]].to_dict("records")


def semantic_search_impl(query: str, n: int = 5) -> list[dict]:
    """Find rows whose instruction is most semantically similar to the query.

    Uses TF-IDF cosine similarity so no external embedding model is needed.
    """
    df = load_bitext()
    vectorizer, tfidf_matrix = _get_tfidf_index()

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:n]

    results = df.iloc[top_indices][["instruction", "response", "category", "intent"]].copy()
    results["similarity_score"] = scores[top_indices].round(3)
    return results.to_dict("records")


# ---------------------------------------------------------------------------
# Text length statistics
# ---------------------------------------------------------------------------

def compute_text_stats_impl(
    field: str,
    group_by: str | None = None,
    unit: str = "words",
) -> dict:
    """Compute length statistics for a text column, optionally grouped.

    Args:
        field:    'instruction' or 'response'.
        group_by: 'category', 'intent', or None for a dataset-wide summary.
        unit:     'words' or 'characters'.

    Returns:
        A dict with overall stats, or per-group stats sorted longest-first.
    """
    df = load_bitext()
    texts = df[field].fillna("") # fill NaNs or None with empty strings
    lengths = texts.str.len() if unit == "characters" else texts.str.split().str.len() # character or word counts
    work = df.assign(_length=lengths)

    def _summary(series: pd.Series) -> dict:
        return {
            "mean": round(float(series.mean()), 1),
            "median": round(float(series.median()), 1),
            "min": int(series.min()),
            "max": int(series.max()),
            "rows": int(series.size),
        }

    if group_by:
        if group_by not in {"category", "intent"}:
            return {"error": "group_by must be 'category' or 'intent'."}
        per_group = {key: _summary(grp) for key, grp in work.groupby(group_by)["_length"]}
        ordered = dict(
            sorted(per_group.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        )
        return {"field": field, "unit": unit, "group_by": group_by, "groups": ordered}

    return {"field": field, "unit": unit, "overall": _summary(work["_length"])}


# ---------------------------------------------------------------------------
# Language-generation flags
# ---------------------------------------------------------------------------

# Legend for the `flags` column, taken from the official Bitext dataset card.
FLAG_LEGEND = {
    "M": "Morphological variation (inflectional/derivational)",
    "L": "Semantic/lexical variation (synonyms, hyphenation, compounding)",
    "B": "Basic syntactic structure",
    "I": "Interrogative structure (phrased as a question)",
    "C": "Coordinated syntactic structure (several requests joined)",
    "N": "Negation",
    "P": "Politeness variation",
    "Q": "Colloquial variation (informal, e.g. 'can u activ8 my SIM')",
    "W": "Offensive language",
    "K": "Keyword mode (terse keywords instead of full sentences)",
    "E": "Use of abbreviations / expanded forms",
    "Z": "Errors and typos (spelling, punctuation)",
}


def inspect_flags_impl(flag: str | None = None, n: int = 5) -> dict:
    """Explore the `flags` column.

    Args:
        flag: a single flag code. If omitted, return the legend + per-flag counts.
        n:    number of example rows to return when a flag is given.

    Returns:
        Either a dataset-wide flag summary or example rows carrying the flag.
    """
    df = load_bitext()
    flag_series = df["flags"].fillna("").astype(str)

    if not flag:
        counter: Counter = Counter()
        for tags in flag_series:
            counter.update(code for code in set(tags) if code.isalpha())
        summary = {
            code: {
                "meaning": FLAG_LEGEND.get(code, "Unknown / undocumented flag code"),
                "row_count": int(counter[code]),
            }
            for code in sorted(counter)
        }
        return {
            "total_rows": int(len(df)),
            "note": (
                "Each row's `flags` value is a string of single-letter codes; "
                "a row can carry several flags at once."
            ),
            "flags": summary,
        }

    code = flag.strip().upper()[:1]
    matched = df[flag_series.str.contains(code, regex=False)]
    if matched.empty:
        return {
            "error": f"No rows carry flag '{code}'. "
            "Call inspect_flags() with no argument to see valid codes."
        }
    sample = matched.sample(min(n, len(matched)))
    return {
        "flag": code,
        "meaning": FLAG_LEGEND.get(code, "Unknown / undocumented flag code"),
        "matching_rows": int(len(matched)),
        "examples": sample[
            ["instruction", "response", "category", "intent", "flags"]
        ].to_dict("records"),
    }
