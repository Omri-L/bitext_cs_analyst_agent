"""
agent/tools.py

LangChain tool definitions for the data analyst agent.
Each tool wraps a raw function from data/loader.py.

Design principles:
- Every tool has a clear description the LLM can act on without extra context.
- Input schemas use Pydantic BaseModel so LangGraph validates arguments.
- Return types are typed so the LLM knows what to expect.

User memory is NOT a tool: it is handled automatically by the load_profile and
update_profile graph nodes, so the agent never has to manage it explicitly.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data.loader import (
    ANALYSIS_SEED,
    compute_text_stats_impl,
    count_rows_impl,
    get_categories_impl,
    get_distribution_impl,
    get_examples_impl,
    get_intents_impl,
    inspect_flags_impl,
    semantic_search_impl,
)

# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


@tool("get_categories")
def get_categories() -> list[str]:
    """Return all unique top-level categories in the Bitext dataset
    (e.g. ORDER, ACCOUNT, REFUND, FEEDBACK, SHIPPING_ADDRESS).
    Use this first when you are unsure which categories exist or how one is named.
    Example:
        get_categories() -> ['ACCOUNT', 'CANCELLATION_FEE', 'CONTACT', ...]
    """
    return get_categories_impl()


class GetIntentsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter intents to this category (e.g. 'ORDER', 'ACCOUNT'). "
        "Leave empty to list all intents across the dataset.",
    )


@tool("get_intents", args_schema=GetIntentsInput)
def get_intents(category: str | None = None) -> list[str]:
    """Return all unique intents in the dataset, optionally filtered by category.
    Use this to discover exact intent names before filtering by intent in other
    tools — intent names are non-obvious from natural language ('refund' ->
    'get_refund', 'complaint' -> 'complaint').
    Example:
        get_intents(category='ORDER') -> ['cancel_order', 'place_order', ...]
    """
    return get_intents_impl(category)


# ---------------------------------------------------------------------------
# Counting and distribution
# ---------------------------------------------------------------------------


class CountRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter by category name (e.g. 'REFUND'). Optional.",
    )
    intent: str | None = Field(
        default=None,
        description="Filter by intent name (e.g. 'get_refund'). Optional.",
    )


@tool("count_rows", args_schema=CountRowsInput)
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count the number of rows matching the given category and/or intent filters.
    Use this for any 'how many' question. With no filters it returns the total
    dataset size.
    Example:
        count_rows(intent='get_refund') -> 1012
    """
    return count_rows_impl(category, intent)


class GetDistributionInput(BaseModel):
    category: str = Field(
        description="The category whose intent distribution you want "
        "(e.g. 'ACCOUNT', 'ORDER')."
    )


@tool("get_distribution", args_schema=GetDistributionInput)
def get_distribution(category: str) -> dict:
    """Return the count of each intent within a given category.
    Use this for 'what is the distribution / breakdown' questions.
    Example:
        get_distribution('ACCOUNT') -> {'create_account': 998, 'delete_account': 1023, ...}
    """
    result = get_distribution_impl(category)
    if not result:
        return {
            "error": f"Category '{category}' not found. "
            "Use get_categories() to see valid categories."
        }
    return result


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


class GetExamplesInput(BaseModel):
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of example rows to return (1-20).",
    )
    category: str | None = Field(
        default=None,
        description="Filter examples to this category. Optional.",
    )
    intent: str | None = Field(
        default=None,
        description="Filter examples to this intent (e.g. 'cancel_order'). Optional.",
    )


@tool("get_examples", args_schema=GetExamplesInput)
def get_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
) -> list[dict]:
    """Fetch N sample rows from the dataset, each containing the customer
    instruction, agent response, category, and intent.
    Use this when the user asks to 'show', 'give', or 'display' examples.
    Sampling is random, so calling it again returns fresh rows — which is what
    you want for follow-ups like 'show me 3 more'.
    Example:
        get_examples(n=3, category='SHIPPING_ADDRESS')
    """
    results = get_examples_impl(n, category, intent)
    if not results:
        return [{"error": "No rows found for the given filters."}]
    return results


# ---------------------------------------------------------------------------
# Qualitative summary
# ---------------------------------------------------------------------------


class SummarizeTextInput(BaseModel):
    field: Literal["instruction", "response"] = Field(
        description="Which text column to summarise: "
        "'instruction' for customer queries, 'response' for agent replies.",
    )
    category: str | None = Field(
        default=None,
        description="Restrict summarisation to this category. Optional.",
    )
    intent: str | None = Field(
        default=None,
        description="Restrict summarisation to this intent. Optional.",
    )
    sample_size: int = Field(
        default=40,
        ge=5,
        le=100,
        description="How many rows to sample before summarising (5-100).",
    )


@tool("summarize_text", args_schema=SummarizeTextInput)
def summarize_text(
    field: Literal["instruction", "response"],
    category: str | None = None,
    intent: str | None = None,
    sample_size: int = 40,
) -> str:
    """Sample rows matching the filters and produce a qualitative LLM summary
    of either the customer instructions or the agent responses.
    Use this for open-ended 'summarise', 'describe', 'what patterns', 'what
    problems', or 'how do agents respond' questions. Always apply at least one
    filter (category or intent); never summarise the whole dataset at once.
    For a comparison, call this once per side and contrast the results yourself.
    Example:
        summarize_text(field='response', category='FEEDBACK')
    """
    from agent.llm import get_llm
    from data.loader import load_bitext

    df = load_bitext()
    if category:
        df = df[df["category"] == category.upper()]
    if intent:
        df = df[df["intent"] == intent.lower()]

    series = df[field].dropna()
    if series.empty:
        return "No data found for the given filters."

    # Deterministic sample (fixed seed) so a given summary is reproducible.
    sample = series.sample(
        min(sample_size, len(series)), random_state=ANALYSIS_SEED
    ).tolist()
    joined = "\n---\n".join(sample)

    llm = get_llm(temperature=0)
    prompt = (
        f"You are analysing a customer service dataset.\n"
        f"Below are {len(sample)} sampled '{field}' texts"
        f"{' for category ' + category if category else ''}"
        f"{' and intent ' + intent if intent else ''}.\n\n"
        f"Write a concise summary (3-5 sentences) describing the key themes, "
        f"tone, and patterns you observe.\n\n"
        f"--- TEXTS ---\n{joined}"
    )
    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


class SemanticSearchInput(BaseModel):
    query: str = Field(
        description="Natural-language phrase to search for in customer instructions "
        "(e.g. 'wanting money back', 'cannot log into my account').",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of most-similar rows to return (1-20).",
    )


@tool("semantic_search", args_schema=SemanticSearchInput)
def semantic_search(query: str, n: int = 5) -> list[dict]:
    """Search for dataset rows whose customer instruction is most semantically
    similar to the given natural-language query using TF-IDF cosine similarity.
    Use this when the user's phrasing does not match exact category/intent names
    (e.g. 'people wanting their money back', 'can't access my account'). Never
    guess an intent name — search first, then read the intent from the results.
    """
    results = semantic_search_impl(query, n)
    if not results:
        return [{"error": "No similar rows found."}]
    return results


# ---------------------------------------------------------------------------
# Text length statistics
# ---------------------------------------------------------------------------


class ComputeTextStatsInput(BaseModel):
    field: Literal["instruction", "response"] = Field(
        description="Which text column to measure: 'instruction' for customer messages, 'response' for agent replies.",
    )
    group_by: Literal["category", "intent"] | None = Field(
        default=None,
        description="Break the statistics down per 'category' or per 'intent'. Leave empty for a single dataset-wide summary.",
    )
    unit: Literal["words", "characters"] = Field(
        default="words",
        description="Measure length in 'words' or 'characters'.",
    )


@tool("compute_text_stats", args_schema=ComputeTextStatsInput)
def compute_text_stats(
    field: Literal["instruction", "response"],
    group_by: Literal["category", "intent"] | None = None,
    unit: str = "words",
) -> dict:
    """Measure how LONG the text is in the dataset (mean, median, min, max length).
    Use this for any question about text length or size — e.g. 'which category
    has the longest responses?', 'how long is the average customer message?'.
    With group_by set, the result is sorted longest-first, so the first entry is
    the longest and the last entry is the shortest.
    Example:
        compute_text_stats(field='response', group_by='category')
    """
    return compute_text_stats_impl(field, group_by, unit)


# ---------------------------------------------------------------------------
# Language-generation flags
# ---------------------------------------------------------------------------


class InspectFlagsInput(BaseModel):
    flag: str | None = Field(
        default=None,
        description="A single flag code (e.g. 'Q' colloquial, 'W' offensive "
        "language, 'Z' typos, 'P' politeness). Leave empty to list every flag "
        "code with its meaning and how many rows carry it.",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of example rows to return when a flag is given (1-20).",
    )


@tool("inspect_flags", args_schema=InspectFlagsInput)
def inspect_flags(flag: str | None = None, n: int = 5) -> dict:
    """Explore the dataset's `flags` column: linguistic tags Bitext attached to
    each row (e.g. colloquial, offensive language, typos, politeness, negation).
    Call with NO argument to get the full flag legend and per-flag row counts:
    use this for 'what flagged samples exist?' or 'what do the flags mean?'.
    Call with a flag code to retrieve example rows carrying that flag.
    Example:
        inspect_flags()          -> legend + counts for every flag
        inspect_flags(flag='W')  -> example rows containing offensive language
    """
    return inspect_flags_impl(flag, n)


# Exported list used by the agent node and ToolNode
TOOLS = [
    get_categories,
    get_intents,
    count_rows,
    get_distribution,
    get_examples,
    summarize_text,
    semantic_search,
    compute_text_stats,
    inspect_flags,
]
