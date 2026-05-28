"""
mcp_server.py

FastMCP server that exposes eight dataset-analysis tools as MCP endpoints
(get_categories, get_intents, count_rows, get_examples, get_distribution,
semantic_search, compute_text_stats, inspect_flags). This lets any
MCP-compatible client call these tools directly without going through the 
full LangGraph agent.

Usage:
    python mcp_server.py           # starts the server (stdio transport by default)

Connect a client (stdio example):
    fastmcp run mcp_server.py

The README shows how to call a tool from a Python MCP client.
"""

from __future__ import annotations

from dotenv import load_dotenv
from fastmcp import FastMCP

from data.loader import (
    compute_text_stats_impl,
    count_rows_impl,
    get_categories_impl,
    get_distribution_impl,
    get_examples_impl,
    get_intents_impl,
    inspect_flags_impl,
    semantic_search_impl,
)

load_dotenv()

mcp = FastMCP(
    name="customer-service-analyst",
    instructions=(
        "Tools for analysing the Bitext Customer Service dataset. "
    ),
)


@mcp.tool()
def get_categories() -> list[str]:
    """Return all unique top-level categories in the Bitext dataset
    (e.g. ORDER, ACCOUNT, REFUND, FEEDBACK, SHIPPING).
    """
    return get_categories_impl()


@mcp.tool()
def get_intents(category: str | None = None) -> list[str]:
    """Return all unique intents, optionally filtered by category.

    Args:
        category: Category name to filter by (e.g. 'ORDER'). Optional.
    """
    return get_intents_impl(category)


@mcp.tool()
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count rows matching the optional category and/or intent filters.

    Args:
        category: Filter by category name (e.g. 'REFUND'). Optional.
        intent:   Filter by intent name (e.g. 'get_refund'). Optional.
    """
    return count_rows_impl(category, intent)


@mcp.tool()
def get_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
) -> list[dict]:
    """Fetch N sample rows from the dataset (instruction, response, category, intent).

    Args:
        n:        Number of examples to return (1-20).
        category: Filter by category. Optional.
        intent:   Filter by intent. Optional.
    """
    n = max(1, min(n, 20))
    results = get_examples_impl(n, category, intent)
    return results if results else [{"error": "No rows found for the given filters."}]


@mcp.tool()
def get_distribution(category: str) -> dict:
    """Return the count of each intent within a given category.

    Args:
        category: The category to analyse (e.g. 'ACCOUNT').
    """
    result = get_distribution_impl(category)
    return result if result else {"error": f"Category '{category}' not found."}


@mcp.tool()
def semantic_search(query: str, n: int = 5) -> list[dict]:
    """Find rows whose customer instruction is most similar to the query.

    Args:
        query: Natural language phrase (e.g. 'wanting money back').
        n:     Number of results to return (1-20).
    """
    n = max(1, min(n, 20))
    results = semantic_search_impl(query, n)
    return results if results else [{"error": "No similar rows found."}]


@mcp.tool()
def compute_text_stats(
    field: str,
    group_by: str | None = None,
    unit: str = "words",
) -> dict:
    """Measure text length statistics (mean, median, min, max) for a column.

    Args:
        field:    'instruction' or 'response'.
        group_by: 'category' or 'intent' for a longest-first ranking; omit for overall.
        unit:     'words' or 'characters'.
    """
    return compute_text_stats_impl(field, group_by, unit)


@mcp.tool()
def inspect_flags(flag: str | None = None, n: int = 5) -> dict:
    """Explore the dataset's linguistic `flags` column.

    Args:
        flag: A single flag code (e.g. 'Q', 'W', 'Z'). Omit to list every flag
              code with its meaning and per-flag row counts.
        n:    Number of example rows to return when a flag is given (1-20).
    """
    n = max(1, min(n, 20))
    return inspect_flags_impl(flag, n)


if __name__ == "__main__":
    mcp.run()