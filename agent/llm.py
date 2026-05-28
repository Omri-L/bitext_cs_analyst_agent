"""
agent/llm.py

Factory for the Nebius Token Factory LLM client.
Nebius exposes an OpenAI-compatible API, so we use langchain-openai
with a custom base_url and api_key.

The constructed client is cached (keyed on model + temperature) so nodes that
call get_llm() repeatedly do not rebuild a new HTTP client on every step.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

_DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@lru_cache(maxsize=8)
def _build_llm(model: str, api_key: str, base_url: str, temperature: float) -> ChatOpenAI:
    """Build a ChatOpenAI client. Cached so identical configs are reused."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    """Return a (cached) ChatOpenAI client pointed at Nebius Token Factory.

    Args:
        temperature: Sampling temperature (0 = deterministic).
        model:       Optional model override. Defaults to the NEBIUS_MODEL env var.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        EnvironmentError: If NEBIUS_API_KEY is not set.
    """
    api_key = os.environ.get("NEBIUS_API_KEY")
    base_url = os.environ.get("NEBIUS_BASE_URL", _DEFAULT_BASE_URL)
    model = model or os.environ.get("NEBIUS_MODEL", _DEFAULT_MODEL)

    if not api_key:
        raise EnvironmentError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    return _build_llm(model, api_key, base_url, temperature)
