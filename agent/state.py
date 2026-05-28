"""
agent/state.py

Defines the shared state that flows through the LangGraph graph.
Keeping it in its own file prevents circular imports.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared across all nodes in the graph.

    Attributes:
        messages:        Full conversation history (human + AI + tool messages).
                         The add_messages reducer appends new messages instead
                         of overwriting.
        query_type:      Classification set by the router node:
                         'structured' | 'unstructured' | 'conversational' | 'out_of_scope'.
        iteration_count: How many times the agent node has run this turn.
                         Guards against infinite tool-call loops.
        user_profile:    Distilled, human-readable facts about the user, loaded
                         from disk by the load_profile node at the start of each
                         turn and injected into the agent's system prompt.
        pending_recommendation: Current suggested query, "" if none.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query_type: str
    iteration_count: int
    user_profile: str
    pending_recommendation: str
