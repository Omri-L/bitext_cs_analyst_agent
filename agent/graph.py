"""
agent/graph.py

Assembles the LangGraph ReAct graph and compiles it with a persistent SQLite
checkpointer so conversation history survives restarts (Task 2a).

Graph topology:
    START
      |
    load_profile        (loads the persistent user profile into state)
      |
    router              (classifies query type)
      |
      +- out_of_scope --> decline --------------------> END
      |
      +- structured / unstructured / conversational
            |
          agent         (ReAct step: reason + select tool or finish)
            |
            +- has tool calls --> tools --> agent  (loop)
            |
            +- final answer ----> update_profile --> END
                                  (distils new facts about the user)
"""

from __future__ import annotations

import sqlite3
from typing import Literal

from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import (
    MAX_ITERATIONS,
    agent_node,
    decline_node,
    load_profile_node,
    recommender_node,
    router_node,
    update_profile_node,
)
from agent.state import AgentState
from agent.tools import TOOLS


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_router(state: AgentState) -> Literal["agent", "recommender", "decline"]:
    """Route based on query classification.
    out_of_scope → decline, recommendation → recommender, everything else → agent.
    """
    if state["query_type"] == "out_of_scope":
        return "decline"
    if state["query_type"] == "recommendation":
        return "recommender"
    return "agent"


def route_after_recommender(state: AgentState) -> Literal["agent", "__end__"]:
    """Execute the confirmed query if the recommender injected a HumanMessage;
    otherwise wait for the user's next message.

    The recommender signals execution by appending a HumanMessage (the confirmed
    query) as the last message. For suggest/refine/cancel it appends an AIMessage.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "agent"
    return "__end__"


def route_after_agent(state: AgentState) -> Literal["tools", "finish"]:
    """Loop to the tools node if the agent made tool calls and the iteration
    budget is not spent; otherwise finish via the update_profile node."""
    last_message = state["messages"][-1]
    if state["iteration_count"] < MAX_ITERATIONS and getattr(
        last_message, "tool_calls", None
    ):
        return "tools"
    return "finish"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(db_path: str = "memory.db"):
    """Build and compile the agent graph with a SQLite persistence checkpointer.

    Args:
        db_path: Path to the SQLite file used for conversation checkpoints.
                 Use ':memory:' for a non-persistent in-memory store (testing).

    Returns:
        Compiled LangGraph CompiledStateGraph ready for .invoke() / .stream().
    """
    workflow = StateGraph(AgentState)

    # --- Nodes ---
    workflow.add_node("load_profile", load_profile_node)
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("decline", decline_node)
    workflow.add_node("update_profile", update_profile_node)
    workflow.add_node("recommender", recommender_node)   # ← ADD

    # --- Edges ---
    workflow.add_edge(START, "load_profile")
    workflow.add_edge("load_profile", "router")

    workflow.add_conditional_edges(
            "router",
            route_after_router,
            {"agent": "agent", "recommender": "recommender", "decline": "decline"},
    )

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finish": "update_profile"},
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("decline", END)
    workflow.add_edge("update_profile", END)

    workflow.add_conditional_edges(
        "recommender",
        route_after_recommender,
        {"agent": "agent", "__end__": END},
    )

    # --- Checkpointer (persistent conversation memory) ---
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)
