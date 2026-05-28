"""
agent/runner.py

Shared agent-interaction layer used by BOTH front-ends (CLI and Streamlit).

It owns everything presentation-agnostic:
  - building (and caching) the compiled graph,
  - building the run config,
  - streaming a turn through the graph,
  - parsing raw LangGraph events into structured ``Step`` objects,
  - rebuilding past turns from the SQLite checkpoint.

Front-ends import from here and ONLY render ``Step`` / ``Turn`` objects — they
never touch LangGraph message internals, so the two UIs cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.graph import build_graph


# ---------------------------------------------------------------------------
# Presentation-agnostic data types
# ---------------------------------------------------------------------------

StepKind = Literal["router", "thought", "tool_call", "tool_result", "answer"]


@dataclass
class Step:
    """One event from a single agent turn, ready for any front-end to render.

    kind - what this step is; the front-end picks an icon / colour / placement.
    text - the canonical, display-ready rendering (formatted once, here).

    A turn streams as: one ``router`` step, then zero or more
    'thought' / 'tool_call' / 'tool_result' steps, then one 'answer'.
    """

    kind: StepKind
    text: str


@dataclass
class Turn:
    """A completed exchange - used to re-display a resumed conversation."""

    user: str            # the user's message
    steps: list[Step]    # reasoning steps (everything except the answer)
    answer: str          # the final answer text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TOOL_RESULT_PREVIEW = 400  # max characters shown for a tool result


@lru_cache(maxsize=4)
def _get_graph(db_path: str):
    """Build the compiled graph once per 'db_path' and cache it.
    lru_cache keeps the graph alive for the whole process, so the Streamlit
    app (which re-runs its script on every interaction) does not rebuild it.
    """
    return build_graph(db_path=db_path)


def _format_tool_call(tool_call: dict) -> str:
    """Render a tool call as 'name(arg=value, ...)'.
    For example: 
    """
    args = ", ".join(f"{k}={v!r}" for k, v in (tool_call.get("args") or {}).items())
    return f"{tool_call.get('name', '?')}({args})"


def _truncate(text: object, limit: int = _TOOL_RESULT_PREVIEW) -> str:
    """Trim long text (e.g. a big tool result) for display."""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


def _steps_from_messages(messages: list) -> Iterator[Step]:
    """Convert LangGraph messages into reasoning / answer 'Step's.
    This is the single shared parser — both 'run_turn' (live streaming) and
    'load_history' (replaying the checkpoint) go through it, so the two
    front-ends can never interpret the graph differently.
    """
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            content = str(msg.content).strip() if msg.content else ""
            if content and tool_calls:  # the model reasoned and decided to act
                yield Step("thought", content)
            for tool_call in tool_calls: # tool calls - one at a time
                yield Step("tool_call", _format_tool_call(tool_call))
            if content and not tool_calls: # the model produced its final reply
                yield Step("answer", content)
        elif isinstance(msg, ToolMessage): # a tool result (which may be long, so we truncate it)
            yield Step("tool_result", _truncate(msg.content))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_config(session_id: str, user_id: str | None = None) -> RunnableConfig:
    """Build the LangGraph config for a turn.
    session_id - thread_id  (conversation checkpoint key)
    user_id    - user_id    (profile key; defaults to session_id)
    """
    return {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id or session_id,
        }
    }


def run_turn(graph, config: RunnableConfig, user_text: str) -> Iterator[Step]:
    """ Feeds the user's message ('turn') into the graph and streams events as 'Step' objects.
    Wraps 'graph.stream(..., stream_mode="updates")' and converts each raw
    graph event into zero or more 'Step's. All event parsing lives here:
    skipping no-op nodes (which stream as None), telling a thought from
    a final answer, and formatting tool calls.

    Yields the reasoning steps live, then one 'Step(kind="answer")'.
    """
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
        stream_mode="updates", # stream updates as they happen, not just the final state
    ):
        for node_name, node_output in event.items():
            if not node_output:  # e.g. update_profile makes no state change
                continue
            if "query_type" in node_output:
                yield Step("router", str(node_output["query_type"]))
            yield from _steps_from_messages(node_output.get("messages", []))


def load_history(graph, config: RunnableConfig) -> list[Turn]:
    """Rebuild the past turns of a session from the SQLite checkpoint.
    Lets a resumed conversation be re-displayed. Returns empty list 
    for a new session or if the checkpoint cannot be read.
    Every HumanMessage begins a new turn, and the messages after it 
    (until the next human message) are that turn's steps and answer.
    """
    try:
        snapshot = graph.get_state(config)
    except Exception:
        return []

    messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    turns: list[Turn] = []
    current: Turn | None = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if current is not None:
                turns.append(current)
            current = Turn(user=str(msg.content), steps=[], answer="")
        elif current is None:
            continue  # skip anything before the first user message
        else:
            for step in _steps_from_messages([msg]):
                if step.kind == "answer":
                    current.answer = step.text
                else:
                    current.steps.append(step)

    if current is not None:
        turns.append(current)
    return turns


# ---------------------------------------------------------------------------
# AgentSession wrapper
# ---------------------------------------------------------------------------

class AgentSession:
    """A compiled graph + config bound to one (session, user).

    Lets a front-end hold a single object instead of juggling graph and
    config. The graph itself is shared and cached (see _get_graph), so
    creating an AgentSession per Streamlit rerun is cheap.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        db_path: str = "memory.db",
    ) -> None:
        self.graph = _get_graph(db_path)
        self.config = make_config(session_id, user_id)

    def ask(self, user_text: str) -> Iterator[Step]:
        """Run one turn; stream 'Step's as they happen."""
        yield from run_turn(self.graph, self.config, user_text)

    def history(self) -> list[Turn]:
        """Return the past turns of this session, from the checkpoint."""
        return load_history(self.graph, self.config)


# ---------------------------------------------------------------------------
# Session discovery (used by the Streamlit sidebar)
# ---------------------------------------------------------------------------

def list_session_ids(db_path: str = "memory.db") -> list[str]:
    """List distinct session / thread IDs stored in the SQLite checkpoint.

    Reads the SqliteSaver's `checkpoints` table directly. Returns [] if the
    database file doesn't exist yet, or if the table isn't there (e.g. on a
    very first launch before any conversation has been saved).
    """
    import sqlite3
    from pathlib import Path
    if not Path(db_path).exists():
        return []
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return []