"""
streamlit_app.py

Streamlit chat UI for the Customer Service Data Analyst Agent (Bonus A).

All agent logic lives in agent/runner.py (shared with the CLI); this file is
only the web front-end: session-state plumbing plus render functions.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from agent.profile import list_user_ids
from agent.runner import AgentSession, Step, Turn, list_session_ids

# Step.kind -> icon (the answer step is rendered separately, as the main bubble)
_ICON = {
    "router": "🔀",
    "thought": "💭",
    "tool_call": "🔧",
    "tool_result": "🚀",
}


# ---------------------------------------------------------------------------
# Rendering - the only display logic the Streamlit app needs
# ---------------------------------------------------------------------------

def render_reasoning(steps: list[Step], *, expanded: bool) -> None:
    """Draw the reasoning steps inside a collapsible expander."""
    if not steps:
        return
    with st.expander("🧠 Reasoning steps", expanded=expanded):
        for step in steps:
            st.markdown(f"- {_ICON.get(step.kind, '•')} **{step.kind}** — {step.text}")


def render_turn(turn: Turn) -> None:
    """Draw one past turn as user + assistant chat bubbles."""
    with st.chat_message("user"):
        st.markdown(turn.user)
    with st.chat_message("assistant"):
        render_reasoning(turn.steps, expanded=False)
        st.markdown(turn.answer or "_(no final answer)_")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CS Data Analyst Agent", page_icon="🤖")
st.title("🤖 Customer Service Data Analyst Agent")

# --- Sentinels for the sidebar selectboxes ---
NEW_SESSION = "+ New session..."
NEW_USER = "+ New user..."
SAME_AS_SESSION = "(same as session)"


# --- Sidebar: session controls ---
with st.sidebar:
    st.header("Session")

    # --- Session ID: dropdown of existing thread_ids + a "new" option ---
    # Always include the currently active session in the list, even before
    # its first checkpoint is written, so the selection sticks across reruns.
    if "active_session" not in st.session_state:
        st.session_state.active_session = "default"
    existing_sessions = set(list_session_ids())
    existing_sessions.add(st.session_state.active_session)
    session_options = sorted(existing_sessions) + [NEW_SESSION]
    default_idx = session_options.index(st.session_state.active_session)

    choice = st.selectbox(
        "Session ID",
        session_options,
        index=default_idx,
        help="Pick an existing conversation to resume, or '+ New session...' to start a new one.",
    )
    if choice == NEW_SESSION:
        session_id = (
            st.text_input("New session ID", value="default").strip() or "default"
        )
    else:
        session_id = choice
    st.session_state.active_session = session_id

    # --- User ID: dropdown of existing profile files + a "new" option ---
    existing_users = sorted(list_user_ids())
    user_options = [SAME_AS_SESSION] + existing_users + [NEW_USER]
    user_choice = st.selectbox(
        "User ID (optional)",
        user_options,
        index=0,
        help="Profile key. Pick an existing user, '(same as session)' to reuse the session ID, or '+ New user...' to create one.",
    )
    if user_choice == NEW_USER:
        user_id = st.text_input("New user ID", value="").strip()
    elif user_choice == SAME_AS_SESSION:
        user_id = ""
    else:
        user_id = user_choice

    st.caption(
        "Conversations persist to SQLite - the same Session ID restores its "
        "history even after restarting the app."
    )

session = AgentSession(session_id, user_id or None)

# --- Load the transcript for this session (only when the session changes) ---
if st.session_state.get("loaded_session") != session_id:
    st.session_state.transcript = session.history()
    st.session_state.loaded_session = session_id

# --- Render the existing transcript ---
if not st.session_state.transcript:
    st.info("Ask a question about the Bitext Customer Service dataset to begin.")
for turn in st.session_state.transcript:
    render_turn(turn)

# --- Handle a new message ---
if prompt := st.chat_input("Ask about the dataset..."): # := assigns and tests in one go: enter the block only if there's a new message
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        steps: list[Step] = []
        answer = ""
        try:
            with st.spinner("Thinking..."): # a spinner while the agent works
                for step in session.ask(prompt):
                    if step.kind == "answer":
                        answer = step.text
                    else:
                        steps.append(step)
        except Exception as exc:
            st.error(f"Error: {exc}")
        else:
            render_reasoning(steps, expanded=True)
            st.markdown(answer or "_(no final answer)_")
            st.session_state.transcript.append(
                Turn(user=prompt, steps=steps, answer=answer)
            )