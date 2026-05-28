"""
agent/profile.py

Persistent per-user profile storage (Task 2b).

The profile is a small list of distilled facts about the user, kept SEPARATELY
from the conversation checkpoint. It is read by the load_profile node and
rewritten by the update_profile node — no LLM logic lives here, just storage.

Storage format (profiles/{user_id}.json):
    {"facts": ["User's name is Sapir", "Frequently asks about refunds", ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

PROFILES_DIR = Path("profiles")


def _profile_path(user_id: str) -> Path:
    """Return the file path for a given user's profile."""
    PROFILES_DIR.mkdir(exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    return PROFILES_DIR / f"{safe_id}.json"


def load_facts(user_id: str) -> list[str]:
    """Load stored profile facts for a user. Returns an empty list if none exist."""
    path = _profile_path(user_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        facts = data.get("facts", [])
        return [str(f) for f in facts if str(f).strip()]
    except (json.JSONDecodeError, OSError):
        return []


def save_facts(user_id: str, facts: list[str]) -> None:
    """Persist the facts list to disk for a user."""
    path = _profile_path(user_id)
    path.write_text(json.dumps({"facts": facts}, indent=2), encoding="utf-8")


def format_facts(facts: list[str]) -> str:
    """Render the facts list as a human-readable block for prompt injection."""
    if not facts:
        return "No stored facts about this user yet."
    return "\n".join(f"- {fact}" for fact in facts)


def list_user_ids() -> list[str]:
    """List all user IDs that have a stored profile, sorted alphabetically.

    Reads the `profiles/` directory and returns the stem of every `*.json`
    file. Used by the Streamlit sidebar to populate the "User ID" dropdown.
    """
    if not PROFILES_DIR.exists():
        return []
    return sorted(f.stem for f in PROFILES_DIR.glob("*.json"))