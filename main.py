"""
main.py

Interactive CLI for the Customer Service Data Analyst Agent.

All agent logic lives in agent/runner.py (shared with the Streamlit app);
this file is only the terminal front-end: an input loop plus render_cli().

Usage:
    python main.py                                  # default session
    python main.py --session alice                  # named, resumable session
    python main.py --session alice --user alice      # explicit profile key
    python main.py --session alice --db my.db        # custom checkpoint DB
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv(override=True)

from agent.runner import AgentSession, Step # runner pulls in the graph module (which loads the dataset)
console = Console()


# ---------------------------------------------------------------------------
# Rendering — the only display logic the CLI needs
# ---------------------------------------------------------------------------

# Step.kind -> (icon, label, rich style)
_STYLE: dict[str, tuple[str, str, str]] = {
    "thought":     ("💭", "thought:",      "magenta"),
    "tool_call":   ("🔧", "tool_use:",     "bold green"),
    "tool_result": ("🚀", "tool_result:",  "yellow"),
    "answer":      ("🤖", "Agent Answer:", "bold cyan"),
}

# Router line is coloured by the classified query type.
_ROUTER_COLOURS = {
    "structured": "blue",
    "unstructured": "magenta",
    "conversational": "green",
    "out_of_scope": "red",
}


def render_cli(step: Step) -> None:
    """Render one Step to the terminal. The runner already produced a
    display-ready ``step.text``, so this only picks an icon + colour."""
    if step.kind == "router":
        colour = _ROUTER_COLOURS.get(step.text, "white")
        console.print(f"  [bold {colour}]🔀 Router → {step.text}[/bold {colour}]")
        return
    icon, label, style = _STYLE.get(step.kind, ("•", step.kind, "white"))
    console.print(f"  [{style}]{icon} {label}[/{style}] {step.text}")
    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session", default="default",
        help="Session ID - the same ID restores the conversation after restart.",
    )
    parser.add_argument(
        "--user", default=None,
        help="User ID for the persistent profile (defaults to the session ID).",
    )
    parser.add_argument(
        "--db", default="memory.db",
        help="Path to the SQLite checkpoint file (default: memory.db).",
    )
    return parser.parse_args()


def print_welcome(session_id: str, user_id: str) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]Customer Service Data Analyst Agent 🤗[/bold cyan]\n"
            f"Session: [yellow]{session_id}[/yellow]  |  "
            f"User: [yellow]{user_id}[/yellow]  |  "
            f"Type [bold]exit[/bold] or [bold]quit[/bold] to leave.",
            border_style="cyan",
        )
    )
    console.print()


def main() -> None:
    """Main input loop: create an AgentSession, print a welcome message, then
    repeatedly read user input and render the agent's streamed response."""
    
    args = parse_args()
    session = AgentSession(args.session, args.user, args.db)
    print_welcome(args.session, args.user or args.session)

    while True:
        try:
            console.print(Rule(style="dim"))
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye! 👋[/dim]")
            sys.exit(0)

        console.print()
        try:
            for step in session.ask(user_input):
                render_cli(step)
        except Exception as exc:  # keep the loop alive on any runtime error
            console.print(f"[bold red]🚨 Error:[/bold red] {exc}")
        console.print()


if __name__ == "__main__":
    main()
