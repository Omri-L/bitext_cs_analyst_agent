# Customer Service Data Analyst Agent

A **LangGraph ReAct agent** that answers natural-language questions about the
[Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).


## Background

The agent acts as a *data analyst* over a customer-support dataset. A user asks
questions in plain English and the agent answers them by reasoning step-by-step
and calling data tools. It handles five kinds of message:

- **Structured** - concrete, data-driven answers: counts, lists, distributions,
  examples (*"How many refund requests are there?"*, *"Show me 5 examples from the REFUND category"*).
- **Unstructured** - open-ended, qualitative summaries (*"Summarize the FEEDBACK category"*,
  *"How do agents respond to cancellations?"*).
- **Conversational** - personal/social/memory messages handled naturally
  without dataset tools (*"My name is Jhon"*, *"What do you remember about me?"*).
- **Query recommendation** - an interactive flow: the user asks *"What should I
  query next?"*, the agent suggests a follow-up based on the conversation
  history and the user profile, the user can refine it, and the agent only
  runs it once the user confirms.
- **Out-of-scope** - anything unrelated to the dataset (*"Who won the 2022 World Cup?"*).
  These are politely declined (the agent does not answer them from the model's general knowledge).

The agent is built as a LangGraph ReAct graph with **persistent memory** (conversation
history + a per-user profile), and its data tools are also exposed through a
**FastMCP server** so any MCP-compatible client can call them directly.

## The dataset

[**Bitext – Customer Service Tagged Training Dataset**](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
is a synthetic dataset of ~26,900 customer-support question/answer pairs. Each row has
five columns:

| Column | Description |
|---|---|
| `instruction` | a user request from the Customer Service domain |
| `response` | an example expected response from the virtual assistant |
| `category` | the high-level semantic category for the intent (e.g. `ORDER`, `ACCOUNT`, `REFUND`, `FEEDBACK`, `SHIPPING`) |
| `intent` |the intent corresponding to the user instruction (e.g. `get_refund`, `cancel_order`) |
| `flags` | linguistic generation tags (e.g. `Q` colloquial, `W` offensive, `Z` typos) |

The dataset is downloaded automatically from HuggingFace on first run and cached
in-process (no manual download needed).

## Setup

### 1. Prerequisites

- **Python 3.10+**
- A **Nebius Token Factory** API key (get one at [studio.nebius.com](https://studio.nebius.com)).

### 2. Clone, create a virtual environment, install dependencies

```bash
git clone https://github.com/Omri-L/bitext_cs_analyst_agent.git
cd bitext_cs_analyst_agent

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows (PowerShell / cmd)

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then open `.env` and set your key. The file contains:

```ini
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
```

Only `NEBIUS_API_KEY` must be filled in; the model and base URL already have
working defaults.

## Running the agent via CLI
In order to run the agent, execute:

```bash
python main.py
```

This drops you into an interactive conversation loop. The first message triggers
a one-time dataset download (a few seconds), then the agent is ready. Type `exit`,
`quit`, or `q` to leave.

### CLI options

| Option | Default | Purpose |
|---|---|---|
| `--session <id>` | `default` | Conversation thread ID. The **same ID restores the same conversation** after a restart. |
| `--user <id>` | = `--session` | Profile key. Different sessions can share one profile by passing the same `--user`. |
| `--db <path>` | `memory.db` | Path to the SQLite checkpoint file. |

```bash
python main.py --session session_a                        # named, resumable session
python main.py --session session_a --user Jhon            # explicit per-user profile
python main.py --session session_a --db my_checkpoints.db # custom checkpoint DB
```

### What you see

The CLI prints the agent's **reasoning steps**, not just the final answer: the
router decision, each tool call, each tool result, and the answer:

```
You: How many refund requests are there?
  🔀 Router → structured
  🔧 tool_use: get_intents(category='REFUND')
  🚀 tool_result: ["check_refund_policy", "get_refund", "track_refund"]
  🔧 tool_use: count_rows(intent='get_refund')
  🚀 tool_result: 1012
  🤖 Agent Answer: There are 1,012 refund requests (intent `get_refund`).
```

## Streamlit chat UI

In addition to the CLI, the agent has a browser chat front-end built with
[Streamlit](https://streamlit.io/).

### Launch

```bash
streamlit run streamlit_app.py
```

A browser tab opens with a chat interface. The first message triggers the
one-time dataset download (cached afterwards), then the agent is ready.

### Features

- **Session controls in the sidebar** - a `Session ID` and an optional
  `User ID` input. Type a new Session ID to start a fresh conversation, or
  reuse an ID to resume a previous one.
- **Reasoning steps in a collapsible expander** - every assistant turn shows
  its `🧠 Reasoning steps` (router decision, thoughts, tool calls, tool
  results) inside an `st.expander`; the final answer is rendered below.
- **Shared persistence with the CLI** - the Streamlit app uses the same
  `memory.db` SQLite checkpointer as `main.py`. A conversation started in the
  CLI can be resumed in the browser by entering its Session ID, and vice
  versa.

### Shared code with the CLI

Both front-ends consume `agent/runner.py`, which owns all LangGraph-event
parsing and produces presentation-agnostic `Step` / `Turn` objects. `main.py`
and `streamlit_app.py` only contain *rendering* code – the agent logic is
written once.


## Architecture

The agent is a **LangGraph ReAct graph** compiled with a SQLite checkpointer.

```
                  ┌──────────────┐
   START ───────▶│ load_profile │   load profiles/<user>.json into state
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │    router    │   LLM classifies into 5 query types
                  └──────┬───────┘
                         │
   ┌─────────────┬───────┴───────────┬
   │             │                   │                          
 out_of_scope  recommendation        │  structured /            
   │             │                   │  unstructured /          
   │             │                   │  conversational          
   ▼             ▼                   │                          
 ┌─────────┐ ┌──────────────┐        ▼                          
 │ decline │ │  recommender │─────▶┌──────────────┐◀───────┐    
 └────┬────┘ └──────┬───────┘ user │     agent    │        │    
      │   suggest/  │       confirm└──────┬───────┘        │    
      │    refine/  │   (inject query)    │ has tool calls │    
      │    cancel   ▼                     ▼                │    
      │            END                ┌──────────┐         │    
      │                               │   tools  │─────────┘    
      │                               └──────────┘              
      │                                     │ final answer      
      │                                     ▼                   
      │                            ┌────────────────┐           
      │                            │ update_profile │           
      │                            └────────┬───────┘           
      ▼                                     ▼                   
     END                                   END                  
```

### Nodes

| Node | Role |
|---|---|
| `load_profile` | Runs first every turn. Reads the persistent user profile and injects it into state. |
| `router` | A dedicated LLM classification node (see below). |
| `agent` | The ReAct reasoning step: reasons about the state and either calls tools or writes the final answer. |
| `tools` | Executes the tool calls the agent requested, then loops back to `agent`. |
| `decline` | Politely refuses out-of-scope queries: no tools, no general-knowledge answers. |
| `recommender` | Handles the interactive query-recommendation flow (suggest / refine / execute / cancel). Only **execute** injects a confirmed query for the `agent` to run. |
| `update_profile` | Runs after the final answer. An LLM pass distils durable facts about the user and saves them. |

### The router 🔀

Before any tool is selected, a dedicated `router` node makes a separate LLM call
that classifies the latest message into exactly one of:

- `structured` / `unstructured` → go to the `agent` with **strict tool-grounding**
  (the agent must answer only from tool results from the current turn).
- `conversational` → go to the `agent` in **conversational mode** (greetings,
  personal statements, "what do you remember about me?" - answered naturally
  without tools).
- `recommendation` → go to the `recommender` (interactive suggestion flow, see below).
- `out_of_scope` → go to `decline`.

The router only sees plain-text conversation history (tool-call fragments are
filtered out) so follow-ups like *"show me 3 more"* are classified correctly. If
the classification is malformed it safely defaults to `structured`.

### ReAct loop & safety limit

The `agent` ⇄ `tools` loop is the ReAct cycle: the agent reasons, calls a tool,
sees the result, and reasons again. A **maximum-iteration limit** (`MAX_ITERATIONS = 12`)
guards against infinite loops. If it is reached, the agent returns a graceful
fallback message instead of spinning forever.

### Tools

All tools have a clear name, a Pydantic input schema, and a typed return value.
The agent has **9 tools**:

| Tool | Purpose |
|---|---|
| `get_categories` | List all dataset categories. |
| `get_intents` | List intents, optionally filtered by category. |
| `count_rows` | Count rows matching category/intent filters ("how many"). |
| `get_distribution` | Intent breakdown within a category. |
| `get_examples` | Fetch N sample rows. |
| `summarize_text` | LLM-based qualitative summary of instructions or responses. |
| `semantic_search` | TF-IDF cosine-similarity search over customer instructions. |
| `compute_text_stats` | Text-length statistics (longest/shortest, averages), optionally ranked. |
| `inspect_flags` | Explore the linguistic `flags` column (legend, counts, examples). |

`semantic_search` uses scikit-learn TF-IDF (a local, non-LLM technique), so the
only LLM calls in the system go to Nebius Token Factory.

## Model choice

| Setting | Value |
|---|---|
| Provider | Nebius Token Factory (OpenAI-compatible API) |
| Model | `meta-llama/Llama-3.3-70B-Instruct` |
| Temperature | `0` (deterministic) |

A **single model** is used for all three LLM roles: router classification, the
ReAct agent (reasoning + tool calling), and profile fact-extraction. Reasons:

- **Strong instruction-following**: the router must return strict JSON and the
  agent must respect the tool-grounding rules; a 70B instruct model does this reliably.
- **Native tool/function-calling**: the ReAct loop depends on well-formed tool
  calls, which `Llama-3.3-70B-Instruct` supports well.
- **Capability/latency balance**: a 70B model is accurate enough for multi-step
  reasoning while staying responsive on Token Factory.

**Note**: The LLM factory (`agent/llm.py`) accepts a per-call `model` override, and the
model is set via the `NEBIUS_MODEL` environment variable, so a smaller model
could be assigned to the (simpler) routing role without code changes. We kept a
single model for consistent behaviour across all roles.

## Memory

### Conversation memory

Conversation state is persisted with LangGraph's **`SqliteSaver` checkpointer**,
which writes to `memory.db` (configurable via `--db`). Each turn's full state
(message history, etc.) is checkpointed under the `thread_id` set by `--session`.

Running the app again with the **same `--session` ID restores the exact
conversation**, even after the process is restarted. Because the whole message
history is replayed to the agent, follow-up questions that reference earlier
turns work naturally:

```
Show me 3 examples from the REFUND category   →  [examples]
Show me 3 more                                →  [3 fresh examples]
How many complaints are there?                →  [count]
What about refunds?                           →  [count]
What is the total of the last two?            →  [sum]
```

### User profile

The user profile is stored **separately** from the conversation history, as a
small list of distilled facts in `profiles/<user_id>.json`. It is maintained by
two dedicated graph nodes:

- **`load_profile`** runs at the start of every turn: it reads the profile file
  and injects the facts into the agent's system prompt, so the agent always
  "knows" the user without having to call a tool.
- **`update_profile`** runs after the final answer: an LLM pass extracts durable
  facts (the user's name, role, recurring interests, stated preferences), merges
  them with the existing profile, and writes the updated list back.

The profile is keyed by `--user` (which defaults to `--session`), so it persists
across restarts and can be shared between sessions. This is what lets the agent
answer *"What do you remember about me?"*

## Query recommendation

When the user asks something like *"What should I query next?"*, *"Give me a
suggestion"*, or *"What else can I look at?"*, the router classifies the message
as `recommendation` and routes to a dedicated **`recommender`** node instead of
the regular agent path.

The recommender combines the **conversation history** and the **user profile**
(so a user who has been asking about refunds gets refund-related suggestions),
then runs one LLM call that picks one of four actions:

| Action | What it means |
|---|---|
| `suggest` | Generate a fresh suggestion based on what the user has been exploring. The suggestion is shown but **not executed**: the reply ends with *"Should I go ahead?"*. |
| `refine` | The user pushed back (e.g. *"I'd rather see examples instead"*) - update the pending suggestion accordingly. Still not executed. |
| `execute` | The user confirmed (e.g. *"yes, do it"*) - the recommender appends the pending suggestion as a new `HumanMessage` and routes to the `agent`, which then runs it like any other data query. |
| `cancel` | The user declined with no concrete alternative (e.g. *"no thanks"*) - close the flow gracefully. |

The pending suggestion lives in a graph-state field, `pending_recommendation`,
so it persists across the multi-turn conversation. It's cleared automatically
whenever the user runs a regular data query (the suggestion is no longer
relevant once they've moved on).

### Example flow

```
You: What should I query next?
  🔀 Router → recommendation
  🤖 Based on your interest in refund data, you might want to see the
     distribution of intents in the REFUND category. Should I go ahead?

You: I'd rather see examples instead.
  🔀 Router → recommendation
  🤖 Then I'd suggest: show 5 examples from the REFUND category. Should I
     go ahead?

You: Yes, do it.
  🔀 Router → recommendation
  🔧 tool_use: get_examples(n=5, category='REFUND')
  🚀 tool_result: [ ... 5 example rows ... ]
  🤖 Agent Answer: Here are 5 examples from the REFUND category: ...
```

The user can also break out at any point with a concrete request: e.g.
*"No, give me 6 examples for SHIPPING instead"* - and the router will classify
that as a regular `structured` query, bypassing the recommender entirely.

The relevant code lives in two places: the `recommender_node` function in
`agent/nodes.py`, and the `RECOMMENDER_PROMPT` template in `agent/prompts.py`
that asks the LLM to choose the action and write the reply.

## MCP server

`mcp_server.py` is a [**FastMCP**](https://github.com/PrefectHQ/fastmcp) server that exposes **8 of the data tools** as
MCP endpoints: `get_categories`, `get_intents`, `count_rows`, `get_examples`,
`get_distribution`, `semantic_search`, `compute_text_stats`, `inspect_flags`.
This lets any MCP-compatible client (Claude Desktop, a custom script, the MCP
Inspector) call the tools directly, without going through the full agent.

### Start the server

```bash
python mcp_server.py            # runs over stdio
# or, equivalently:
fastmcp run mcp_server.py
```

To explore the tools interactively in a browser UI:

```bash
fastmcp dev mcp_server.py       # launches the MCP Inspector
```

### Connect a client and call a tool

A ready-to-run client is included:

```bash
python mcp_client_test.py
```

It connects to the server and calls two tools:

```python
import asyncio
from fastmcp import Client


async def main():
    # the client launches mcp_server.py as a subprocess and talks to it over stdio
    async with Client("mcp_server.py") as client:
        tools = await client.list_tools()
        print("available tools:", [t.name for t in tools])

        result = await client.call_tool("get_categories", {})
        print(result)

        result = await client.call_tool("count_rows", {"intent": "get_refund"})
        print(result)


asyncio.run(main())
```

## Project structure

```
bitext_cs_analyst_agent/
├── main.py              # interactive CLI entrypoint
├── streamlit_app.py     # Streamlit chat UI
├── mcp_server.py        # FastMCP server
├── mcp_client_test.py   # example MCP client
├── requirements.txt
├── .env.example
├── memory.db            # created on first run (SQLite conversation checkpoints)
├── profiles/            # created on first run (per-user profile JSON files)
├── data/
│   ├── __init__.py
│   └── loader.py        # dataset loading + raw data operations
└── agent/
    ├── __init__.py
    ├── state.py         # AgentState TypedDict (includes pending_recommendation)
    ├── llm.py           # Nebius Token Factory LLM factory
    ├── prompts.py       # router + agent + profile + recommender prompts
    ├── tools.py         # 9 LangChain tools with Pydantic schemas
    ├── profile.py       # per-user profile load/save
    ├── nodes.py         # load_profile, router, agent, decline, recommender, update_profile
    ├── runner.py        # shared agent-interaction layer used by CLI + Streamlit
    └── graph.py         # graph assembly + SQLite checkpointer
```
