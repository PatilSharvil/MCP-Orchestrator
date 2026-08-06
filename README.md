# MCP Orchestrator: Dynamic Tool Retrieval & Gateway Middleware

**MCP Orchestrator** is a high-performance middleware architecture sitting between AI coding agents (such as Claude Code, Cursor, or custom LLM agents) and downstream Model Context Protocol (MCP) servers.

---

## 1. Project Overview & Motivation

### The Problem: Context Window Bloat in Large MCP Ecosystems
Standard AI coding agents load all tool definitions from every connected downstream MCP server directly into the LLM context window upfront. As the number of connected MCP servers grows (e.g., GitHub, Filesystem, Jira, Postgres, Slack, AWS), loading hundreds of verbose JSON Schema tool definitions leads to:
1. **Context Window Exhaustion**: Thousands of tokens consumed solely by tool schemas.
2. **Degraded LLM Accuracy**: "Tool confusion" where the LLM struggles to select the correct tool among dozens of irrelevant options.
3. **High Latency & API Cost**: Increased prompt token count on every turn.

### The Solution: Meta-Tool Gateway Pattern
MCP Orchestrator acts as both an **MCP Client** and an **MCP Server**:
- **Downstream (MCP Client)**: Connects to real downstream MCP servers via `stdio` or `SSE` and indexes their tool schemas.
- **Upstream (MCP Server)**: Exposes a single, unified MCP endpoint to the coding agent with **only two meta-tools**:
  1. `discover_tools(query: string)`: Two-stage vector retrieval returning only the top-K relevant tool descriptions for the user's specific prompt.
  2. `invoke_tool(server_id: string, tool_id: string, arguments: object)`: Proxies execution directly to the appropriate downstream MCP server.

```
+-------------------+                 +-------------------+                 +--------------------------------+
|  AI Coding Agent  |  --- (SSE) ---> |  MCP Orchestrator |  --- (stdio) -> | Filesystem MCP Server          |
| (Claude / Cursor) |                 |    (Port 8000)    |                 | (@modelcontextprotocol/fs)     |
+-------------------+                 +-------------------+                 +--------------------------------+
                                                |                           +--------------------------------+
                                                +-------------- (stdio) --> | Downstream Server N (GitHub)   |
                                                |                           +--------------------------------+
                                                v
                                      +-------------------+
                                      | PostgreSQL (5432) |
                                      | ChromaDB   (8001) |
                                      +-------------------+
```

---

## 2. Dynamic Server Extensibility (`mcp_orc.json`)

Downstream MCP servers are configured externally via `mcp_orc.json` at the project root (similar to `claude_desktop_config.json` or Cursor's `mcp.json`). Adding or modifying downstream servers requires zero Python code changes.

### Configuration Format & Secret Interpolation
Secrets (such as API keys or access tokens) are **never hardcoded in JSON**. Use `${ENV_VAR}` syntax to reference environment variables stored in your local `.env` file (which is gitignored):

```json
{
  "mcpServers": {
    "filesystem-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {}
    },
    "memory-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {}
    },
    "github-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

If a required environment variable is not set, MCP Orchestrator logs a warning and gracefully skips that server during registration without crashing the rest of the pipeline.

---

## 3. Directory Structure

```
MCP Orchestrator/
├── orchestrator/
│   ├── __init__.py
│   ├── config.py             # External mcp_orc.json loader & Pydantic validation
│   ├── db.py                 # Async SQLAlchemy PostgreSQL engine & table definitions
│   ├── client.py             # Client session manager for downstream stdio MCP servers
│   ├── registrar.py          # Idempotent tool discovery & PostgreSQL/ChromaDB upsert
│   └── server.py             # FastMCP HTTP/SSE server running on port 8000
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py           # Sentence-transformers (all-MiniLM-L6-v2) & ChromaDB collections
│   └── retriever.py          # Two-stage vector retrieval engine (Stage 1 & Stage 2)
├── ranking/                  # Reranking engine (Phase 4)
│   └── __init__.py
├── eval/
│   ├── __init__.py
│   ├── ground_truth_queries.json # 20 hand-labeled evaluation queries
│   └── eval_retrieval.py     # Retrieval hit-rate evaluation runner
├── dashboard/                # Analytics & Web UI (Future phase)
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_config_loader.py # Tests for mcp_orc.json loading, validation, and interpolation
│   ├── test_embedding_pipeline.py # Tests for vector embedding and metadata tagging
│   ├── test_registration.py  # Tests for idempotent registration and Postgres tables
│   └── test_two_stage_retrieval.py # Tests for two-stage retrieval hit-rate
├── mcp_orc.json              # Active downstream MCP server configuration file
├── mcp_orc.example.json      # Configuration template file
├── docker-compose.yml        # Docker stack (Postgres: 5432, ChromaDB: 8001, Orchestrator: 8000)
├── Dockerfile                # Multi-stage container (Python 3.11 + Node.js runtime)
├── pyproject.toml            # Project package configuration and dependencies
├── seed_registration.py      # Entrypoint script to run tool registration
└── README.md                 # Project documentation & viva preparation guide
```

---

## 4. Quickstart Guide

### Step 1: Install Dependencies
```bash
pip install -e .[dev]
```

### Step 2: Configure Environment & MCP Servers
```bash
# Copy template config if mcp_orc.json does not exist
cp mcp_orc.example.json mcp_orc.json
```

### Step 3: Run Downstream Tool Registration
Execute the idempotent registration script to read `mcp_orc.json`, query downstream servers, and index tools into PostgreSQL and ChromaDB:
```bash
python seed_registration.py
```

### Step 4: Start Orchestrator Gateway Server
Start the FastMCP HTTP/SSE server on `0.0.0.0:8000`:
```bash
python -m orchestrator.server
```

### Step 5: Execute Test Suite
Run automated unit tests using `pytest`:
```bash
pytest
```

---

## 5. Academic Viva & Report Highlights

- **Extensibility via JSON Configuration**: Adding new downstream MCP tools requires zero code modifications—simply update `mcp_orc.json`.
- **Pydantic Schema Guardrails**: Invalid or malformed JSON server definitions are caught at startup with clear validation errors.
- **Secret Hygiene**: All environment variables use `${VAR}` interpolation; `.env` is gitignored to protect personal access tokens.
- **Resilient Registration**: Downstream servers missing required API keys are skipped gracefully without breaking active servers.
