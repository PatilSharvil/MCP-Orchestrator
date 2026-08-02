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
  1. `discover_tools(query: string)`: Dynamic candidate retrieval returning only the top-K relevant tool descriptions for the user's specific prompt.
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

## 2. Architecture & Design Principles

### Dual Role Architecture
- **MCP Server Side**: Exposes an HTTP/SSE endpoint on `http://localhost:8000/sse` using `FastMCP`.
- **MCP Client Side**: Spawns and manages sub-processes for downstream `stdio` MCP servers (e.g. Node.js `npx` servers) using the standard Python `mcp` SDK client sessions.

### Relational Storage Schema (PostgreSQL)
The relational schema tracks server topologies, tool schemas, query logs, retrieval benchmarks, and evaluation runs:

```sql
-- 1. Downstream Server Definitions
CREATE TABLE mcp_servers (
    id VARCHAR(64) PRIMARY KEY,          -- e.g., 'filesystem-server'
    name VARCHAR(255) NOT NULL,          -- Human-readable name
    transport_type VARCHAR(32) NOT NULL, -- 'stdio' or 'sse'
    connection_config JSONB NOT NULL,    -- Command, args, environment variables
    status VARCHAR(32) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Registered Tool Metadata
CREATE TABLE mcp_tools (
    id VARCHAR(128) PRIMARY KEY,         -- Format: 'server_id:tool_name'
    server_id VARCHAR(64) REFERENCES mcp_servers(id) ON DELETE CASCADE,
    name VARCHAR(128) NOT NULL,          -- Tool function name (e.g., 'read_file')
    description TEXT,                    -- Full tool description
    input_schema JSONB NOT NULL,         -- Tool input JSON Schema specification
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_server_tool UNIQUE (server_id, name)
);

-- 3. Query Execution Logs
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    discovered_tool_ids JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Retrieval & Ranking Metrics (Phase 2+)
CREATE TABLE retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    query_log_id BIGINT REFERENCES query_logs(id) ON DELETE CASCADE,
    retrieved_tool_ids JSONB,
    scores JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 5. System Evaluation Benchmark Runs
CREATE TABLE eval_runs (
    id BIGSERIAL PRIMARY KEY,
    run_name VARCHAR(255) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. Directory Structure

```
MCP Orchestrator/
├── orchestrator/
│   ├── __init__.py
│   ├── config.py             # Server registry configuration & environment settings
│   ├── db.py                 # Async SQLAlchemy PostgreSQL engine & table definitions
│   ├── client.py             # Client session manager for downstream stdio MCP servers
│   ├── registrar.py          # Idempotent tool discovery & PostgreSQL upsert
│   └── server.py             # FastMCP HTTP/SSE server running on port 8000
├── retrieval/                # Vector embedding & retrieval logic (Phase 2)
│   └── __init__.py
├── ranking/                  # Reranking / cross-encoder module (Phase 2)
│   └── __init__.py
├── eval/                     # Evaluation benchmark runner
│   └── __init__.py
├── dashboard/                # Analytics & Web UI
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_registration.py  # Pytest suite for client discovery, DB, and meta-tools
├── docker-compose.yml        # Docker compose stack (Postgres: 5432, ChromaDB: 8001, Orchestrator: 8000)
├── Dockerfile                # Multi-stage container (Python 3.11 + Node.js runtime)
├── pyproject.toml            # Project package configuration and dependencies
├── seed_registration.py      # Entrypoint script to run tool registration
└── README.md                 # Project documentation & viva preparation guide
```

---

## 4. Quickstart Guide

### Prerequisites
- **Python 3.10+** installed
- **Node.js / npm** (required to run `npx` stdio servers like `@modelcontextprotocol/server-filesystem`)
- **Docker Desktop** (for running PostgreSQL and ChromaDB)

### Step 1: Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/PatilSharvil/MCP-Orchestrator.git
cd "MCP Orchestrator"

# Install dependencies using editable mode
pip install -e .[dev]
```

### Step 2: Start Infrastructure Stack with Docker Compose
```bash
# Start PostgreSQL (5432) and ChromaDB (8001)
docker compose up -d postgres chromadb
```

### Step 3: Run Downstream Server Registration
Execute the idempotent registration script to discover tools from the downstream filesystem server and populate PostgreSQL:
```bash
python seed_registration.py
```
*Output:*
```text
=== Starting MCP Orchestrator Downstream Server Registration ===
2026-08-02 [INFO] orchestrator.client: Discovered tools from 'filesystem-server'
=== Registration Completed Successfully ===
Servers Registered / Processed: 1
Tools Registered / Processed: 14
```

### Step 4: Run the MCP Orchestrator Server
Start the FastMCP HTTP/SSE gateway server on `0.0.0.0:8000`:
```bash
python -m orchestrator.server
```

### Step 5: Execute Test Suite
Run automated unit and integration tests using `pytest`:
```bash
pytest
```
## 6. License
MIT License
