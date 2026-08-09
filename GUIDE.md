# 📖 MCP Orchestrator: Complete Beginner & Deployment Guide

Welcome to **MCP Orchestrator**! This comprehensive guide will walk you through setting up, configuring, and running MCP Orchestrator—even if you have zero prior experience with Docker or the Model Context Protocol (MCP).

---

## 📌 Table of Contents
1. [What is MCP Orchestrator?](#1-what-is-mcp-orchestrator)
2. [Configuration Guide (`mcp_orc.json`)](#2-configuration-guide-mcp_orcjson)
   - [Option A: Direct API Keys (Simplicity / Quick Start)](#option-a-direct-api-keys-simplicity--quick-start)
   - [Option B: `.env` Variables (Security / Recommended)](#option-b-env-variables-security--recommended)
3. [Docker Setup Guide (Beginner Friendly)](#3-docker-setup-guide-beginner-friendly)
   - [Prerequisites: Installing Docker Desktop](#prerequisites-installing-docker-desktop)
   - [Method 1: Docker Compose Deployment (Easiest)](#method-1-docker-compose-deployment-easiest)
   - [Method 2: Standalone Docker Image (`docker run`)](#method-2-standalone-docker-image-docker-run)
   - [Docker Troubleshooting](#docker-troubleshooting)
4. [Connecting AI Clients (Claude Code, OpenCode, Cursor)](#4-connecting-ai-clients)
   - [Claude Code (CLI)](#claude-code-cli)
   - [OpenCode & Qwen-Code](#opencode--qwen-code)
   - [Cursor & Windsurf](#cursor--windsurf)
   - [Claude Desktop](#claude-desktop)
5. [Testing with MCP Inspector (Visual UI)](#5-testing-with-mcp-inspector-visual-ui)
6. [Viewing the Analytics Dashboard](#6-viewing-the-analytics-dashboard)

---

## 1. What is MCP Orchestrator?

When AI tools (like Claude Code or Cursor) connect to external capabilities (like GitHub, Filesystem, or Memory), standard implementations load **all tool definitions** upfront. When you connect 50+ tools, this consumes thousands of prompt tokens every single turn.

**MCP Orchestrator** sits as a smart proxy between your AI tool and downstream MCP servers. It exposes only **two lightweight meta-tools** (`discover_tools` and `invoke_tool`), filtering and retrieving only the top candidate tools relevant to your prompt and active project context. This achieves a **90.3% reduction in token usage** while preserving high task execution accuracy.

---

## 2. Configuration Guide (`mcp_orc.json`)

Downstream MCP servers are defined in a configuration file named `mcp_orc.json`. MCP Orchestrator supports **two flexible configuration styles**:

### Option A: Direct API Keys (Simplicity / Quick Start)
If you want to quickly test a server without creating a `.env` file, paste the raw API key directly into `mcp_orc.json`:

```json
{
  "mcpServers": {
    "github-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_raw_github_token_pasted_here"
      }
    }
  }
}
```

### Option B: `.env` Variables (Security / Recommended)
If you want to commit `mcp_orc.json` to version control safely, use template syntax `${VAR_NAME}` in `mcp_orc.json` and keep your real secret key inside `.env`:

* **`mcp_orc.json`**:
  ```json
  {
    "mcpServers": {
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

* **`.env`**:
  ```env
  GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_real_github_token
  ```

> [!NOTE]
> You can freely mix Option A and Option B in the exact same `mcp_orc.json` file!

---

## 3. Docker Setup Guide (Beginner Friendly)

Using Docker allows you to run MCP Orchestrator, PostgreSQL, and ChromaDB without installing database dependencies on your local machine.

### Prerequisites: Installing Docker Desktop
1. Download **Docker Desktop** for Windows/macOS from [docker.com](https://www.docker.com/products/docker-desktop/).
2. Install and launch **Docker Desktop**.
3. Ensure Docker Desktop displays **"Engine Running"** (green whale icon in system tray).

---

### Method 1: Docker Compose Deployment (Easiest)

Docker Compose automatically launches all three container services (PostgreSQL, ChromaDB, and MCP Orchestrator Gateway):

1. **Step 1**: Ensure your `mcp_orc.json` and `.env` files exist in the project directory.
2. **Step 2**: Open your terminal in the project folder and run:
   ```bash
   docker-compose up -d --build
   ```
3. **Step 3**: Verify the container status:
   ```bash
   docker-compose ps
   ```
   *You will see `mcp-orchestrator-app` listening on port `8000` (`http://localhost:8000/sse`).*

4. **To stop the Docker stack**:
   ```bash
   docker-compose down
   ```

---

### Method 2: Standalone Docker Image (`docker run`)

If you want to run only the pre-built MCP Orchestrator image using a single command:

```bash
docker run -d \
  --name mcp-orchestrator \
  -p 8000:8000 \
  --env-file .env \
  -v /path/to/your/mcp_orc.json:/app/mcp_orc.json \
  mcp-orchestrator:latest
```

---

### Docker Troubleshooting

* **Issue: `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`**
  * **Fix**: Docker Desktop application is closed. Open Docker Desktop from your Windows Start Menu, wait until it shows "Running", and re-run your docker command.
* **Issue: Missing `mcp_orc.json` in container**
  * **Fix**: Ensure your local `mcp_orc.json` exists before running `docker-compose up`.

---

## 4. Connecting AI Clients

Once your Docker containers or Python server are running on port `8000`, connect your AI assistant using the **SSE Transport URL**: `http://localhost:8000/sse`.

### Claude Code (CLI)
Run in your terminal:
```bash
claude mcp add mcp-orchestrator --url http://localhost:8000/sse
```

### OpenCode & Qwen-Code
Add to your `opencode.json` configuration file:
```json
{
  "mcp": {
    "mcp-orchestrator": {
      "type": "sse",
      "url": "http://localhost:8000/sse",
      "enabled": true
    }
  }
}
```

### Cursor & Windsurf
1. Open **Cursor Settings** $\rightarrow$ **Features** $\rightarrow$ **MCP**.
2. Click **+ Add New MCP Server**.
3. **Name**: `mcp-orchestrator`
4. **Type**: `SSE`
5. **URL**: `http://localhost:8000/sse`

### Claude Desktop
Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):
```json
{
  "mcpServers": {
    "mcp-orchestrator": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

---

## 5. Testing with MCP Inspector (Visual UI)

You can test candidate retrieval and tool execution interactively using the official MCP Inspector.

1. Launch MCP Inspector in terminal:
   ```bash
   npx @modelcontextprotocol/inspector
   ```
2. Open `http://localhost:5173` in your browser.
3. Set connection parameters:
   * **Server ID**: `mcp-orchestrator`
   * **Transport Type**: `sse`
   * **URL**: `http://localhost:8000/sse`
   * Click **Connect**.

4. **Test `discover_tools`**:
   * Set `query`: `"read configuration file"`
   * Set `workspace_dir`: `.`
   * Click **Execute Tool** $\rightarrow$ Returns top-ranked filesystem tools with score breakdowns!

5. **Test `invoke_tool`**:
   * Set `server_id`: `filesystem-server` *(use hyphens, not underscores)*
   * Set `tool_id`: `read_file`
   * Set `arguments`: `{"path": "mcp_orc.json"}`
   * Click **Execute Tool** $\rightarrow$ Proxies downstream call and returns file contents!

---

## 6. Viewing the Analytics Dashboard

MCP Orchestrator provides a web interface displaying token reduction metrics, task success rates, and ablation performance charts.

1. Start the dashboard server:
   ```bash
   python -m uvicorn dashboard.app:app --port 8002
   ```
2. Open `http://localhost:8002` in your browser to view live KPI cards and Chart.js benchmark graphs!
