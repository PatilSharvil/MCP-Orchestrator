"""FastMCP Server implementation for MCP Orchestrator.

Exposes exactly two meta-tools over HTTP/SSE (0.0.0.0:8000):
1. discover_tools(query: str, workspace_dir: str) -> 3-Term Ranked tool descriptions powered by LangGraph state machine
2. invoke_tool(server_id: str, tool_id: str, arguments: dict) -> Proxies call to downstream MCP server & logs outcome
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import select
from fastmcp import FastMCP

from orchestrator.config import settings, load_downstream_servers
from orchestrator.db import (
    AsyncSessionLocal,
    MCPTool,
    MCPServer,
    log_tool_invocation_outcome,
    init_db,
)
from orchestrator.client import DownstreamMCPClient
from orchestrator.graph import run_online_query_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp_server = FastMCP(
    name="MCP-Orchestrator",
    instructions="Middleware MCP gateway providing tool discovery and proxy tool execution.",
)


@mcp_server.tool()
async def discover_tools(
    query: str = "",
    workspace_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover relevant tools across all connected downstream MCP servers using 3-Term Weighted Ranking & LangGraph pipeline.

    Score = w1 * QuerySim + w2 * ProjMatch + w3 * PastSuccess

    Args:
        query: The user query or task description to match against tool descriptions.
        workspace_dir: Optional workspace directory path (auto-detected if omitted).

    Returns:
        List of candidate tool objects containing tool_id, server_id, name, description, input_schema,
        final weighted score, and per-term score_breakdown.
    """
    logger.info(f"Received discover_tools query: '{query}' | workspace_dir: '{workspace_dir}'")

    if not query or not query.strip():
        async with AsyncSessionLocal() as session:
            stmt = select(MCPTool).limit(settings.TOP_M_TOOLS)
            res = await session.execute(stmt)
            tools = res.scalars().all()
            return [
                {
                    "tool_id": t.id,
                    "server_id": t.server_id,
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "score": 1.0,
                    "score_breakdown": {
                        "query_sim": 1.0,
                        "proj_match": 0.0,
                        "past_success": 0.5,
                        "weights": {"w1": 0.5, "w2": 0.0, "w3": 0.5},
                    },
                }
                for t in tools
            ]

    # Execute LangGraph Online Query State Machine
    ranked_tools = await run_online_query_pipeline(query=query, workspace_dir=workspace_dir)
    logger.info(f"discover_tools returning {len(ranked_tools)} ranked tools via LangGraph.")
    return ranked_tools


@mcp_server.tool()
async def invoke_tool(
    server_id: str,
    tool_id: str,
    arguments: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Proxy execution of a tool to its target downstream MCP server and log outcome.

    Args:
        server_id: Target downstream server ID (e.g., 'filesystem-server')
        tool_id: Unique tool identifier or tool name (e.g., 'read_file')
        arguments: Parameters object to pass to the tool.

    Returns:
        Result output returned by the downstream MCP server.
    """
    logger.info(
        f"invoke_tool called for server_id='{server_id}', tool_id='{tool_id}' with args: {arguments}"
    )

    full_tool_id = tool_id if ":" in tool_id else f"{server_id}:{tool_id}"
    real_tool_name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id

    # Resolve server configuration dynamically
    target_cfg = None
    server_list = load_downstream_servers()
    for cfg in server_list:
        if cfg.id == server_id:
            target_cfg = cfg
            break

    if not target_cfg:
        async with AsyncSessionLocal() as session:
            db_server = await session.get(MCPServer, server_id)
            if db_server and db_server.connection_config:
                from orchestrator.config import DownstreamServerConfig

                conn = db_server.connection_config
                target_cfg = DownstreamServerConfig(
                    id=db_server.id,
                    name=db_server.name,
                    transport_type=db_server.transport_type,
                    command=conn.get("command", ""),
                    args=conn.get("args", []),
                    env=conn.get("env", {}),
                )

    if not target_cfg:
        raise ValueError(
            f"Downstream server '{server_id}' is not registered or unreachable."
        )

    start_time = time.perf_counter()
    success = False
    error_msg = None
    result = {}

    try:
        result = await DownstreamMCPClient.invoke_tool(
            server_cfg=target_cfg,
            tool_name=real_tool_name,
            arguments=arguments,
        )
        success = not result.get("isError", False)
        if not success:
            error_msg = "Tool execution returned isError=True"
    except Exception as e:
        success = False
        error_msg = str(e)
        result = {"isError": True, "content": [{"type": "text", "text": f"Error: {e}"}]}
    finally:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        try:
            async with AsyncSessionLocal() as session:
                await log_tool_invocation_outcome(
                    session=session,
                    tool_id=full_tool_id,
                    server_id=server_id,
                    tool_name=real_tool_name,
                    success=success,
                    error_message=error_msg,
                    latency_ms=latency_ms,
                )
        except Exception as log_err:
            logger.warning(f"Could not log tool invocation outcome: {log_err}")

    return result


if __name__ == "__main__":
    logger.info(
        f"Starting MCP Orchestrator FastMCP server over SSE on http://{settings.HOST}:{settings.PORT}..."
    )

    # Initialize database tables on startup
    asyncio.run(init_db())

    # Run FastMCP server over SSE transport on 0.0.0.0:8000
    mcp_server.run(transport="sse", host=settings.HOST, port=settings.PORT)
