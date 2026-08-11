"""FastMCP Server implementation for MCP Orchestrator.

Exposes exactly two meta-tools over HTTP/SSE (0.0.0.0:8000):
1. discover_tools(query: str, workspace_dir: str, bypass: bool) -> 3-Term Ranked tool descriptions (or full unranked list in bypass mode)
2. invoke_tool(server_id: str, tool_id: str, arguments: dict) -> Proxies call to downstream MCP server & logs outcome
"""

import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
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
from retrieval.embedder import sync_all_embeddings_from_db

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
    bypass: Optional[bool] = None,
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover relevant tools across all connected downstream MCP servers using 3-Term Weighted Ranking & LangGraph pipeline.

    Args:
        query: The user query or task description to match against tool descriptions.
        workspace_dir: Optional workspace directory path (auto-detected if omitted).
        bypass: Optional boolean flag to enable naive bypass mode (returns ALL unranked tools).
        ablation_mode: Optional string indicating ablation mode.

    Returns:
        List of candidate tool objects containing tool_id, server_id, name, description, input_schema,
        and score details.
    """
    logger.info(f"Received discover_tools query: '{query}' | workspace_dir: '{workspace_dir}' | bypass: {bypass} | ablation_mode: {ablation_mode}")

    is_bypass = bypass if bypass is not None else settings.ORCHESTRATOR_BYPASS_MODE

    # Mode A: Naive Bypass Mode (Returns ALL unranked tools from database)
    if is_bypass:
        logger.info("Executing Naive Bypass Mode: Returning ALL unranked tools from database...")
        async with AsyncSessionLocal() as session:
            stmt = select(MCPTool)
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
                    "bypass": True,
                    "score_breakdown": None,
                }
                for t in tools
            ]

    # Empty Query Fallback
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

    # Mode B: Full Orchestrator Pipeline (LangGraph Online Query State Machine)
    ranked_tools = await run_online_query_pipeline(
        query=query,
        workspace_dir=workspace_dir,
        ablation_mode=ablation_mode,
    )

    # Minimum Relevance Threshold check for conversational/irrelevant queries
    if ranked_tools and not is_bypass:
        max_query_sim = max(t.get("score_breakdown", {}).get("query_sim", 0.0) for t in ranked_tools)
        if max_query_sim < settings.MIN_RELEVANCE_THRESHOLD:
            logger.info(
                f"Max query_sim ({max_query_sim:.4f}) is below MIN_RELEVANCE_THRESHOLD ({settings.MIN_RELEVANCE_THRESHOLD}). Returning empty tool list."
            )
            return []

    logger.info(f"discover_tools returning {len(ranked_tools)} ranked tools via LangGraph.")
    return ranked_tools


@mcp_server.tool()
async def invoke_tool(
    server_id: str,
    tool_id: str,
    arguments: Union[Dict[str, Any], str] = {},
) -> Dict[str, Any]:
    """Proxy execution of a tool to its target downstream MCP server and log outcome.

    Args:
        server_id: Target downstream server ID (e.g., 'filesystem-server')
        tool_id: Unique tool identifier or tool name (e.g., 'read_file')
        arguments: Parameters object to pass to the tool.

    Returns:
        Result output returned by the downstream MCP server.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception:
            arguments = {}

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


async def init_services():
    logger.info("Initializing MCP Orchestrator DB and embedding index...")
    # 1. Initialize DB schema
    await init_db()

    # 2. Check if DB has tools; if empty, auto-register downstream servers from mcp_orc.json
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(MCPTool))
            tools = res.scalars().all()
            if not tools:
                logger.info("Database is empty on startup. Automatically registering downstream servers...")
                from seed_registration import register_downstream_servers
                await register_downstream_servers(session)

            # 3. Sync vector embeddings into ChromaDB
            await sync_all_embeddings_from_db(session)
    except Exception as e:
        logger.warning(f"Could not sync vector embeddings on startup: {e}")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="MCP Orchestrator Gateway Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="Transport mode (stdio or sse)")
    parser.add_argument("--host", default=settings.HOST, help="Host for SSE transport")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port for SSE transport")
    args = parser.parse_args()

    # 1. Run async database and vector embedding initialization
    asyncio.run(init_services())

    # 2. Run FastMCP server synchronously (anyio.run creates its own clean event loop)
    logger.info(f"Starting MCP Orchestrator FastMCP server (transport: '{args.transport}')...")
    if args.transport == "sse":
        mcp_server.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp_server.run(transport="stdio")


if __name__ == "__main__":
    cli_main()
