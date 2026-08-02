"""FastMCP Server implementation for MCP Orchestrator.

Exposes exactly two meta-tools over HTTP/SSE (0.0.0.0:8000):
1. discover_tools(query: str) -> Ranked tool descriptions using Two-Stage Vector Retrieval (ChromaDB + PostgreSQL)
2. invoke_tool(server_id: str, tool_id: str, arguments: dict) -> Proxies call to real downstream MCP server
"""

import asyncio
import logging
from typing import Dict, Any, List
from sqlalchemy import select
from fastmcp import FastMCP

from orchestrator.config import settings, DEFAULT_DOWNSTREAM_SERVERS
from orchestrator.db import AsyncSessionLocal, MCPTool, MCPServer, QueryLog, RetrievalLog, init_db
from orchestrator.client import DownstreamMCPClient
from retrieval.retriever import two_stage_retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp_server = FastMCP(
    name="MCP-Orchestrator",
    instructions="Middleware MCP gateway providing tool discovery and proxy tool execution.",
)


@mcp_server.tool()
async def discover_tools(query: str = "") -> List[Dict[str, Any]]:
    """Discover relevant tools across all connected downstream MCP servers.

    Phase 3 implementation performs Two-Stage Vector Retrieval against ChromaDB:
    Stage 1: Narrows down to top-N candidate servers.
    Stage 2: Retrieves top-M tools from candidate servers via metadata filtering.

    Args:
        query: The user query or task description to match against tool descriptions.

    Returns:
        List of candidate tool objects containing tool_id, server_id, name, description, input_schema, and similarity score.
    """
    logger.info(f"Received discover_tools query: '{query}'")

    if not query or not query.strip():
        # Fallback to returning recent/all tools from DB if query is empty
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
                }
                for t in tools
            ]

    # 1. Execute Two-Stage Retrieval via ChromaDB
    vector_results = two_stage_retrieve(query=query)

    discovered = []
    async with AsyncSessionLocal() as session:
        for item in vector_results:
            tool_id = item["tool_id"]
            db_tool = await session.get(MCPTool, tool_id)
            if db_tool:
                discovered.append(
                    {
                        "tool_id": db_tool.id,
                        "server_id": db_tool.server_id,
                        "name": db_tool.name,
                        "description": db_tool.description,
                        "input_schema": db_tool.input_schema,
                        "score": item["score"],
                    }
                )
            else:
                # Fallback if DB record missing
                discovered.append(
                    {
                        "tool_id": tool_id,
                        "server_id": item["server_id"],
                        "name": item["name"],
                        "description": "",
                        "input_schema": {},
                        "score": item["score"],
                    }
                )

        # 2. Log execution to query_logs and retrieval_logs
        try:
            query_entry = QueryLog(
                query=query,
                discovered_tool_ids=[t["tool_id"] for t in discovered],
            )
            session.add(query_entry)
            await session.flush()  # Obtain query_entry.id

            retrieval_entry = RetrievalLog(
                query_log_id=query_entry.id,
                retrieved_tool_ids=[t["tool_id"] for t in discovered],
                scores={t["tool_id"]: t["score"] for t in discovered},
            )
            session.add(retrieval_entry)
            await session.commit()
        except Exception as e:
            logger.warning(f"Failed to log query/retrieval: {e}")

    logger.info(f"discover_tools returning {len(discovered)} tools.")
    return discovered


@mcp_server.tool()
async def invoke_tool(
    server_id: str,
    tool_id: str,
    arguments: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Proxy execution of a tool to its target downstream MCP server.

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

    # Resolve server configuration
    target_cfg = None
    for cfg in DEFAULT_DOWNSTREAM_SERVERS:
        if cfg.id == server_id:
            target_cfg = cfg
            break

    if not target_cfg:
        # Fallback to querying DB for server config if not in static config list
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

    # Extract clean tool name if tool_id is prefixed with 'server_id:'
    real_tool_name = tool_id
    if ":" in tool_id:
        real_tool_name = tool_id.split(":", 1)[1]

    # Proxy call downstream
    result = await DownstreamMCPClient.invoke_tool(
        server_cfg=target_cfg,
        tool_name=real_tool_name,
        arguments=arguments,
    )
    return result


if __name__ == "__main__":
    logger.info(
        f"Starting MCP Orchestrator FastMCP server over SSE on http://{settings.HOST}:{settings.PORT}..."
    )

    # Initialize database tables on startup
    asyncio.run(init_db())

    # Run FastMCP server over SSE transport on 0.0.0.0:8000
    mcp_server.run(transport="sse", host=settings.HOST, port=settings.PORT)
