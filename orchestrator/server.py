"""FastMCP Server implementation for MCP Orchestrator.

Exposes exactly two meta-tools over HTTP/SSE (0.0.0.0:8000):
1. discover_tools(query: str, workspace_dir: str, bypass: bool) -> 3-Term Ranked tool descriptions (or full unranked list in bypass mode)
2. invoke_tool(server_id: str, tool_id: str, arguments: dict) -> Proxies call to downstream MCP server & logs outcome
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from sqlalchemy import select
from fastmcp import FastMCP
from orchestrator.config import (
    settings,
    load_downstream_servers,
    ensure_global_user_config,
    get_global_sync_metadata_path,
    compute_config_hash,
)
from orchestrator.db import (
    AsyncSessionLocal,
    MCPTool,
    MCPServer,
    log_tool_invocation_outcome,
    init_db,
)
from orchestrator.client import DownstreamMCPClient

import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

_services_initialized = False
_is_warming_up = False


async def background_warmup():
    """Proactively initialize database and preload vector embeddings in the background."""
    global _is_warming_up, _services_initialized
    if _services_initialized or _is_warming_up:
        return
    _is_warming_up = True
    try:
        await init_services(force_sync=False)
        from retrieval.embedder import get_embedding_model
        model = get_embedding_model()
        _ = model.encode(["warmup query"])
        _services_initialized = True
        logger.info("Background warmup complete: Database, SentenceTransformers, and ChromaDB ready.")
    except Exception as e:
        logger.debug(f"Background warmup note: {e}")
    finally:
        _is_warming_up = False


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """FastMCP lifespan that kicks off non-blocking background pre-warming while keeping stdio handshake instant."""
    warmup_task = asyncio.create_task(background_warmup())
    try:
        yield {}
    finally:
        if not warmup_task.done():
            warmup_task.cancel()


# Initialize FastMCP Server with lifespan
mcp_server = FastMCP(
    name="MCP-Orchestrator",
    instructions="Middleware MCP gateway providing tool discovery and proxy tool execution.",
    lifespan=server_lifespan,
)


async def ensure_services_initialized(force_sync: bool = False):
    """Lazily initialize DB schema, collections, and server sync before tool execution."""
    global _services_initialized
    if _services_initialized and not force_sync:
        return
    await init_services(force_sync=force_sync)
    _services_initialized = True


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

    if not is_bypass:
        await ensure_services_initialized()

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
                    "bypass": False,
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
    from orchestrator.graph import run_online_query_pipeline

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
    # Ensure database schema exists for telemetry logging (fast <5ms)
    await init_db()

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception:
            arguments = {}

    # Tolerance for LLM parameter format variations
    if isinstance(arguments, dict):
        if not server_id and "server_id" in arguments:
            server_id = str(arguments.pop("server_id"))
        if not tool_id and "tool_id" in arguments:
            tool_id = str(arguments.pop("tool_id"))

    logger.info(
        f"invoke_tool called for server_id='{server_id}', tool_id='{tool_id}' with args: {arguments}"
    )

    full_tool_id = tool_id if ":" in tool_id else f"{server_id}:{tool_id}"
    real_tool_name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id

    # Resolve server configuration dynamically (with alias tolerance for -server suffix)
    target_cfg = None
    server_list = load_downstream_servers()
    clean_target = server_id.removesuffix("-server")
    for cfg in server_list:
        clean_cfg = cfg.id.removesuffix("-server")
        if cfg.id == server_id or clean_target == clean_cfg:
            target_cfg = cfg
            break

    if not target_cfg:
        async with AsyncSessionLocal() as session:
            db_server = await session.get(MCPServer, server_id)
            if not db_server:
                alt_id = server_id.removesuffix("-server") if server_id.endswith("-server") else f"{server_id}-server"
                db_server = await session.get(MCPServer, alt_id)

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


async def sync_downstream_servers_if_needed(force: bool = False):
    """Sync downstream servers only if config file has changed, data is missing, or force=True."""
    config_path = ensure_global_user_config()
    current_hash = compute_config_hash(config_path)
    metadata_path = get_global_sync_metadata_path()

    stored_hash = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                stored_hash = meta.get("config_sha256")
        except Exception as e:
            logger.debug(f"Could not read sync metadata: {e}")

    async with AsyncSessionLocal() as session:
        db_servers_res = await session.execute(select(MCPServer))
        db_server_ids = set(s.id for s in db_servers_res.scalars().all())

        configured_servers = load_downstream_servers(config_path)
        configured_ids = set(s.id for s in configured_servers)

        missing_servers = configured_ids - db_server_ids

        from retrieval.embedder import sync_all_embeddings_from_db, get_or_create_collections

        # Check ChromaDB collections
        chroma_empty = False
        try:
            servers_col, tools_col = get_or_create_collections()
            if servers_col.count() == 0 or tools_col.count() == 0:
                chroma_empty = True
        except Exception:
            chroma_empty = True

        needs_sync = (
            force
            or (stored_hash != current_hash)
            or bool(missing_servers)
            or not db_server_ids
            or chroma_empty
        )

        if needs_sync:
            logger.info(
                f"Config change or missing data detected (hash: {current_hash[:8] if current_hash else 'none'}). "
                f"Registering downstream MCP servers into global ChromaDB & SQLite..."
            )
            try:
                from orchestrator.registrar import register_downstream_servers
                await register_downstream_servers(session, server_configs=configured_servers)
            except Exception as reg_err:
                logger.warning(f"Registration sync warning: {reg_err}")

            # Sync vector embeddings into persistent ChromaDB
            try:
                await sync_all_embeddings_from_db(session)
            except Exception as e:
                logger.warning(f"Could not sync vector embeddings: {e}")

            # Save updated sync metadata
            try:
                sync_meta = {
                    "config_sha256": current_hash,
                    "config_path": config_path,
                    "servers": list(configured_ids),
                    "last_synced_at": time.time(),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(sync_meta, f, indent=2)
                logger.info(f"Saved sync metadata to '{metadata_path}'.")
            except Exception as meta_err:
                logger.warning(f"Could not save sync metadata: {meta_err}")
        else:
            logger.info(
                f"Global ChromaDB and SQLite cache is up-to-date (hash: {current_hash[:8]}). "
                f"Skipping re-seeding."
            )


async def init_services(force_sync: bool = False):
    logger.info("Initializing MCP Orchestrator DB and embedding index...")
    # 1. Initialize DB schema
    await init_db()

    # 2. Automatically sync downstream servers if needed or force_sync requested
    await sync_downstream_servers_if_needed(force=force_sync)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="MCP Orchestrator Gateway Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="Transport mode (stdio or sse)")
    parser.add_argument("--host", default=settings.HOST, help="Host for SSE transport")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port for SSE transport")
    parser.add_argument("--sync", action="store_true", help="Sync downstream MCP servers from config and exit")
    args = parser.parse_args()

    if args.sync:
        asyncio.run(ensure_services_initialized(force_sync=True))
        logger.info("Downstream server sync complete. Exiting (--sync specified).")
        return

    # In stdio mode, start the server IMMEDIATELY so OpenCode can handshake in < 300ms.
    # Services are initialized lazily on the first tool invocation.
    logger.info(f"Starting MCP Orchestrator FastMCP server (transport: '{args.transport}')...")
    if args.transport == "sse":
        asyncio.run(ensure_services_initialized())
        mcp_server.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp_server.run(transport="stdio")


if __name__ == "__main__":
    cli_main()
