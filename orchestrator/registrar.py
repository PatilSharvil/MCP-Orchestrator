"""Registration logic for downstream MCP servers and tools."""

import os
import logging
from typing import List, Dict, Optional
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from orchestrator.config import DownstreamServerConfig, load_downstream_servers
from orchestrator.client import DownstreamMCPClient
from orchestrator.db import MCPServer, MCPTool, init_db, AsyncSessionLocal
from retrieval.embedder import upsert_server_and_tools_embeddings

logger = logging.getLogger(__name__)


async def register_downstream_servers(
    session: AsyncSession,
    server_configs: Optional[List[DownstreamServerConfig]] = None,
    chroma_client=None,
) -> Dict[str, int]:
    """Discover tools from configured downstream MCP servers and upsert into PostgreSQL & ChromaDB.

    Guarantees idempotency via PostgreSQL ON CONFLICT DO UPDATE upserts and ChromaDB collection upserts.
    Returns stats summary: {'servers_processed': int, 'tools_registered': int}.
    """
    if server_configs is None:
        server_configs = load_downstream_servers()

    servers_processed = 0
    tools_registered = 0

    for cfg in server_configs:
        # Check authentication requirements if needed
        if cfg.requires_auth and cfg.auth_env_var:
            token_val = cfg.env.get(cfg.auth_env_var) or os.getenv(cfg.auth_env_var, "")
            if not token_val or token_val.startswith("${"):
                logger.warning(
                    f"Skipping downstream server '{cfg.id}': Required auth env var '{cfg.auth_env_var}' is missing or unresolved."
                )
                continue

        logger.info(f"Starting registration for downstream server '{cfg.id}'...")
        try:
            # 1. Fetch tools directly from downstream MCP server
            discovered_tools = await DownstreamMCPClient.fetch_tools(cfg)

            # 2. Upsert mcp_servers row in Postgres
            server_stmt = pg_insert(MCPServer).values(
                id=cfg.id,
                name=cfg.name,
                transport_type=cfg.transport_type,
                connection_config={
                    "command": cfg.command,
                    "args": cfg.args,
                    "env": cfg.env,
                },
                status="active",
            )
            server_upsert = server_stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": server_stmt.excluded.name,
                    "transport_type": server_stmt.excluded.transport_type,
                    "connection_config": server_stmt.excluded.connection_config,
                    "status": "active",
                },
            )
            await session.execute(server_upsert)
            servers_processed += 1

            # 3. Upsert each tool into mcp_tools in Postgres
            for tool_data in discovered_tools:
                tool_id = f"{cfg.id}:{tool_data['name']}"
                tool_stmt = pg_insert(MCPTool).values(
                    id=tool_id,
                    server_id=cfg.id,
                    name=tool_data["name"],
                    description=tool_data["description"],
                    input_schema=tool_data["input_schema"],
                )
                tool_upsert = tool_stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "description": tool_stmt.excluded.description,
                        "input_schema": tool_stmt.excluded.input_schema,
                    },
                )
                await session.execute(tool_upsert)
                tools_registered += 1

            await session.commit()
            logger.info(
                f"Successfully registered server '{cfg.id}' into PostgreSQL with {len(discovered_tools)} tools."
            )

            # 4. Upsert vector embeddings into ChromaDB
            try:
                upsert_server_and_tools_embeddings(
                    server_id=cfg.id,
                    server_name=cfg.name,
                    tools=discovered_tools,
                    chroma_client=chroma_client,
                )
            except Exception as embed_err:
                logger.warning(
                    f"Could not update ChromaDB embeddings for '{cfg.id}': {embed_err}"
                )

        except Exception as e:
            await session.rollback()
            logger.warning(
                f"Downstream server '{cfg.id}' unreachable ({e}). Marking as inactive in database."
            )
            try:
                server_stmt = pg_insert(MCPServer).values(
                    id=cfg.id,
                    name=cfg.name,
                    transport_type=cfg.transport_type,
                    connection_config={
                        "command": cfg.command,
                        "args": cfg.args,
                        "env": cfg.env,
                    },
                    status="inactive",
                )
                server_upsert = server_stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={"status": "inactive"},
                )
                await session.execute(server_upsert)
                await session.commit()
            except Exception as db_err:
                logger.debug(f"Could not record inactive status for '{cfg.id}': {db_err}")

    return {
        "servers_processed": servers_processed,
        "tools_registered": tools_registered,
    }


async def run_registration():
    """CLI / programmatic helper to initialize DB schema and run registration."""
    await init_db()
    async with AsyncSessionLocal() as session:
        stats = await register_downstream_servers(session)
        logger.info(f"Registration Complete: {stats}")
        return stats
