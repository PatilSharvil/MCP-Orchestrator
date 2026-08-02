"""Registration logic for downstream MCP servers and tools."""

import logging
from typing import List, Dict
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from orchestrator.config import DownstreamServerConfig, DEFAULT_DOWNSTREAM_SERVERS
from orchestrator.client import DownstreamMCPClient
from orchestrator.db import MCPServer, MCPTool, init_db, AsyncSessionLocal

logger = logging.getLogger(__name__)


async def register_downstream_servers(
    session: AsyncSession,
    server_configs: List[DownstreamServerConfig] = DEFAULT_DOWNSTREAM_SERVERS,
) -> Dict[str, int]:
    """Discover tools from configured downstream MCP servers and upsert into PostgreSQL.

    Guarantees idempotency via PostgreSQL ON CONFLICT DO UPDATE upserts.
    Returns stats summary: {'servers_processed': int, 'tools_registered': int}.
    """
    servers_processed = 0
    tools_registered = 0

    for cfg in server_configs:
        logger.info(f"Starting registration for downstream server '{cfg.id}'...")
        try:
            # 1. Fetch tools directly from downstream MCP server
            discovered_tools = await DownstreamMCPClient.fetch_tools(cfg)

            # 2. Upsert mcp_servers row
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

            # 3. Upsert each tool into mcp_tools
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
                f"Successfully registered server '{cfg.id}' with {len(discovered_tools)} tools."
            )

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Failed to register downstream server '{cfg.id}': {e}",
                exc_info=True,
            )
            raise e

    return {
        "servers_processed": servers_processed,
        "tools_registered": tools_registered,
    }


async def run_registration():
    """CLI / programmatic helper to initialize DB schema and run registration."""
    await init_db()
    async with AsyncSessionLocal() as session:
        stats = await register_downstream_servers(session)
        print(f"Registration Complete: {stats}")
        return stats

