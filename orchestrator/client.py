import os
import time
import shutil
import asyncio
import logging
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from orchestrator.config import DownstreamServerConfig

logger = logging.getLogger(__name__)


class ActiveServerSession:
    """Represents a long-lived active downstream MCP session and its underlying transport."""

    def __init__(self, server_id: str, transport_cm: Any, read: Any, write: Any, session: ClientSession):
        self.server_id = server_id
        self.transport_cm = transport_cm
        self.read = read
        self.write = write
        self.session = session
        self.last_accessed_at = time.time()

    async def close(self):
        """Cleanly tear down the ClientSession and underlying transport context."""
        try:
            await self.session.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Error closing session for '{self.server_id}': {e}")
        try:
            await self.transport_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Error closing transport for '{self.server_id}': {e}")


class DownstreamMCPClient:
    """Manages long-lived persistent connections and tool invocations to downstream MCP servers."""

    _sessions: Dict[str, ActiveServerSession] = {}
    _locks: Dict[str, asyncio.Lock] = {}

    @classmethod
    async def get_or_create_session(cls, server_cfg: DownstreamServerConfig) -> ClientSession:
        """Get an existing active ClientSession or establish a new long-lived connection."""
        server_id = server_cfg.id
        if server_id not in cls._locks:
            cls._locks[server_id] = asyncio.Lock()

        async with cls._locks[server_id]:
            if server_id in cls._sessions:
                active = cls._sessions[server_id]
                active.last_accessed_at = time.time()
                return active.session

            logger.info(f"Establishing long-lived MCP session for downstream server '{server_id}'...")
            if server_cfg.transport_type in ("sse", "remote"):
                if not server_cfg.url:
                    raise ValueError(f"URL is required for SSE transport on server '{server_id}'.")
                cm = sse_client(server_cfg.url, headers=server_cfg.headers)
            else:
                if server_cfg.transport_type != "stdio":
                    raise NotImplementedError(
                        f"Transport type '{server_cfg.transport_type}' is not supported yet."
                    )
                resolved_cmd = shutil.which(server_cfg.command) or server_cfg.command
                env = {**os.environ, **server_cfg.env}
                server_params = StdioServerParameters(
                    command=resolved_cmd,
                    args=server_cfg.args,
                    env=env,
                )
                cm = stdio_client(server_params)

            read, write = await cm.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            active = ActiveServerSession(
                server_id=server_id,
                transport_cm=cm,
                read=read,
                write=write,
                session=session,
            )
            cls._sessions[server_id] = active
            return session

    @classmethod
    async def close_session(cls, server_id: str):
        """Close and remove a specific downstream server session."""
        if server_id in cls._sessions:
            logger.info(f"Closing downstream session for server '{server_id}'...")
            active = cls._sessions.pop(server_id)
            await active.close()

    @classmethod
    async def close_all_sessions(cls):
        """Close all downstream active server sessions."""
        server_ids = list(cls._sessions.keys())
        for sid in server_ids:
            await cls.close_session(sid)

    @classmethod
    async def fetch_tools(cls, server_cfg: DownstreamServerConfig) -> List[Dict[str, Any]]:
        """Connect to a downstream MCP server and retrieve its list of tools."""
        try:
            session = await cls.get_or_create_session(server_cfg)
            response = await session.list_tools()
        except Exception as e:
            logger.warning(f"Failed to fetch tools from session '{server_cfg.id}': {e}. Retrying with fresh session...")
            await cls.close_session(server_cfg.id)
            session = await cls.get_or_create_session(server_cfg)
            response = await session.list_tools()

        tools_data = []
        for tool in response.tools:
            tools_data.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else getattr(tool, "input_schema", {}),
                }
            )
        logger.info(f"Discovered {len(tools_data)} tools from server '{server_cfg.id}'")
        return tools_data

    @classmethod
    async def invoke_tool(
        cls,
        server_cfg: DownstreamServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Connect to downstream MCP server and invoke a specific tool with arguments.

        Reuses persistent long-lived session to keep stateful servers (Playwright, Shell, DevTools) alive.
        """
        logger.info(
            f"Invoking tool '{tool_name}' on downstream server '{server_cfg.id}' with arguments: {arguments}"
        )

        try:
            session = await cls.get_or_create_session(server_cfg)
            res = await session.call_tool(tool_name, arguments=arguments)
        except Exception as e:
            logger.warning(
                f"Error executing tool '{tool_name}' on server '{server_cfg.id}' (session may have closed): {e}. Reconnecting..."
            )
            await cls.close_session(server_cfg.id)
            session = await cls.get_or_create_session(server_cfg)
            res = await session.call_tool(tool_name, arguments=arguments)

        # Format content items
        content_list = []
        for content in res.content:
            if hasattr(content, "text"):
                content_list.append({"type": "text", "text": content.text})
            elif hasattr(content, "image"):
                content_list.append({"type": "image", "data": str(content.image)})
            else:
                content_list.append({"type": "raw", "data": str(content)})

        return {
            "isError": getattr(res, "isError", False),
            "content": content_list,
        }
