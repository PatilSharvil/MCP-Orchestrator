"""MCP Client manager for connecting to downstream MCP servers."""

import os
import logging
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from orchestrator.config import DownstreamServerConfig

logger = logging.getLogger(__name__)


class DownstreamMCPClient:
    """Manages connections and invocations to downstream MCP servers."""

    @staticmethod
    async def fetch_tools(server_cfg: DownstreamServerConfig) -> List[Dict[str, Any]]:
        """Connect to a downstream stdio MCP server and retrieve its list of tools.

        Returns a list of dictionaries with tool metadata:
        [{'name': str, 'description': str, 'input_schema': dict}]
        """
        if server_cfg.transport_type != "stdio":
            raise NotImplementedError(
                f"Transport type '{server_cfg.transport_type}' is not supported yet."
            )

        # Build StdioServerParameters
        env = {**os.environ, **server_cfg.env}
        server_params = StdioServerParameters(
            command=server_cfg.command,
            args=server_cfg.args,
            env=env,
        )

        logger.info(
            f"Connecting to downstream server '{server_cfg.id}' via stdio: {server_cfg.command} {' '.join(server_cfg.args)}"
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
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
                logger.info(
                    f"Discovered {len(tools_data)} tools from '{server_cfg.id}'"
                )
                return tools_data

    @staticmethod
    async def invoke_tool(
        server_cfg: DownstreamServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Connect to downstream MCP server and invoke a specific tool with arguments.

        Returns the execution result dict.
        """
        if server_cfg.transport_type != "stdio":
            raise NotImplementedError(
                f"Transport type '{server_cfg.transport_type}' is not supported yet."
            )

        env = {**os.environ, **server_cfg.env}
        server_params = StdioServerParameters(
            command=server_cfg.command,
            args=server_cfg.args,
            env=env,
        )

        logger.info(
            f"Invoking tool '{tool_name}' on downstream server '{server_cfg.id}' with arguments: {arguments}"
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool_name, arguments=arguments)

                # Format content items
                content_list = []
                for content in res.content:
                    if hasattr(content, "text"):
                        content_list.append({"type": "text", "text": content.text})
                    else:
                        content_list.append({"type": "raw", "data": str(content)})

                return {
                    "isError": getattr(res, "isError", False),
                    "content": content_list,
                }
