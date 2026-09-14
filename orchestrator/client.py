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


class DownstreamServerWorker:
    """A dedicated background actor task that owns the stdio_client or sse_client lifecycle.

    Ensures anyio task groups are entered, operated, and exited entirely within the same
    dedicated asyncio task, eliminating task-scope cancel errors.
    """

    def __init__(self, server_cfg: DownstreamServerConfig):
        self.server_cfg = server_cfg
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.is_ready: asyncio.Event = asyncio.Event()
        self.error: Optional[Exception] = None
        self.last_accessed_at: float = time.time()

    async def start(self):
        """Start the background worker task and wait for session handshake."""
        self.worker_task = asyncio.create_task(self._run_loop())
        try:
            await asyncio.wait_for(self.is_ready.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            self.error = TimeoutError(f"Downstream server '{self.server_cfg.id}' timed out during initialization handshake.")
        if self.error:
            raise self.error

    async def _run_loop(self):
        try:
            if self.server_cfg.transport_type in ("sse", "remote"):
                if not self.server_cfg.url:
                    raise ValueError(f"URL is required for SSE transport on server '{self.server_cfg.id}'.")
                cm = sse_client(self.server_cfg.url, headers=self.server_cfg.headers)
            else:
                if self.server_cfg.transport_type != "stdio":
                    raise NotImplementedError(
                        f"Transport type '{self.server_cfg.transport_type}' is not supported yet."
                    )
                resolved_cmd = shutil.which(self.server_cfg.command) or self.server_cfg.command
                env = {**os.environ, **self.server_cfg.env}
                server_params = StdioServerParameters(
                    command=resolved_cmd,
                    args=self.server_cfg.args,
                    env=env,
                )
                cm = stdio_client(server_params)

            async with cm as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.is_ready.set()

                    while True:
                        req = await self.request_queue.get()
                        if req is None:  # Graceful stop signal
                            break

                        action, args, future = req
                        self.last_accessed_at = time.time()
                        try:
                            if action == "call_tool":
                                tool_name, tool_args = args
                                res = await session.call_tool(tool_name, arguments=tool_args)
                                if not future.done():
                                    future.set_result(res)
                            elif action == "list_tools":
                                res = await session.list_tools()
                                if not future.done():
                                    future.set_result(res)
                        except Exception as exec_err:
                            if not future.done():
                                future.set_exception(exec_err)
                        finally:
                            self.request_queue.task_done()
        except Exception as startup_err:
            self.error = startup_err
            self.is_ready.set()
        finally:
            self.worker_task = None

    def is_alive(self) -> bool:
        return self.worker_task is not None and not self.worker_task.done()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.is_alive():
            raise RuntimeError(f"Server worker for '{self.server_cfg.id}' is not running.")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.request_queue.put(("call_tool", (tool_name, arguments), future))
        return await future

    async def list_tools(self) -> Any:
        if not self.is_alive():
            raise RuntimeError(f"Server worker for '{self.server_cfg.id}' is not running.")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.request_queue.put(("list_tools", (), future))
        return await future

    async def stop(self):
        """Cleanly stop the worker."""
        if self.worker_task and not self.worker_task.done():
            try:
                await self.request_queue.put(None)
                await asyncio.wait_for(self.worker_task, timeout=3.0)
            except Exception:
                if self.worker_task and not self.worker_task.done():
                    self.worker_task.cancel()


class DownstreamMCPClient:
    """Manages connections and invocations to downstream MCP servers."""

    _workers: Dict[str, DownstreamServerWorker] = {}
    _locks: Dict[str, asyncio.Lock] = {}

    @classmethod
    async def get_or_create_worker(cls, server_cfg: DownstreamServerConfig) -> DownstreamServerWorker:
        """Get an existing active worker or start a new background server worker."""
        server_id = server_cfg.id
        if server_id not in cls._locks:
            cls._locks[server_id] = asyncio.Lock()

        async with cls._locks[server_id]:
            if server_id in cls._workers and cls._workers[server_id].is_alive():
                return cls._workers[server_id]

            # If existing worker is dead, clean it up
            if server_id in cls._workers:
                await cls._workers[server_id].stop()
                cls._workers.pop(server_id, None)

            logger.info(f"Starting long-lived worker for downstream server '{server_id}'...")
            worker = DownstreamServerWorker(server_cfg)
            await worker.start()
            cls._workers[server_id] = worker
            return worker

    @classmethod
    async def close_session(cls, server_id: str):
        """Close and remove a specific downstream server worker."""
        if server_id in cls._workers:
            logger.info(f"Stopping downstream worker for server '{server_id}'...")
            worker = cls._workers.pop(server_id)
            await worker.stop()

    @classmethod
    async def close_all_sessions(cls):
        """Close all active downstream server workers."""
        server_ids = list(cls._workers.keys())
        for sid in server_ids:
            await cls.close_session(sid)

    @classmethod
    async def fetch_tools(cls, server_cfg: DownstreamServerConfig) -> List[Dict[str, Any]]:
        """Connect to a downstream MCP server and retrieve its tool list during registration."""
        logger.info(f"Fetching tools from downstream server '{server_cfg.id}'...")
        
        async def _fetch():
            if server_cfg.transport_type in ("sse", "remote"):
                if not server_cfg.url:
                    raise ValueError(f"URL is required for SSE transport on server '{server_cfg.id}'.")
                cm = sse_client(server_cfg.url, headers=server_cfg.headers)
            else:
                resolved_cmd = shutil.which(server_cfg.command) or server_cfg.command
                env = {**os.environ, **server_cfg.env}
                server_params = StdioServerParameters(
                    command=resolved_cmd,
                    args=server_cfg.args,
                    env=env,
                )
                cm = stdio_client(server_params)

            async with cm as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.list_tools()

        try:
            response = await asyncio.wait_for(_fetch(), timeout=25.0)
        except Exception as e:
            logger.warning(f"Error fetching tools from '{server_cfg.id}': {e}")
            raise

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

        worker = await cls.get_or_create_worker(server_cfg)
        try:
            res = await worker.call_tool(tool_name, arguments=arguments)
        except Exception as e:
            logger.warning(
                f"Error executing tool '{tool_name}' on server '{server_cfg.id}': {e}. Refreshing session and retrying once..."
            )
            await cls.close_session(server_cfg.id)
            worker = await cls.get_or_create_worker(server_cfg)
            res = await worker.call_tool(tool_name, arguments=arguments)

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

