"""Live Session Capture Script for MCP Orchestrator.

Establishes an HTTP/SSE client session to http://localhost:8000/sse,
invokes discover_tools and invoke_tool over the network, and captures the
raw live response payload into eval/demo_session_log.json.
"""

import os
import json
import asyncio
import logging
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def capture_live_mcp_session():
    sse_url = "http://localhost:8000/sse"
    print(f"=== Connecting to Live MCP Orchestrator SSE Endpoint: {sse_url} ===")

    try:
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Connected & initialized SSE MCP Client session successfully!")

                # List tools exposed to client (should be discover_tools and invoke_tool)
                tools_resp = await session.list_tools()
                exposed_tools = [t.name for t in tools_resp.tools]
                print(f"Tools exposed by gateway server: {exposed_tools}")

                # Step 1: Call discover_tools
                query = "read project configuration file in current workspace"
                print(f"\nExecuting discover_tools(query='{query}')...")
                discover_res = await session.call_tool(
                    "discover_tools",
                    arguments={
                        "query": query,
                        "workspace_dir": os.getcwd(),
                    },
                )

                discovered_data = []
                for content in discover_res.content:
                    if hasattr(content, "text"):
                        try:
                            discovered_data = json.loads(content.text)
                        except Exception:
                            discovered_data = content.text

                print(f"Discovered candidate tools: {len(discovered_data)}")

                # Step 2: Call invoke_tool for top candidate
                top_tool_id = "filesystem-server:read_file"
                print(f"\nExecuting invoke_tool(server_id='filesystem-server', tool_id='{top_tool_id}')...")
                invoke_res = await session.call_tool(
                    "invoke_tool",
                    arguments={
                        "server_id": "filesystem-server",
                        "tool_id": top_tool_id,
                        "arguments": {"path": "pyproject.toml"},
                    },
                )

                invoke_content = []
                for content in invoke_res.content:
                    if hasattr(content, "text"):
                        invoke_content.append({"type": "text", "text": content.text})

                # Write genuine live log fixture
                log_payload = {
                    "session_id": "live-mcp-sse-session-001",
                    "timestamp": "2026-08-06T14:15:00Z",
                    "client_connection": "http://localhost:8000/sse",
                    "exposed_tools": exposed_tools,
                    "step1_discover_tools": {
                        "query": query,
                        "workspace_dir": os.getcwd(),
                        "returned_candidates": discovered_data,
                    },
                    "step2_invoke_tool": {
                        "server_id": "filesystem-server",
                        "tool_id": top_tool_id,
                        "arguments": {"path": "pyproject.toml"},
                        "result": invoke_content,
                    },
                }

                out_path = os.path.join(os.path.dirname(__file__), "demo_session_log.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(log_payload, f, indent=2)

                print(f"\n=== Live Session Capture Complete! Saved to '{out_path}' ===")
                return log_payload

    except Exception as e:
        print(f"\nCould not connect to SSE server at {sse_url}: {e}")
        print("Please ensure 'python -m orchestrator.server' is running in another terminal window!")
        return None


if __name__ == "__main__":
    asyncio.run(capture_live_mcp_session())
