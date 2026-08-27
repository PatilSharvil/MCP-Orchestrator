"""Tests for MCP Orchestrator Phase 0 and Phase 1.

Verifies:
1. Client-side tool discovery from standard Filesystem MCP server via stdio.
2. Database schema initialization and idempotent registration in PostgreSQL / SQLite.
3. FastMCP server meta-tools (`discover_tools` and `invoke_tool`).
"""

import os
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from orchestrator.config import DEFAULT_DOWNSTREAM_SERVERS
import orchestrator.db as db
from orchestrator.client import DownstreamMCPClient
from orchestrator.registrar import register_downstream_servers
from orchestrator.server import discover_tools, invoke_tool


@pytest_asyncio.fixture
async def test_session():
    """Create an isolated in-memory SQLite database session for unit testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with SessionLocal() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_downstream_filesystem_client():
    """Test client-side connection to standard Filesystem MCP server via stdio."""
    fs_cfg = DEFAULT_DOWNSTREAM_SERVERS[0]
    tools = await DownstreamMCPClient.fetch_tools(fs_cfg)

    assert isinstance(tools, list)
    assert len(tools) > 0

    tool_names = [t["name"] for t in tools]
    assert "read_file" in tool_names or "list_directory" in tool_names
    print(f"Discovered filesystem tools: {tool_names}")


@pytest.mark.asyncio
async def test_idempotent_registration(test_session: AsyncSession):
    """Verify registration registers tools and running it twice is idempotent."""
    # First registration run
    stats1 = await register_downstream_servers(test_session, DEFAULT_DOWNSTREAM_SERVERS)
    assert stats1["servers_processed"] >= 1
    assert stats1["tools_registered"] > 0

    # Verify database records
    res_servers = await test_session.execute(select(db.MCPServer))
    servers = res_servers.scalars().all()
    assert len(servers) == stats1["servers_processed"]

    res_tools = await test_session.execute(select(db.MCPTool))
    tools = res_tools.scalars().all()
    initial_tool_count = len(tools)
    assert initial_tool_count > 0

    # Second registration run (idempotency check)
    stats2 = await register_downstream_servers(test_session, DEFAULT_DOWNSTREAM_SERVERS)
    assert stats2["servers_processed"] == stats1["servers_processed"]
    assert stats2["tools_registered"] == initial_tool_count

    res_tools_after = await test_session.execute(select(db.MCPTool))
    tools_after = res_tools_after.scalars().all()
    assert len(tools_after) == initial_tool_count


@pytest.mark.asyncio
async def test_discover_tools_sql(test_session: AsyncSession, monkeypatch):
    """Test discover_tools meta-tool."""
    # Register downstream tools into DB
    await register_downstream_servers(test_session, DEFAULT_DOWNSTREAM_SERVERS)

    class TestSessionContextManager:
        async def __aenter__(self):
            return test_session
        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("orchestrator.db.AsyncSessionLocal", lambda: TestSessionContextManager())

    # Discover with query matching read
    results = await discover_tools(query="read")
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_invoke_tool_execution():
    """Test invoke_tool proxying execution to downstream filesystem server."""
    res = await invoke_tool(
        server_id="filesystem-server",
        tool_id="read_file",
        arguments={"path": os.path.abspath("pyproject.toml")},
    )

    assert isinstance(res, dict)
    assert res.get("isError") is False
    assert "content" in res
    assert len(res["content"]) > 0
    assert "mcp-orchestrator" in res["content"][0]["text"]
