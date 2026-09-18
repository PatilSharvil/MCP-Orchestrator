"""Tests for FastMCP gateway stdio transport, lazy vs pre-warmed execution, and tool routing."""

import os
import pytest
from orchestrator.server import invoke_tool, discover_tools


@pytest.mark.asyncio
async def test_invoke_tool_decoupled_execution():
    """Verify that invoke_tool executes immediately without heavy embedding initialization."""
    pyproject_path = os.path.abspath("pyproject.toml")
    res = await invoke_tool(
        server_id="filesystem",
        tool_id="get_file_info",
        arguments={"path": pyproject_path},
    )
    assert isinstance(res, dict)
    assert res.get("isError") is False
    assert "content" in res
    assert len(res["content"]) > 0
    assert any("size:" in str(c) for c in res["content"])


@pytest.mark.asyncio
async def test_invoke_tool_llm_parameter_tolerance():
    """Verify that invoke_tool tolerates LLM parameter wrapping."""
    pyproject_path = os.path.abspath("pyproject.toml")
    res = await invoke_tool(
        server_id="",
        tool_id="",
        arguments={"server_id": "filesystem", "tool_id": "get_file_info", "path": pyproject_path},
    )
    assert isinstance(res, dict)
    assert res.get("isError") is False
    assert "content" in res


@pytest.mark.asyncio
async def test_discover_tools_pipeline():
    """Verify discover_tools retrieves relevant tools via LangGraph online pipeline."""
    tools = await discover_tools(query="read and write files on disk")
    assert isinstance(tools, list)
    assert len(tools) > 0
    tool_ids = [t["tool_id"] for t in tools]
    assert any("filesystem" in tid for tid in tool_ids)


@pytest.mark.asyncio
async def test_discover_tools_bypass_mode():
    """Verify discover_tools in bypass mode returns all registered tools without pipeline delay."""
    tools = await discover_tools(query="", bypass=True)
    assert isinstance(tools, list)
    assert len(tools) > 0
    for t in tools:
        assert t.get("bypass") is True
