"""Unit tests for minimum relevance threshold filtering."""

import pytest
import json
import os
from orchestrator.config import settings
from orchestrator.server import discover_tools

@pytest.mark.asyncio
async def test_generic_conversational_queries_return_empty():
    """Verify that non-tool conversational queries return empty tool lists."""
    conversational_queries = [
        "hi",
        "hello",
        "what is MCP?",
    ]
    for query in conversational_queries:
        tools = await discover_tools(query=query, workspace_dir=".", bypass=False)
        assert tools == [], f"Query '{query}' should return empty list, but got {len(tools)} tools"

@pytest.mark.asyncio
async def test_all_20_benchmark_tasks_return_non_empty():
    """Verify that all 20 real benchmark task queries return non-empty ranked candidate sets."""
    benchmark_path = os.path.join("eval", "benchmark_tasks.json")
    assert os.path.exists(benchmark_path), "eval/benchmark_tasks.json must exist"
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    assert len(tasks) == 20, "Should contain exactly 20 benchmark tasks"
    
    for task in tasks:
        query = task["query"]
        task_id = task["task_id"]
        tools = await discover_tools(query=query, workspace_dir=".", bypass=False)
        assert len(tools) > 0, f"Task '{task_id}' ('{query}') was incorrectly suppressed to 0 tools!"
