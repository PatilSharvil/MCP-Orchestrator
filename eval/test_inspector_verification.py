"""Verification Script for Part A (workspace_dir bug fix & ProjMatch non-zero cosine score) & Part B (invoke_tool end-to-end + database outcome logging)."""

import os
import tempfile
import asyncio
import json
import logging
from sqlalchemy import select

from orchestrator.db import AsyncSessionLocal, ToolInvocationOutcome, QueryLog, init_db
from orchestrator.server import discover_tools, invoke_tool
from retrieval.embedder import sync_all_embeddings_from_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_verification():
    await init_db()

    # Sync all database tool & server embeddings into ChromaDB vector store
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    print("=========================================================")
    print("PART A (1): Testing discover_tools with directory having NO dependency files (Graceful Degradation)")
    print("=========================================================")
    with tempfile.TemporaryDirectory() as empty_temp_dir:
        res_empty_ws = await discover_tools(query="read project configuration file", workspace_dir=empty_temp_dir)
        print(f"Candidates returned: {len(res_empty_ws)}")
        assert len(res_empty_ws) > 0, "Expected candidate tools returned even without project context!"
        top = res_empty_ws[0]
        print(f"Top Candidate: {top['tool_id']} | Score: {top['score']}")
        print(f"Score Breakdown: {json.dumps(top['score_breakdown'], indent=2)}")
        assert top['score_breakdown']['weights']['w2'] == 0.0, "Expected w2=0.0 on graceful degradation!"
        assert top['score_breakdown']['proj_match'] == 0.0, "Expected proj_match=0.0 on graceful degradation!"

    print("\n=========================================================")
    print("PART A (2): Testing discover_tools with REAL workspace_dir (Full 3-Term Scoring & Non-Zero ProjMatch)")
    print("=========================================================")
    real_ws = "c:/Users/patil/OneDrive/Desktop/Projects/MCP Orchestrator"
    query_proj = "search github repository for pull requests and issues"
    res_real_ws = await discover_tools(query=query_proj, workspace_dir=real_ws)
    print(f"Candidates returned: {len(res_real_ws)}")
    assert len(res_real_ws) > 0
    top_real = res_real_ws[0]
    print(f"Top Candidate: {top_real['tool_id']} | Score: {top_real['score']}")
    print(f"Score Breakdown: {json.dumps(top_real['score_breakdown'], indent=2)}")
    assert top_real['score_breakdown']['weights']['w2'] == 0.33, "Expected w2=0.33 with real workspace_dir!"
    assert top_real['score_breakdown']['proj_match'] > 0.0, f"Expected non-zero proj_match cosine score! Got: {top_real['score_breakdown']['proj_match']}"

    print("\n=========================================================")
    print("PART B (1): Testing invoke_tool SUCCESS case")
    print("=========================================================")
    inv_success = await invoke_tool(
        server_id="filesystem-server",
        tool_id="read_file",
        arguments={"path": "pyproject.toml"},
    )
    print(f"invoke_tool Success Result: {json.dumps(inv_success, indent=2)[:300]}...")
    assert inv_success.get("isError") is False, "Expected isError=False for valid path!"

    print("\nChecking PostgreSQL/SQLite 'tool_invocation_outcomes' DB table for success log...")
    async with AsyncSessionLocal() as session:
        stmt = select(ToolInvocationOutcome).where(
            ToolInvocationOutcome.tool_id == "filesystem-server:read_file",
            ToolInvocationOutcome.success == True,
        ).order_by(ToolInvocationOutcome.id.desc())
        res_db = await session.execute(stmt)
        record_success = res_db.scalars().first()
        assert record_success is not None, "DB success record must be present!"
        print(f"DB Success Record Found: id={record_success.id}, tool_id='{record_success.tool_id}', success={record_success.success}, latency_ms={record_success.latency_ms}, error_message={record_success.error_message}")
        assert record_success.success is True
        assert record_success.latency_ms > 0
        assert record_success.error_message is None

    print("\n=========================================================")
    print("PART B (2): Testing invoke_tool FAILURE case")
    print("=========================================================")
    inv_failure = await invoke_tool(
        server_id="filesystem-server",
        tool_id="read_file",
        arguments={"path": "non_existent_file_xyz_123.txt"},
    )
    print(f"invoke_tool Failure Result: {json.dumps(inv_failure, indent=2)}")
    assert inv_failure.get("isError") is True, "Expected isError=True for non-existent file!"

    print("\nChecking PostgreSQL/SQLite 'tool_invocation_outcomes' DB table for failure log...")
    async with AsyncSessionLocal() as session:
        stmt = select(ToolInvocationOutcome).where(
            ToolInvocationOutcome.tool_id == "filesystem-server:read_file",
            ToolInvocationOutcome.success == False,
        ).order_by(ToolInvocationOutcome.id.desc())
        res_db_fail = await session.execute(stmt)
        record_failure = res_db_fail.scalars().first()
        assert record_failure is not None, "DB failure record must be present!"
        print(f"DB Failure Record Found: id={record_failure.id}, tool_id='{record_failure.tool_id}', success={record_failure.success}, latency_ms={record_failure.latency_ms}, error_message='{record_failure.error_message}'")
        assert record_failure.success is False
        assert record_failure.latency_ms > 0
        assert record_failure.error_message is not None

    print("\n=========================================================")
    print("ALL PART A AND PART B VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(run_verification())
