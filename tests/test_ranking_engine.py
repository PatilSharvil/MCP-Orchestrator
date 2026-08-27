"""Unit tests for Phase 4: 3-Term Weighted Ranking Engine.

Verifies:
1. ProjMatch boosting tools matching project context (e.g. GitHub context boosting GitHub tools).
2. PastSuccess cold-start default (0.5) and update after invocation outcomes logged.
3. QuerySim_norm clamping to [0,1] and formula calculation.
4. Tool invocation outcome logging via invoke_tool.
"""

import os
import pytest
import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from orchestrator.db import Base, MCPServer, MCPTool, ToolInvocationOutcome, log_tool_invocation_outcome, get_past_success_rate
from retrieval.embedder import upsert_server_and_tools_embeddings
from ranking.ranker import compute_3term_ranking, cosine_similarity
from ranking.project_context import build_project_context_text, get_project_context_embedding
from orchestrator.server import discover_tools, invoke_tool
from eval.eval_retrieval import seed_eval_mock_data


@pytest.fixture
def mock_chroma_client():
    """Create isolated EphemeralClient seeded with test tools."""
    client = chromadb.EphemeralClient()
    seed_eval_mock_data(client)
    return client


@pytest.mark.asyncio
async def test_projmatch_boosting(mock_chroma_client):
    """Verify ProjMatch boosts GitHub tools when project context contains GitHub dependencies."""
    candidate_tools = [
        {"tool_id": "filesystem-server:read_file", "server_id": "filesystem-server", "name": "read_file", "score": 0.80},
        {"tool_id": "github-server:create_issue", "server_id": "github-server", "name": "create_issue", "score": 0.78},
    ]

    # Project context referencing GitHub API / octokit
    context_gh = "Dependencies: octokit, @octokit/rest, github-api | File Types: .ts, .js"

    # Score with GitHub project context
    ranked_with_context = await compute_3term_ranking(
        candidate_tools=candidate_tools,
        query="file an issue report",
        project_context=context_gh,
        w1=0.33, w2=0.33, w3=0.34,
        chroma_client=mock_chroma_client,
    )

    assert len(ranked_with_context) == 2
    # GitHub issue tool should be boosted to top rank due to ProjMatch term
    assert ranked_with_context[0]["tool_id"] == "github-server:create_issue"
    assert ranked_with_context[0]["score_breakdown"]["proj_match"] > ranked_with_context[1]["score_breakdown"]["proj_match"]


@pytest.mark.asyncio
async def test_past_success_cold_start_and_update():
    """Verify PastSuccess defaults to 0.5 for cold start and updates after invocation logging."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        tool_id = "filesystem-server:read_file"

        # 1. Cold start check
        rate_cold = await get_past_success_rate(session, tool_id)
        assert rate_cold == 0.5

        # 2. Log 3 successful outcomes
        for _ in range(3):
            await log_tool_invocation_outcome(
                session=session,
                tool_id=tool_id,
                server_id="filesystem-server",
                tool_name="read_file",
                success=True,
                latency_ms=12.5,
            )

        rate_updated = await get_past_success_rate(session, tool_id)
        assert rate_updated == 1.0

        # 3. Log 1 failed outcome
        await log_tool_invocation_outcome(
            session=session,
            tool_id=tool_id,
            server_id="filesystem-server",
            tool_name="read_file",
            success=False,
            error_message="File not found",
            latency_ms=8.0,
        )

        rate_mixed = await get_past_success_rate(session, tool_id)
        assert rate_mixed == 0.75

    await engine.dispose()


@pytest.mark.asyncio
async def test_clamped_query_sim_and_weighted_formula(mock_chroma_client):
    """Verify QuerySim_norm is clamped to [0,1] and formula calculates correctly."""
    candidate_tools = [
        {"tool_id": "filesystem-server:read_file", "server_id": "filesystem-server", "name": "read_file", "score": -0.5},
    ]

    # Without project context (graceful degradation: w1=0.49, w2=0, w3=0.51)
    ranked = await compute_3term_ranking(
        candidate_tools=candidate_tools,
        query="test query",
        project_context=None,
        w1=0.33, w2=0.33, w3=0.34,
        chroma_client=mock_chroma_client,
    )

    breakdown = ranked[0]["score_breakdown"]
    # Clamped QuerySim_norm must be >= 0.0
    assert breakdown["query_sim"] == 0.0
    assert 0.0 <= ranked[0]["score"] <= 1.0


@pytest.mark.asyncio
async def test_tool_invocation_outcome_logging(monkeypatch):
    """Verify invoke_tool records execution outcome into tool_invocation_outcomes table."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        # Insert DB records for filesystem server
        server = MCPServer(
            id="filesystem-server",
            name="Filesystem MCP Server",
            transport_type="stdio",
            connection_config={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        )
        tool = MCPTool(
            id="filesystem-server:read_file",
            server_id="filesystem-server",
            name="read_file",
            description="Read file",
            input_schema={"properties": {"path": {}}},
        )
        session.add(server)
        session.add(tool)
        await session.commit()

        class TestSessionContextManager:
            async def __aenter__(self):
                return session
            async def __aexit__(self, exc_type, exc, tb):
                pass

        monkeypatch.setattr("orchestrator.server.AsyncSessionLocal", lambda: TestSessionContextManager())

        res = await invoke_tool(
            server_id="filesystem-server",
            tool_id="read_file",
            arguments={"path": os.path.abspath("pyproject.toml")},
        )

        assert res.get("isError") is False

        # Query tool_invocation_outcomes table
        outcomes_res = await session.execute(select(ToolInvocationOutcome))
        outcomes = outcomes_res.scalars().all()

        assert len(outcomes) == 1
        record = outcomes[0]
        assert record.tool_id == "filesystem-server:read_file"
        assert record.success is True
        assert record.latency_ms is not None
        assert record.latency_ms > 0

    await engine.dispose()
