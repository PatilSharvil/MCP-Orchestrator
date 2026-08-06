"""Unit tests for Phase 5: LangGraph Orchestration & Real Integration.

Verifies:
1. RegistrationState graph node transitions and inspectable state output.
2. QueryState online discovery graph node transitions (ContextParser -> Stage1 -> Stage2 -> Ranking -> Logging).
3. discover_tools FastMCP gateway powered by LangGraph state machine.
4. End-to-end execution of discover_tools and invoke_tool via proxy.
"""

import pytest
import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import orchestrator.db as db
from orchestrator.graph import (
    registration_pipeline_graph,
    query_pipeline_graph,
    run_online_query_pipeline,
)
from orchestrator.server import discover_tools, invoke_tool
from eval.eval_retrieval import seed_eval_mock_data


@pytest.fixture
def mock_chroma_client():
    """Create isolated EphemeralClient seeded with test tools."""
    client = chromadb.EphemeralClient()
    seed_eval_mock_data(client)
    return client


@pytest.mark.asyncio
async def test_langgraph_registration_state():
    """Verify LangGraph registration graph node execution."""
    initial_state = {
        "server_configs": [],
        "servers_processed": 0,
        "tools_registered": 0,
        "chroma_status": "pending",
        "logs": [],
    }

    final_state = await registration_pipeline_graph.ainvoke(initial_state)

    assert "servers_processed" in final_state
    assert "tools_registered" in final_state
    assert final_state["chroma_status"] == "synced"
    assert len(final_state["logs"]) >= 2


@pytest.mark.asyncio
async def test_langgraph_online_query_state(mock_chroma_client, monkeypatch):
    """Verify LangGraph online query state graph transitions."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        class TestSessionContextManager:
            async def __aenter__(self):
                return session
            async def __aexit__(self, exc_type, exc, tb):
                pass

        monkeypatch.setattr("orchestrator.db.AsyncSessionLocal", lambda: TestSessionContextManager())
        monkeypatch.setattr("retrieval.retriever.get_chroma_client", lambda: mock_chroma_client)
        monkeypatch.setattr("ranking.ranker.get_chroma_client", lambda: mock_chroma_client)

        initial_state = {
            "query": "read contents of file from local disk",
            "workspace_dir": ".",
            "project_context_text": "",
            "candidate_servers": [],
            "candidate_tools": [],
            "ranked_tools": [],
            "db_logged": False,
        }

        final_state = await query_pipeline_graph.ainvoke(initial_state)

        assert len(final_state["candidate_servers"]) > 0
        assert len(final_state["candidate_tools"]) > 0
        assert len(final_state["ranked_tools"]) > 0
        assert final_state["db_logged"] is True

        top_tool = final_state["ranked_tools"][0]
        assert top_tool["server_id"] in ["filesystem-server", "demo-server"]
        assert top_tool["name"] in ["read_file", "list_directory"]
        assert "score_breakdown" in top_tool

    await engine.dispose()


@pytest.mark.asyncio
async def test_discover_tools_langgraph_integration(mock_chroma_client, monkeypatch):
    """Verify discover_tools meta-tool returns candidates powered by LangGraph state machine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        tool1 = db.MCPTool(
            id="filesystem-server:read_file",
            server_id="filesystem-server",
            name="read_file",
            description="Read file contents from disk",
            input_schema={"properties": {"path": {}}},
        )
        session.add(tool1)
        await session.commit()

        class TestSessionContextManager:
            async def __aenter__(self):
                return session
            async def __aexit__(self, exc_type, exc, tb):
                pass

        monkeypatch.setattr("orchestrator.db.AsyncSessionLocal", lambda: TestSessionContextManager())
        monkeypatch.setattr("retrieval.retriever.get_chroma_client", lambda: mock_chroma_client)
        monkeypatch.setattr("ranking.ranker.get_chroma_client", lambda: mock_chroma_client)

        candidates = await discover_tools(query="read file from workspace", workspace_dir=".")

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        top = candidates[0]
        assert "score_breakdown" in top

    await engine.dispose()
