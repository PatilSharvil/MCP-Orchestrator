"""Unit tests for Phase 6: Baseline Mode + Benchmark Harness.

Verifies:
1. Bypass mode returning full unranked tool set.
2. Token counting via tiktoken cl100k_base.
3. Precision, Recall, F1@K calculations.
4. Benchmark runner producing valid metric comparison reports.
"""

import pytest
import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import orchestrator.db as db
from orchestrator.server import discover_tools
from retrieval.embedder import reset_chroma_client
from eval.run_benchmark import count_tokens, compute_schema_tokens, compute_precision_recall_f1, run_benchmark_task_suite
from eval.eval_retrieval import seed_eval_mock_data


@pytest.fixture
def mock_chroma_client():
    """Create isolated EphemeralClient seeded with test tools."""
    client = chromadb.EphemeralClient()
    reset_chroma_client(client)
    seed_eval_mock_data(client)
    return client


@pytest.mark.asyncio
async def test_bypass_mode_unranked_tools(monkeypatch):
    """Verify discover_tools(bypass=True) returns all unranked tools from database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        for i in range(5):
            t = db.MCPTool(
                id=f"server-1:tool_{i}",
                server_id="server-1",
                name=f"tool_{i}",
                description=f"Description for tool {i}",
                input_schema={"properties": {}},
            )
            session.add(t)
        await session.commit()

        class TestSessionContextManager:
            async def __aenter__(self):
                return session
            async def __aexit__(self, exc_type, exc, tb):
                pass

        monkeypatch.setattr("orchestrator.db.AsyncSessionLocal", lambda: TestSessionContextManager())
        monkeypatch.setattr("orchestrator.server.AsyncSessionLocal", lambda: TestSessionContextManager())

        tools = await discover_tools(query="test", bypass=True)

        assert isinstance(tools, list)
        assert len(tools) == 5
        for t in tools:
            assert t["bypass"] is True
            assert t["score"] == 1.0

    await engine.dispose()


def test_token_counting_tiktoken():
    """Verify tiktoken token counting."""
    sample = "The quick brown fox jumps over the lazy dog."
    cnt = count_tokens(sample)
    assert isinstance(cnt, int)
    assert cnt > 0


def test_precision_recall_f1_calculation():
    """Verify precision, recall, and F1 calculations."""
    returned = ["tool_a", "tool_b", "tool_c"]
    ground_truth = ["tool_a", "tool_b"]

    p, r, f1 = compute_precision_recall_f1(returned, ground_truth)
    assert p == 0.6667
    assert r == 1.0
    assert f1 == 0.8


@pytest.mark.asyncio
async def test_benchmark_runner_sample_execution(mock_chroma_client, monkeypatch):
    """Verify run_benchmark_task_suite returns valid report data structure."""
    monkeypatch.setattr("retrieval.retriever.get_chroma_client", lambda: mock_chroma_client)
    monkeypatch.setattr("ranking.ranker.get_chroma_client", lambda: mock_chroma_client)

    report = await run_benchmark_task_suite()

    assert "num_tasks" in report
    assert report["num_tasks"] == 20
    assert "mode_a_bypass" in report
    assert "mode_b_orchestrator" in report
    assert "improvements" in report
    assert "token_reduction_pct" in report["improvements"]
