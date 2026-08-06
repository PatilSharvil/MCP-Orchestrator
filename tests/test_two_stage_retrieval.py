"""Unit tests for Phase 3: Two-Stage Vector Retrieval.

Verifies:
1. Stage 1 server-level vector retrieval narrows down candidate server IDs.
2. Stage 2 tool-level retrieval uses metadata filtering to restrict tools to candidate server IDs.
3. discover_tools FastMCP tool returns candidate tools end-to-end with similarity scores.
4. Evaluation hit-rate checkpoint runs deterministically against EphemeralClient test fixtures.
"""

import pytest
import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import orchestrator.db as db
from retrieval.embedder import get_or_create_collections, upsert_server_and_tools_embeddings
from retrieval.retriever import stage_1_retrieve_servers, stage_2_retrieve_tools, two_stage_retrieve
from orchestrator.server import discover_tools
from eval.eval_retrieval import load_ground_truth_dataset, evaluate_retrieval_hit_rate, seed_eval_mock_data


@pytest.fixture
def mock_chroma_client():
    """Create an isolated in-memory EphemeralClient with multi-server test fixtures."""
    client = chromadb.EphemeralClient()
    seed_eval_mock_data(client)
    return client


@pytest.mark.asyncio
async def test_stage_1_server_retrieval(mock_chroma_client):
    """Verify Stage 1 correctly narrows query to relevant candidate servers."""
    query_gh = "open a pull request on github repository"
    servers_gh = stage_1_retrieve_servers(query_gh, top_n=2, chroma_client=mock_chroma_client)

    assert len(servers_gh) <= 2
    server_ids = [s["server_id"] for s in servers_gh]
    assert "github-server" in server_ids

    query_fs = "read text file from local folder"
    servers_fs = stage_1_retrieve_servers(query_fs, top_n=2, chroma_client=mock_chroma_client)

    server_ids_fs = [s["server_id"] for s in servers_fs]
    assert "filesystem-server" in server_ids_fs


@pytest.mark.asyncio
async def test_stage_2_tool_metadata_filtering(mock_chroma_client):
    """Verify Stage 2 restricts tool search ONLY to specified candidate server_ids."""
    query = "read contents"
    
    tools_gh_only = stage_2_retrieve_tools(
        query=query,
        candidate_server_ids=["github-server"],
        top_m=5,
        chroma_client=mock_chroma_client,
    )

    assert len(tools_gh_only) > 0
    for t in tools_gh_only:
        assert t["server_id"] == "github-server"
        assert t["tool_id"].startswith("github-server:")

    tools_fs_only = stage_2_retrieve_tools(
        query=query,
        candidate_server_ids=["filesystem-server"],
        top_m=5,
        chroma_client=mock_chroma_client,
    )

    assert len(tools_fs_only) > 0
    for t in tools_fs_only:
        assert t["server_id"] == "filesystem-server"
        assert t["tool_id"].startswith("filesystem-server:")


@pytest.mark.asyncio
async def test_two_stage_retrieve_end_to_end(mock_chroma_client):
    """Verify complete Two-Stage Retrieval pipeline ranks expected tool highest."""
    query = "search knowledge graph for saved entities"
    results = two_stage_retrieve(query=query, top_n=2, top_m=5, chroma_client=mock_chroma_client)

    assert len(results) > 0
    top_tool = results[0]
    assert top_tool["server_id"] == "memory-server"
    assert top_tool["name"] in ["search_nodes", "read_graph"]
    assert "score" in top_tool
    assert 0.0 <= top_tool["score"] <= 1.0


@pytest.mark.asyncio
async def test_discover_tools_server_integration(mock_chroma_client, monkeypatch):
    """Verify discover_tools meta-tool returns candidates with full database details."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        tool1 = db.MCPTool(
            id="filesystem-server:read_file",
            server_id="filesystem-server",
            name="read_file",
            description="Read file contents from local disk storage",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
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

        candidates = await discover_tools(query="read a file from disk")

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        
        match = [c for c in candidates if c["tool_id"] == "filesystem-server:read_file"]
        assert len(match) == 1
        assert match[0]["name"] == "read_file"
        assert match[0]["description"] == "Read file contents from local disk storage"
        assert "score" in match[0]

    await engine.dispose()


@pytest.mark.asyncio
async def test_retrieval_evaluation_hit_rate_checkpoint(mock_chroma_client):
    """Sanity checkpoint evaluating retrieval hit-rate on ground-truth dataset."""
    dataset = load_ground_truth_dataset()
    results = evaluate_retrieval_hit_rate(
        dataset=dataset, top_n=2, top_m=5, chroma_client=mock_chroma_client
    )

    assert results["total_queries"] == len(dataset)
    assert results["stage1_hit_rate"] >= 0.80
    assert results["stage2_hit_rate_at_m"] >= 0.85
