"""Unit and integration tests for Phase 7 Ablation modes.

Verifies that each ablation mode is correctly isolated and functions as designed.
"""

import pytest
import numpy as np
from sqlalchemy import select

from orchestrator.config import settings
from orchestrator.db import AsyncSessionLocal, MCPTool, MCPServer
from ranking.ranker import compute_3term_ranking, get_mismatched_space_embedding, VALID_ABLATION_MODES
from retrieval.retriever import stage_1_retrieve_servers, stage_2_retrieve_tools


def test_valid_ablation_mode_set():
    """Verify that VALID_ABLATION_MODES is set correctly."""
    assert "no_projmatch" in VALID_ABLATION_MODES
    assert "no_ranking" in VALID_ABLATION_MODES
    assert "no_shared_space" in VALID_ABLATION_MODES
    assert "no_topk" in VALID_ABLATION_MODES
    assert "stage1_lexical_only" in VALID_ABLATION_MODES


@pytest.mark.asyncio
async def test_no_projmatch_ablation():
    """Verify that no_projmatch sets w2=0 and renormalizes w1 and w3."""
    candidates = [
        {"tool_id": "filesystem-server:read_file", "score": 0.8},
        {"tool_id": "filesystem-server:write_file", "score": 0.6},
    ]
    # We pass a non-empty project context, which normally triggers w2 calculation
    ranked = await compute_3term_ranking(
        candidate_tools=candidates,
        query="read file",
        project_context="Dependencies: python",
        w1=0.4, w2=0.3, w3=0.3,
        ablation_mode="no_projmatch",
    )
    
    assert len(ranked) == 2
    for r in ranked:
        bd = r["score_breakdown"]
        # Ensure w2 is forced to 0
        assert bd["weights"]["w2"] == 0.0
        # Check renormalization of w1 and w3: 0.4 / (0.4 + 0.3) = 0.5714, 0.3 / 0.7 = 0.4286
        assert abs(bd["weights"]["w1"] - 0.5714) < 1e-3
        assert abs(bd["weights"]["w3"] - 0.4286) < 1e-3
        # ProjMatch value must be 0
        assert bd["proj_match"] == 0.0


@pytest.mark.asyncio
async def test_no_ranking_ablation():
    """Verify that no_ranking bypasses 3-term weights and returns raw query similarity only."""
    candidates = [
        {"tool_id": "filesystem-server:read_file", "score": 0.9},
        {"tool_id": "filesystem-server:write_file", "score": 0.4},
    ]
    ranked = await compute_3term_ranking(
        candidate_tools=candidates,
        query="read file",
        project_context="Dependencies: python",
        w1=0.3, w2=0.3, w3=0.4,
        ablation_mode="no_ranking",
    )

    assert len(ranked) == 2
    assert ranked[0]["score"] == 0.9
    assert ranked[1]["score"] == 0.4
    
    bd = ranked[0]["score_breakdown"]
    assert bd["weights"]["w1"] == 1.0
    assert bd["weights"]["w2"] == 0.0
    assert bd["weights"]["w3"] == 0.0
    assert bd["ablation_mode"] == "no_ranking"


def test_no_shared_space_mismatched_embedding():
    """Verify that no_shared_space creates a mismatched embedding space."""
    text = "Dependencies: aiosqlite, pytest"
    vec1 = get_mismatched_space_embedding(text, dim=384)
    vec2 = get_mismatched_space_embedding(text, dim=384)
    
    # Check dimensionality
    assert len(vec1) == 384
    # Check unit vector length
    assert abs(np.linalg.norm(vec1) - 1.0) < 1e-5
    # Check deterministic generation
    assert np.allclose(vec1, vec2)
    
    # Check that another text produces a different vector
    vec3 = get_mismatched_space_embedding("Different text context", dim=384)
    dot_prod = np.dot(vec1, vec3)
    # They should be largely uncorrelated in high dimensions
    assert abs(dot_prod) < 0.5


@pytest.mark.asyncio
async def test_no_topk_retrieval_bypass(monkeypatch):
    """Verify that no_topk sets top_n and top_m limits to maximum size to disable Top-K truncation."""
    # We mock the ChromaDB queries to assert that the n_results/top_n is bypassed
    stage1_called_with_n = None
    stage2_called_with_m = None

    class MockCollection:
        def count(self):
            return 2000
        def query(self, query_embeddings, n_results, include=None, where=None):
            nonlocal stage1_called_with_n, stage2_called_with_m
            if include and "documents" in include:
                stage2_called_with_m = n_results
                return {"ids": [["server1:tool1"]], "metadatas": [[{"server_id": "server1", "name": "tool1"}]], "distances": [[0.1]], "documents": [["doc"]]}
            else:
                stage1_called_with_n = n_results
                return {"ids": [["server1"]], "metadatas": [[{"name": "server1"}]], "distances": [[0.1]]}
        def get(self, ids=None, include=None):
            return {"ids": [], "documents": [], "metadatas": []}

    def mock_collections(client):
        return MockCollection(), MockCollection()

    monkeypatch.setattr("retrieval.retriever.get_or_create_collections", mock_collections)
    monkeypatch.setattr("retrieval.retriever.get_embedding_model", lambda: type('Model', (), {'encode': lambda self, q: np.zeros((1, 384))})())

    # Call Stage 1 with no_topk
    stage_1_retrieve_servers("test query", ablation_mode="no_topk")
    assert stage1_called_with_n == 100  # Bypassed to large limit

    # Call Stage 2 with no_topk
    stage_2_retrieve_tools("test query", ["server1"], ablation_mode="no_topk")
    assert stage2_called_with_m == 1000  # Bypassed to large limit


@pytest.mark.asyncio
async def test_invalid_ablation_mode_raises():
    """Verify that unrecognised ablation_mode strings raise ValueError."""
    candidates = [{"tool_id": "fs:read", "score": 0.5}]
    with pytest.raises(ValueError, match="Unknown ablation_mode"):
        await compute_3term_ranking(
            candidate_tools=candidates,
            query="test",
            ablation_mode="invalid_ablation_string",
        )
