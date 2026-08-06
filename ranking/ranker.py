"""3-Term Weighted Ranking Engine for MCP Orchestrator.

Score = w1 * QuerySim_norm + w2 * ProjMatch_norm + w3 * PastSuccess_norm

Terms:
1. QuerySim_norm: Cosine similarity between query vector and tool vector, clamped to [0,1].
2. ProjMatch_norm: Cosine similarity between project context vector and tool vector, clamped to [0,1].
3. PastSuccess_norm: Historical invocation success rate (cold-start default = 0.5).
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from orchestrator.config import settings
from orchestrator.db import AsyncSessionLocal, get_past_success_rate
from retrieval.embedder import get_embedding_model, get_chroma_client, get_or_create_collections
from ranking.project_context import build_project_context_text, get_project_context_embedding

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors, clamped to [0, 1]."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = float(np.dot(a, b))
    sim = dot_product / (norm_a * norm_b)
    return round(max(0.0, min(1.0, sim)), 4)


async def compute_3term_ranking(
    candidate_tools: List[Dict[str, Any]],
    query: str,
    project_context: Optional[str] = None,
    workspace_path: Optional[str] = None,
    w1: float = settings.RANKING_W1_QUERY_SIM,
    w2: float = settings.RANKING_W2_PROJ_MATCH,
    w3: float = settings.RANKING_W3_PAST_SUCCESS,
    chroma_client=None,
    db_session=None,
) -> List[Dict[str, Any]]:
    """Compute 3-term weighted score and per-term breakdown for candidate tools.

    Handles graceful degradation when project_context is None (redistributes w2 weight).
    Returns re-ranked candidate list sorted by final weighted score descending.
    """
    if not candidate_tools:
        return []

    # 1. Determine active project context vector
    proj_vector: Optional[List[float]] = None
    has_proj_context = False

    if project_context and project_context.strip():
        proj_text = f"Project Context: {project_context.strip()}"
        proj_vector = get_project_context_embedding(proj_text)
        has_proj_context = True
    elif workspace_path:
        proj_text = build_project_context_text(workspace_path)
        proj_vector = get_project_context_embedding(proj_text)
        has_proj_context = True

    # 2. Adjust weights if project context is absent (Graceful Degradation)
    if not has_proj_context:
        w2_eff = 0.0
        total_other = w1 + w3
        if total_other > 0:
            w1_eff = round(w1 / total_other, 4)
            w3_eff = round(w3 / total_other, 4)
        else:
            w1_eff, w3_eff = 0.5, 0.5
    else:
        w1_eff, w2_eff, w3_eff = w1, w2, w3

    # 3. Retrieve tool vectors from ChromaDB for ProjMatch calculation
    client = chroma_client or get_chroma_client()
    _, tools_col = get_or_create_collections(client)

    tool_ids = [t["tool_id"] for t in candidate_tools]
    chroma_records = {}
    if tool_ids and tools_col.count() > 0:
        try:
            fetched = tools_col.get(ids=tool_ids, include=["embeddings"])
            if fetched and fetched.get("ids"):
                for tid, emb in zip(fetched["ids"], fetched["embeddings"]):
                    chroma_records[tid] = emb
        except Exception as e:
            logger.warning(f"Could not fetch tool vectors from ChromaDB: {e}")

    # 4. Score each candidate tool
    ranked_results = []
    
    # Obtain or reuse db_session for PastSuccess lookup
    async def get_success_score(session, tid):
        return await get_past_success_rate(session, tid)

    session_to_use = db_session

    for tool_item in candidate_tools:
        tid = tool_item["tool_id"]

        # Term 1: QuerySim_norm (clamped to [0,1])
        raw_sim = tool_item.get("score", 0.0)
        query_sim_norm = round(max(0.0, min(1.0, float(raw_sim))), 4)

        # Term 2: ProjMatch_norm (clamped to [0,1])
        proj_match_norm = 0.0
        if has_proj_context and proj_vector and tid in chroma_records:
            tool_vector = chroma_records[tid]
            proj_match_norm = cosine_similarity(proj_vector, tool_vector)

        # Term 3: PastSuccess_norm (cold-start = 0.5)
        past_success_norm = 0.5
        if session_to_use:
            try:
                past_success_norm = await get_past_success_rate(session_to_use, tid)
            except Exception as e:
                logger.warning(f"Error fetching past success rate for {tid}: {e}")
                past_success_norm = 0.5
        else:
            try:
                async with AsyncSessionLocal() as session:
                    past_success_norm = await get_past_success_rate(session, tid)
            except Exception:
                past_success_norm = 0.5

        # Final Weighted Score
        final_score = (
            (w1_eff * query_sim_norm)
            + (w2_eff * proj_match_norm)
            + (w3_eff * past_success_norm)
        )
        final_score = round(max(0.0, min(1.0, final_score)), 4)

        ranked_results.append(
            {
                **tool_item,
                "score": final_score,
                "score_breakdown": {
                    "query_sim": query_sim_norm,
                    "proj_match": proj_match_norm,
                    "past_success": past_success_norm,
                    "weights": {
                        "w1": w1_eff,
                        "w2": w2_eff,
                        "w3": w3_eff,
                    },
                },
            }
        )

    # Sort descending by final weighted score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"3-Term Ranking completed for {len(ranked_results)} tools.")
    return ranked_results
