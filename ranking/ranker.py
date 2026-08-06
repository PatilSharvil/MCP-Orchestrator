"""3-Term Weighted Ranking Engine.

Formula:
  Score = w1 * QuerySim_norm + w2 * ProjMatch_norm + w3 * PastSuccess_norm

Where:
  - QuerySim_norm: Cosine similarity between query vector and tool vector, strictly clamped to [0, 1].
  - ProjMatch_norm: Cosine similarity between project context vector (dependencies/file types) and tool vector.
  - PastSuccess_norm: Historical invocation success rate derived from PostgreSQL (default cold start = 0.5).

Graceful Degradation:
  If project_context is missing or empty, w2 is set to 0 and weights w1 and w3 are re-normalized:
    w1_eff = w1 / (w1 + w3)
    w3_eff = w3 / (w1 + w3)
"""

import math
import logging
import hashlib
import random
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.config import settings
from orchestrator.db import AsyncSessionLocal, get_past_success_rate
from retrieval.embedder import get_embedding_model, get_or_create_collections, get_chroma_client
from ranking.project_context import get_project_context_embedding, build_project_context_text

logger = logging.getLogger(__name__)

# Valid ablation modes in Phase 7
VALID_ABLATION_MODES = frozenset({
    "full",
    "no_projmatch",
    "no_ranking",
    "no_shared_space",
    "no_topk",
    "stage1_lexical_only",
})


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors, strictly clamped to [0.0, 1.0]."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, float(sim)))


def get_mismatched_space_embedding(text: str, dim: int = 384) -> List[float]:
    """Generate a deterministic unit vector in a mismatched random space based on the input text.
    
    This simulates embedding the text using a completely unrelated vector space.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(h, "big"))
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


async def compute_3term_ranking(
    candidate_tools: List[Dict[str, Any]],
    query: str,
    project_context: Optional[str] = None,
    workspace_path: Optional[str] = None,
    w1: float = settings.RANKING_W1_QUERY_SIM,
    w2: float = settings.RANKING_W2_PROJ_MATCH,
    w3: float = settings.RANKING_W3_PAST_SUCCESS,
    db_session: Optional[AsyncSession] = None,
    chroma_client: Optional[Any] = None,
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute 3-term weighted scores for candidate tools and return sorted tools with score breakdowns.

    Args:
        candidate_tools: Candidate tools to rank.
        query: User query string.
        project_context: Text description of project context.
        workspace_path: Path to workspace for context extraction.
        w1, w2, w3: Base weights.
        db_session: Optional session for past success rate lookup.
        chroma_client: Optional client for ChromaDB.
        ablation_mode: Optional string indicating ablation mode.
    """
    if not candidate_tools:
        return []

    mode = (ablation_mode or "full").lower()
    if mode not in VALID_ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode '{ablation_mode}'. Must be one of {sorted(VALID_ABLATION_MODES)}."
        )

    # 1. Check for "no_ranking" ablation mode
    if mode == "no_ranking":
        scored_candidates = []
        for tool in candidate_tools:
            raw_score = max(0.0, min(1.0, float(tool.get("score", 0.0))))
            enriched_tool = dict(tool)
            enriched_tool["score"] = raw_score
            enriched_tool["score_breakdown"] = {
                "query_sim": raw_score,
                "proj_match": 0.0,
                "past_success": 0.0,
                "weights": {
                    "w1": 1.0,
                    "w2": 0.0,
                    "w3": 0.0,
                },
                "ablation_mode": "no_ranking",
            }
            scored_candidates.append(enriched_tool)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates

    # 2. Determine if project context is available
    proj_vector = []
    has_proj_context = False

    if mode != "no_projmatch":
        if project_context and isinstance(project_context, str) and project_context.strip():
            clean_ctx = project_context.strip()
            if clean_ctx != "Project Context:":
                if mode == "no_shared_space":
                    proj_vector = get_mismatched_space_embedding(clean_ctx)
                else:
                    proj_vector = get_project_context_embedding(clean_ctx)
                if proj_vector:
                    has_proj_context = True

        if not has_proj_context and workspace_path:
            ctx_text = build_project_context_text(workspace_path)
            if ctx_text and ctx_text.strip():
                if mode == "no_shared_space":
                    proj_vector = get_mismatched_space_embedding(ctx_text)
                else:
                    proj_vector = get_project_context_embedding(ctx_text)
                if proj_vector:
                    has_proj_context = True

    # Calculate effective weights
    if has_proj_context:
        w1_eff, w2_eff, w3_eff = w1, w2, w3
    else:
        # Graceful degradation: set w2 = 0, re-normalize w1 and w3
        sum_w = w1 + w3
        if sum_w > 0:
            w1_eff = round(w1 / sum_w, 4)
            w2_eff = 0.0
            w3_eff = round(w3 / sum_w, 4)
        else:
            w1_eff, w2_eff, w3_eff = 0.5, 0.0, 0.5

    # 3. Fetch tool embeddings from ChromaDB 'tools' collection
    client = chroma_client or get_chroma_client()
    _, tools_col = get_or_create_collections(client=client)

    tool_ids = [t["tool_id"] for t in candidate_tools]
    chroma_records = {}
    try:
        fetched = tools_col.get(ids=tool_ids, include=["embeddings"])
        if fetched and fetched.get("ids"):
            for tid, emb in zip(fetched["ids"], fetched.get("embeddings", [])):
                if emb is not None:
                    chroma_records[tid] = list(emb)
    except Exception as e:
        logger.warning(f"Could not fetch tool embeddings from ChromaDB: {e}")

    # 4. Score each candidate tool
    scored_candidates = []

    for tool in candidate_tools:
        tid = tool["tool_id"]

        # Term 1: QuerySim_norm (clamped to [0,1])
        raw_score = tool.get("score", 0.0)
        query_sim_norm = max(0.0, min(1.0, float(raw_score)))

        # Term 2: ProjMatch_norm
        proj_match_norm = 0.0
        if has_proj_context and tid in chroma_records:
            tool_emb = chroma_records[tid]
            proj_match_norm = cosine_similarity(proj_vector, tool_emb)

        # Term 3: PastSuccess_norm (cold start = 0.5)
        past_success_norm = 0.5
        if db_session:
            try:
                past_success_norm = await get_past_success_rate(db_session, tid)
            except Exception as e:
                logger.warning(f"Error fetching past success rate for '{tid}': {e}")

        # Compute final weighted score
        final_score = (
            (w1_eff * query_sim_norm)
            + (w2_eff * proj_match_norm)
            + (w3_eff * past_success_norm)
        )
        final_score = round(max(0.0, min(1.0, final_score)), 4)

        enriched_tool = dict(tool)
        enriched_tool["score"] = final_score
        enriched_tool["score_breakdown"] = {
            "query_sim": round(query_sim_norm, 4),
            "proj_match": round(proj_match_norm, 4) if has_proj_context else 0.0,
            "past_success": round(past_success_norm, 4),
            "weights": {
                "w1": w1_eff,
                "w2": w2_eff,
                "w3": w3_eff,
            },
            "ablation_mode": mode,
        }
        scored_candidates.append(enriched_tool)

    # Sort descending by final weighted score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates
