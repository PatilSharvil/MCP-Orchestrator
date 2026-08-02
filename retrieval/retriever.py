"""Two-Stage Retrieval Pipeline for MCP Orchestrator.

Stage 1: Vector search over 'servers' collection -> narrows query to top-N candidate servers.
Stage 2: Metadata-filtered vector search over 'tools' collection -> retrieves top-M tools from candidate servers.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import chromadb

from orchestrator.config import settings
from retrieval.embedder import get_embedding_model, get_chroma_client, get_or_create_collections

logger = logging.getLogger(__name__)


def distance_to_similarity(distance: float) -> float:
    """Convert ChromaDB cosine distance (0..2) to similarity score (0..1)."""
    # Cosine distance = 1 - cosine_similarity => cosine_similarity = 1 - distance
    score = 1.0 - distance
    return round(max(0.0, min(1.0, score)), 4)


def stage_1_retrieve_servers(
    query: str,
    top_n: int = None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
) -> List[Dict[str, Any]]:
    """Stage 1: Search 'servers' collection to identify top-N relevant candidate servers.

    Returns:
        List of dicts: [{'server_id': str, 'name': str, 'score': float}]
    """
    n = top_n if top_n is not None else settings.TOP_N_SERVERS
    client = chroma_client or get_chroma_client()
    servers_col, _ = get_or_create_collections(client)

    if servers_col.count() == 0:
        logger.warning("ChromaDB 'servers' collection is empty.")
        return []

    model = get_embedding_model()
    query_vector = model.encode([query]).tolist()

    # Query ChromaDB servers collection
    results = servers_col.query(
        query_embeddings=query_vector,
        n_results=min(n, servers_col.count()),
        include=["metadatas", "distances"],
    )

    candidate_servers = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for sid, meta, dist in zip(ids, metadatas, distances):
            candidate_servers.append(
                {
                    "server_id": sid,
                    "name": meta.get("name", sid),
                    "score": distance_to_similarity(dist),
                }
            )

    logger.info(
        f"Stage 1 retrieved {len(candidate_servers)} candidate servers for query '{query}': {[s['server_id'] for s in candidate_servers]}"
    )
    return candidate_servers


def stage_2_retrieve_tools(
    query: str,
    candidate_server_ids: List[str],
    top_m: int = None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
) -> List[Dict[str, Any]]:
    """Stage 2: Search 'tools' collection restricted ONLY to candidate server_ids via metadata filter.

    Returns:
        List of dicts: [{'tool_id': str, 'server_id': str, 'name': str, 'score': float}]
    """
    if not candidate_server_ids:
        logger.warning("Stage 2 received empty candidate_server_ids list.")
        return []

    m = top_m if top_m is not None else settings.TOP_M_TOOLS
    client = chroma_client or get_chroma_client()
    _, tools_col = get_or_create_collections(client)

    if tools_col.count() == 0:
        logger.warning("ChromaDB 'tools' collection is empty.")
        return []

    model = get_embedding_model()
    query_vector = model.encode([query]).tolist()

    # Build metadata filter for candidate_server_ids
    if len(candidate_server_ids) == 1:
        where_filter = {"server_id": candidate_server_ids[0]}
    else:
        where_filter = {"server_id": {"$in": candidate_server_ids}}

    results = tools_col.query(
        query_embeddings=query_vector,
        n_results=min(m, tools_col.count()),
        where=where_filter,
        include=["metadatas", "distances", "documents"],
    )

    retrieved_tools = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for tid, meta, dist in zip(ids, metadatas, distances):
            retrieved_tools.append(
                {
                    "tool_id": tid,
                    "server_id": meta.get("server_id"),
                    "name": meta.get("name"),
                    "score": distance_to_similarity(dist),
                }
            )

    logger.info(
        f"Stage 2 retrieved {len(retrieved_tools)} candidate tools for query '{query}'."
    )
    return retrieved_tools


def two_stage_retrieve(
    query: str,
    top_n: int = None,
    top_m: int = None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
) -> List[Dict[str, Any]]:
    """Execute complete Two-Stage Retrieval pipeline (Stage 1 -> Stage 2).

    Returns ranked candidate tools list.
    """
    n = top_n if top_n is not None else settings.TOP_N_SERVERS
    m = top_m if top_m is not None else settings.TOP_M_TOOLS

    # 1. Stage 1: Server Retrieval
    candidate_servers = stage_1_retrieve_servers(
        query=query, top_n=n, chroma_client=chroma_client
    )
    candidate_server_ids = [s["server_id"] for s in candidate_servers]

    # 2. Stage 2: Tool Retrieval (filtered to candidate servers)
    retrieved_tools = stage_2_retrieve_tools(
        query=query,
        candidate_server_ids=candidate_server_ids,
        top_m=m,
        chroma_client=chroma_client,
    )

    return retrieved_tools
