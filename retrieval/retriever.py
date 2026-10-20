"""Two-Stage Retrieval Pipeline for MCP Orchestrator.

Stage 1: Vector search over 'servers' collection -> narrows query to top-N candidate servers.
Stage 2: Metadata-filtered vector search over 'tools' collection -> retrieves top-M tools from candidate servers.
"""

import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional
import chromadb

from orchestrator.config import settings
from retrieval.embedder import get_embedding_model, get_chroma_client, get_or_create_collections

logger = logging.getLogger(__name__)


def distance_to_similarity(distance: float) -> float:
    """Convert ChromaDB cosine distance (0..2) to similarity score (0..1)."""
    score = 1.0 - distance
    return round(max(0.0, min(1.0, score)), 4)


def retrieve_servers_lexical(
    query: str,
    servers_col,
    n: int,
) -> List[Dict[str, Any]]:
    """Stage 1 Lexical ablation: Pure Python TF-IDF lexical retrieval over servers' descriptions."""
    res = servers_col.get(include=["documents", "metadatas"])
    if not res or not res.get("ids"):
        return []

    ids = res["ids"]
    docs = res.get("documents", []) or []
    metadatas = res.get("metadatas", []) or []

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    query_tokens = tokenize(query)
    if not query_tokens:
        return [
            {
                "server_id": sid,
                "name": meta.get("name", sid),
                "score": 0.0,
            }
            for sid, meta in zip(ids, metadatas)
        ][:n]

    # Calculate Term Frequencies (TF) and Document Frequencies (DF)
    doc_tokens = [tokenize(doc) for doc in docs]
    doc_tfs = []
    df = {}

    for tokens in doc_tokens:
        tf = {}
        unique_tokens = set(tokens)
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        doc_tfs.append(tf)
        for t in unique_tokens:
            df[t] = df.get(t, 0) + 1

    # Score each server
    num_docs = len(docs)
    scored = []

    for i, (sid, meta) in enumerate(zip(ids, metadatas)):
        tf = doc_tfs[i]
        score = 0.0
        for q_t in query_tokens:
            if q_t in tf:
                tf_val = tf[q_t]
                term_df = df.get(q_t, 0)
                # Standard TF-IDF score
                term_idf = math.log(1.0 + float(num_docs) / (term_df + 1.0))
                score += tf_val * term_idf

        scored.append(
            {
                "server_id": sid,
                "name": meta.get("name", sid),
                "score": round(score, 4),
            }
        )

    # Sort descending by TF-IDF score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n]


def stage_1_retrieve_servers(
    query: str,
    top_n: int = None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stage 1: Search 'servers' collection to identify top-N relevant candidate servers.

    Under ablation_mode='no_topk', top_n defaults to a very large number (100).
    Under ablation_mode='stage1_lexical_only', pure TF-IDF lexical search is used.
    """
    mode = (ablation_mode or "full").lower()

    if mode == "no_topk":
        n = 100  # Return all servers
    else:
        n = top_n if top_n is not None else settings.TOP_N_SERVERS

    client = chroma_client or get_chroma_client()
    servers_col, _ = get_or_create_collections(client)

    if servers_col.count() == 0:
        logger.warning("ChromaDB 'servers' collection is empty.")
        return []

    # 1. Check for lexical Stage 1 ablation mode
    if mode == "stage1_lexical_only":
        candidate_servers = retrieve_servers_lexical(query, servers_col, n)
        logger.info(
            f"Stage 1 Lexical retrieved {len(candidate_servers)} candidate servers for query '{query}': {[s['server_id'] for s in candidate_servers]}"
        )
        return candidate_servers

    # 2. Default Semantic Stage 1 retrieval
    model = get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False).tolist()

    try:
        results = servers_col.query(
            query_embeddings=query_vector,
            n_results=min(n, servers_col.count()),
            include=["metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"ChromaDB Stage 1 query failed ({e}). Falling back to lexical retrieval.")
        return retrieve_servers_lexical(query, servers_col, n)

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
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stage 2: Search 'tools' collection restricted ONLY to candidate server_ids via metadata filter.

    Under ablation_mode='no_topk', top_m defaults to a very large number (1000).
    """
    if not candidate_server_ids:
        logger.warning("Stage 2 received empty candidate_server_ids list.")
        return []

    mode = (ablation_mode or "full").lower()

    if mode == "no_topk":
        m = 1000  # Return all tools from candidate servers
    else:
        m = top_m if top_m is not None else settings.TOP_M_TOOLS

    client = chroma_client or get_chroma_client()
    _, tools_col = get_or_create_collections(client)

    if tools_col.count() == 0:
        logger.warning("ChromaDB 'tools' collection is empty.")
        return []

    model = get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False).tolist()

    # Build metadata filter for candidate_server_ids
    if len(candidate_server_ids) == 1:
        where_filter = {"server_id": candidate_server_ids[0]}
    else:
        where_filter = {"server_id": {"$in": candidate_server_ids}}

    try:
        results = tools_col.query(
            query_embeddings=query_vector,
            n_results=min(m, tools_col.count()),
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )
    except Exception as e:
        logger.warning(f"ChromaDB Stage 2 query failed ({e}).")
        results = None

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
    ablation_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute complete Two-Stage Retrieval pipeline (Stage 1 -> Stage 2)."""
    # 1. Stage 1: Server Retrieval
    candidate_servers = stage_1_retrieve_servers(
        query=query,
        top_n=top_n,
        chroma_client=chroma_client,
        ablation_mode=ablation_mode,
    )
    candidate_server_ids = [s["server_id"] for s in candidate_servers]

    # 2. Stage 2: Tool Retrieval
    retrieved_tools = stage_2_retrieve_tools(
        query=query,
        candidate_server_ids=candidate_server_ids,
        top_m=top_m,
        chroma_client=chroma_client,
        ablation_mode=ablation_mode,
    )

    return retrieved_tools
