"""Retrieval Evaluation Runner for MCP Orchestrator.

Evaluates Two-Stage Vector Retrieval hit-rate against hand-labeled ground-truth queries.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import chromadb

from orchestrator.config import settings
from retrieval.retriever import two_stage_retrieve
from retrieval.embedder import get_chroma_client, upsert_server_and_tools_embeddings

logger = logging.getLogger(__name__)


def load_ground_truth_dataset(filepath: str = None) -> List[Dict[str, Any]]:
    """Load hand-labeled queries from JSON fixture file."""
    path = filepath or os.path.join(
        os.path.dirname(__file__), "ground_truth_queries.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval_hit_rate(
    dataset: List[Dict[str, Any]],
    top_n: int = settings.TOP_N_SERVERS,
    top_m: int = settings.TOP_M_TOOLS,
    chroma_client: Optional[chromadb.ClientAPI] = None,
) -> Dict[str, Any]:
    """Run evaluation dataset through two-stage retrieval pipeline and compute hit-rate.

    Hit-rate @ M: Fraction of queries where at least 1 ground-truth tool is in top-M results.
    Stage 1 Hit-rate: Fraction of queries where expected_server_id is in top-N servers.
    """
    total_queries = len(dataset)
    stage1_hits = 0
    stage2_hits = 0

    for item in dataset:
        query = item["query"]
        expected_server = item.get("expected_server_id")
        expected_tools = set(item.get("relevant_tool_ids", []))

        # Run Two-Stage Retrieval
        retrieved_tools = two_stage_retrieve(
            query=query, top_n=top_n, top_m=top_m, chroma_client=chroma_client
        )
        retrieved_tool_ids = set([t["tool_id"] for t in retrieved_tools])
        retrieved_server_ids = set([t["server_id"] for t in retrieved_tools])

        # Stage 1 check
        if expected_server in retrieved_server_ids:
            stage1_hits += 1

        # Stage 2 check (Hit-rate @ M)
        if len(expected_tools.intersection(retrieved_tool_ids)) > 0:
            stage2_hits += 1

    stage1_hit_rate = round(stage1_hits / total_queries, 4) if total_queries else 0.0
    stage2_hit_rate = round(stage2_hits / total_queries, 4) if total_queries else 0.0

    return {
        "total_queries": total_queries,
        "stage1_hits": stage1_hits,
        "stage1_hit_rate": stage1_hit_rate,
        "stage2_hits": stage2_hits,
        "stage2_hit_rate_at_m": stage2_hit_rate,
        "top_n_servers": top_n,
        "top_m_tools": top_m,
    }


def seed_eval_mock_data(chroma_client: chromadb.ClientAPI):
    """Seed ChromaDB with multi-server demo dataset for standalone evaluation."""
    # 1. Filesystem Server
    fs_tools = [
        {"name": "read_file", "description": "Read file contents from local disk storage", "input_schema": {"properties": {"path": {}}}},
        {"name": "write_file", "description": "Write data output to a file path on disk", "input_schema": {"properties": {"path": {}, "content": {}}}},
        {"name": "list_directory", "description": "List files and subdirectories inside folder", "input_schema": {"properties": {"path": {}}}},
        {"name": "move_file", "description": "Move or rename file path on disk", "input_schema": {"properties": {"source": {}, "destination": {}}}},
        {"name": "create_directory", "description": "Create new folder directory", "input_schema": {"properties": {"path": {}}}},
        {"name": "get_file_info", "description": "Get file metadata size and attributes", "input_schema": {"properties": {"path": {}}}},
        {"name": "delete_file", "description": "Remove file from disk", "input_schema": {"properties": {"path": {}}}},
    ]
    upsert_server_and_tools_embeddings("filesystem-server", "Filesystem MCP Server", fs_tools, chroma_client)

    # 2. Memory Server
    mem_tools = [
        {"name": "create_entities", "description": "Create knowledge graph entities and observations", "input_schema": {"properties": {"entities": {}}}},
        {"name": "create_relations", "description": "Create directed relations between entities in graph", "input_schema": {"properties": {"relations": {}}}},
        {"name": "read_graph", "description": "Read entire knowledge graph structure", "input_schema": {}},
        {"name": "search_nodes", "description": "Search knowledge graph for stored entities and nodes", "input_schema": {"properties": {"query": {}}}},
    ]
    upsert_server_and_tools_embeddings("memory-server", "Memory Knowledge Graph Server", mem_tools, chroma_client)

    # 3. GitHub Server
    gh_tools = [
        {"name": "create_issue", "description": "Open a new issue or report bug on GitHub repository", "input_schema": {"properties": {"owner": {}, "repo": {}, "title": {}}}},
        {"name": "search_repositories", "description": "Search repositories on GitHub by query string", "input_schema": {"properties": {"query": {}}}},
        {"name": "create_pull_request", "description": "Create pull request to merge code branch on GitHub", "input_schema": {"properties": {"owner": {}, "repo": {}, "head": {}, "base": {}}}},
        {"name": "get_file_contents", "description": "Get file contents from remote GitHub repository", "input_schema": {"properties": {"owner": {}, "repo": {}, "path": {}}}},
    ]
    upsert_server_and_tools_embeddings("github-server", "GitHub MCP Server", gh_tools, chroma_client)


def run_standalone_eval():
    """CLI runner for retrieval evaluation."""
    print("=== Running MCP Orchestrator Retrieval Evaluation ===")
    chroma_client = get_chroma_client()
    
    # Check if collections populated; if not, seed demo mock data
    servers_col, _ = get_or_create_collections(chroma_client)
    if servers_col.count() == 0:
        print("Seeding multi-server mock data into ChromaDB for evaluation...")
        seed_eval_mock_data(chroma_client)

    dataset = load_ground_truth_dataset()
    results = evaluate_retrieval_hit_rate(dataset, chroma_client=chroma_client)

    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Total Ground Truth Queries: {results['total_queries']}")
    print(f"Stage 1 (Server Level) Hits: {results['stage1_hits']} / {results['total_queries']} ({results['stage1_hit_rate'] * 100:.1f}%)")
    print(f"Stage 2 (Tool Hit-Rate @ {results['top_m_tools']}): {results['stage2_hits']} / {results['total_queries']} ({results['stage2_hit_rate_at_m'] * 100:.1f}%)")
    print("===================================================\n")
    return results


if __name__ == "__main__":
    run_standalone_eval()
