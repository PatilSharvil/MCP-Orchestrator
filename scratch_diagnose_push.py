"""Diagnostic script to inspect score breakdown and raw embedding text for push_files query."""

import asyncio
import json
from orchestrator.db import init_db, AsyncSessionLocal
from retrieval.embedder import sync_all_embeddings_from_db, format_tool_embedding_text
from orchestrator.server import discover_tools
from ranking.ranker import compute_3term_ranking

async def main():
    await init_db()
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    query = "push code to github repository"
    print("=" * 80)
    print(f"PART 1 DIAGNOSIS: QUERY = '{query}'")
    print("=" * 80)

    # Discover candidate tools
    candidates = await discover_tools(query=query, workspace_dir=".", bypass=False)

    # Compute ranking under current full system
    ranked_tools = await compute_3term_ranking(
        candidate_tools=candidates,
        query=query,
        workspace_path=".",
        ablation_mode="full",
    )

    print("\n--- 1. FULL SCORE BREAKDOWN ---")
    for idx, t in enumerate(ranked_tools, 1):
        tid = t["tool_id"]
        score = t["score"]
        bd = t.get("score_breakdown", {})
        q_sim = bd.get("query_sim", 0.0)
        p_match = bd.get("proj_match", 0.0)
        p_succ = bd.get("past_success", 0.0)
        print(f"Rank #{idx:2d} | {tid:<40} | Final: {score:.4f} | QuerySim: {q_sim:.4f} | ProjMatch: {p_match:.4f} | PastSuccess: {p_succ:.4f}")

    print("\n--- 2. RAW EMBEDDING TEXT STRINGS ---")
    target_ids = ["github-server:push_files", "github-server:create_or_update_file", "github-server:search_code", "github-server:get_file_contents"]
    
    for tool in candidates:
        tid = tool["tool_id"]
        if tid in target_ids:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            schema = tool.get("input_schema", {})
            raw_text = format_tool_embedding_text(name, desc, schema)
            print(f"\n[Tool ID: {tid}]")
            print("Raw Embedding Text:")
            print(raw_text)
            print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
