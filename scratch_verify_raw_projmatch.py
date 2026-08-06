"""Verify raw proj_match values under No-ProjMatch vs No-Shared-Space modes."""

import asyncio
import json

from orchestrator.db import init_db, AsyncSessionLocal
from retrieval.embedder import sync_all_embeddings_from_db
from orchestrator.server import discover_tools
from ranking.ranker import compute_3term_ranking

async def main():
    await init_db()
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    with open("eval/benchmark_tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Sample 3 tasks
    sample_tasks = [tasks[0], tasks[1], tasks[2]]

    print("=" * 80)
    print("VERIFYING RAW PROJ_MATCH VALUES FOR CANDIDATE TOOLS")
    print("=" * 80)

    for task in sample_tasks:
        query = task["query"]
        print(f"\nTASK QUERY: '{query}'")

        # Discover tools under Stage 1 + Stage 2 retrieval
        candidates = await discover_tools(query=query, workspace_dir=".", bypass=False)

        # 1. Evaluate under No-ProjMatch mode
        ranked_no_proj = await compute_3term_ranking(
            candidate_tools=candidates,
            query=query,
            workspace_path=".",
            ablation_mode="no_projmatch",
        )

        # 2. Evaluate under No-Shared-Space mode
        ranked_no_space = await compute_3term_ranking(
            candidate_tools=candidates,
            query=query,
            workspace_path=".",
            ablation_mode="no_shared_space",
        )

        print("\n--- NO-PROJMATCH MODE ---")
        for t in ranked_no_proj[:5]:
            bd = t.get("score_breakdown", {})
            print(f"  Tool: {t['tool_id']:<35} | score: {t['score']:.4f} | query_sim: {bd['query_sim']:.4f} | proj_match: {bd['proj_match']:.6f}")

        print("\n--- NO-SHARED-SPACE MODE ---")
        for t in ranked_no_space[:5]:
            bd = t.get("score_breakdown", {})
            print(f"  Tool: {t['tool_id']:<35} | score: {t['score']:.4f} | query_sim: {bd['query_sim']:.4f} | proj_match: {bd['proj_match']:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
