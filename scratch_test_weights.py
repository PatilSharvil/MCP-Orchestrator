"""Test ranking scores under different weight configurations."""

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

    queries = [
        "save this memory",
        "push code to github",
        "read project configuration file in current workspace",
    ]

    weight_configs = [
        ("Default Equal (0.33, 0.33, 0.34)", 0.33, 0.33, 0.34),
        ("Query Heavy (0.60, 0.25, 0.15)", 0.60, 0.25, 0.15),
        ("Query Dominant (0.70, 0.20, 0.10)", 0.70, 0.20, 0.10),
    ]

    for q in queries:
        print("=" * 80)
        print(f"QUERY: '{q}'")
        print("=" * 80)
        candidates = await discover_tools(query=q, workspace_dir=".", bypass=False)

        for label, w1, w2, w3 in weight_configs:
            ranked = await compute_3term_ranking(
                candidate_tools=candidates,
                query=q,
                workspace_path=".",
                w1=w1, w2=w2, w3=w3,
            )
            print(f"\n--- {label} ---")
            for idx, t in enumerate(ranked[:3], 1):
                bd = t.get("score_breakdown", {})
                print(f"  #{idx} {t['tool_id']:<40} | score: {t['score']:.4f} | q_sim: {bd.get('query_sim', 0):.4f} | p_match: {bd.get('proj_match', 0):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
