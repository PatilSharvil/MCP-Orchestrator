"""Test Part 2 Minimum Relevance Threshold against conversational queries and all 20 benchmark tasks."""

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

    # 1. Conversational Queries
    conversational_queries = [
        "hi",
        "hello",
        "what can you do for me?",
        "what is MCP?",
        "thanks",
    ]

    # 2. Load 20 benchmark tasks
    with open("eval/benchmark_tasks.json", "r") as f:
        tasks = json.load(f)

    print("=" * 80)
    print("PART 2 DIAGNOSIS: TESTING TOP CANDIDATE QUERY_SIM & FINAL SCORES")
    print("=" * 80)

    print("\n--- 1. CONVERSATIONAL / GREETING QUERIES ---")
    conv_top_scores = []
    for q in conversational_queries:
        candidates = await discover_tools(query=q, workspace_dir=".", bypass=False)
        ranked = await compute_3term_ranking(candidate_tools=candidates, query=q, workspace_path=".")
        if ranked:
            top = ranked[0]
            bd = top.get("score_breakdown", {})
            q_sim = bd.get("query_sim", 0.0)
            final_score = top["score"]
            conv_top_scores.append((q, top["tool_id"], q_sim, final_score))
            print(f"Query: '{q:<30}' | Top Tool: {top['tool_id']:<35} | QuerySim: {q_sim:.4f} | Final: {final_score:.4f}")
        else:
            print(f"Query: '{q:<30}' | Returned ZERO tools.")

    print("\n--- 2. BENCHMARK REAL TASKS (ALL 20 TASKS) ---")
    task_top_scores = []
    for t in tasks:
        tid = t.get("task_id", t.get("id", 0))
        q = t["query"]
        candidates = await discover_tools(query=q, workspace_dir=".", bypass=False)
        ranked = await compute_3term_ranking(candidate_tools=candidates, query=q, workspace_path=".")
        if ranked:
            top = ranked[0]
            bd = top.get("score_breakdown", {})
            q_sim = bd.get("query_sim", 0.0)
            final_score = top["score"]
            task_top_scores.append((tid, q, top["tool_id"], q_sim, final_score))
            print(f"Task {str(tid):<12} | Query: '{q[:35]:<35}' | Top Tool: {top['tool_id']:<35} | QuerySim: {q_sim:.4f} | Final: {final_score:.4f}")

    # Minimum and Maximum analysis
    min_conv_qsim = min(x[2] for x in conv_top_scores) if conv_top_scores else 0
    max_conv_qsim = max(x[2] for x in conv_top_scores) if conv_top_scores else 0
    min_task_qsim = min(x[3] for x in task_top_scores) if task_top_scores else 0
    
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS:")
    print(f"Conversational Queries QuerySim Range : {min_conv_qsim:.4f} to {max_conv_qsim:.4f}")
    print(f"Real Benchmark Tasks Minimum QuerySim  : {min_task_qsim:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
