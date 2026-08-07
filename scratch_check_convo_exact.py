"""Check exact discover_tools output for conversational queries under live server."""

import asyncio
from orchestrator.db import init_db, AsyncSessionLocal
from retrieval.embedder import sync_all_embeddings_from_db
from orchestrator.server import discover_tools

async def main():
    await init_db()
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    queries = ["hi", "hello", "what is MCP?", "thanks"]
    print("=" * 80)
    print("EXACT LIVE DISCOVER_TOOLS OUTPUT (Threshold = 0.15, Weights = 0.33/0.33/0.34)")
    print("=" * 80)

    for q in queries:
        tools = await discover_tools(query=q, workspace_dir=".", bypass=False)
        print(f"\nQuery: '{q}' -> Returned {len(tools)} tools")
        if tools:
            top = tools[0]
            bd = top.get("score_breakdown", {})
            print(f"  Top Tool: {top['tool_id']} | QuerySim: {bd.get('query_sim', 0):.4f} | Final Score: {top['score']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
