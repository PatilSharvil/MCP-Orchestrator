"""Quick sanity check: exact name query cosine similarity."""
import asyncio, math, logging
from retrieval.embedder import get_chroma_client, get_or_create_collections, get_embedding_model, sync_all_embeddings_from_db
from orchestrator.db import AsyncSessionLocal, init_db
logging.basicConfig(level=logging.WARNING)

async def main():
    await init_db()
    client = get_chroma_client()
    _, tools_col = get_or_create_collections(client)
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    model = get_embedding_model()

    tests = [
        ("read_file",                          "filesystem-server:read_file"),
        ("read project configuration file",    "filesystem-server:read_file"),
        ("search github pull requests",        "github-server:search_issues"),
        ("store entity in knowledge graph",    "memory-server:create_entities"),
    ]

    print(f"{'Query':<45} {'Tool':<40} {'CosSim':>7}")
    print("-" * 95)
    for query, tool_id in tests:
        qv = model.encode([query])[0].tolist()
        rec = tools_col.get(ids=[tool_id], include=["embeddings"])
        if not rec["ids"]:
            print(f"{query:<45} {tool_id:<40} MISSING")
            continue
        tv = list(rec["embeddings"][0])
        dot = sum(a*b for a,b in zip(qv, tv))
        na  = math.sqrt(sum(a*a for a in qv))
        nb  = math.sqrt(sum(b*b for b in tv))
        sim = dot/(na*nb) if na and nb else 0.0
        print(f"{query:<45} {tool_id:<40} {sim:>7.4f}")

if __name__ == "__main__":
    asyncio.run(main())
