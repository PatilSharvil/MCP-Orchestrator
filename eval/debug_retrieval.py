"""Diagnostic script for retrieval quality investigation."""

import asyncio
import logging
from retrieval.embedder import (
    get_chroma_client, get_or_create_collections, get_embedding_model,
    sync_all_embeddings_from_db
)
from retrieval.retriever import stage_1_retrieve_servers, stage_2_retrieve_tools
from orchestrator.db import AsyncSessionLocal, init_db

logging.basicConfig(level=logging.WARNING)


async def main():
    await init_db()
    client = get_chroma_client()
    servers_col, tools_col = get_or_create_collections(client)

    # Sync embeddings
    print("=== Syncing embeddings from DB ===")
    async with AsyncSessionLocal() as session:
        await sync_all_embeddings_from_db(session)

    model = get_embedding_model()
    query = "read project configuration file"

    # ---- 1. Stage 1 raw server scores ----
    print(f"\n=== Stage 1: Server-Level Retrieval for '{query}' ===")
    q_vec = model.encode([query])[0].tolist()
    server_results = servers_col.query(
        query_embeddings=[q_vec],
        n_results=min(10, servers_col.count()),
        include=["distances", "metadatas", "documents"],
    )
    for i, (sid, dist, meta) in enumerate(zip(
        server_results["ids"][0],
        server_results["distances"][0],
        server_results["metadatas"][0],
    )):
        sim = max(0.0, 1.0 - dist)
        print(f"  [{i+1}] server_id={sid}  cosine_sim={sim:.4f}  (distance={dist:.4f})")

    # ---- 2. Stage 2 raw tool scores for filesystem-server ----
    print(f"\n=== Stage 2: Tool-Level Retrieval for '{query}' (only filesystem-server) ===")
    tool_count = tools_col.count()
    print(f"  Total tools in ChromaDB: {tool_count}")
    fs_results = tools_col.query(
        query_embeddings=[q_vec],
        n_results=min(20, tool_count),
        where={"server_id": "filesystem-server"},
        include=["distances", "metadatas"],
    )
    for i, (tid, dist, meta) in enumerate(zip(
        fs_results["ids"][0],
        fs_results["distances"][0],
        fs_results["metadatas"][0],
    )):
        sim = max(0.0, 1.0 - dist)
        print(f"  [{i+1}] tool_id={tid}  cosine_sim={sim:.4f}")

    # ---- 3. Check read_file embedding exists ----
    print("\n=== ChromaDB: Check read_file embedding ===")
    try:
        rec = tools_col.get(ids=["filesystem-server:read_file"], include=["documents", "embeddings"])
        if rec["ids"]:
            doc = rec["documents"][0]
            emb = rec["embeddings"][0]
            print(f"  Document text stored:\n    {doc}")
            print(f"  Embedding dim: {len(emb)}, first 5 dims: {emb[:5]}")
        else:
            print("  NOT FOUND in ChromaDB!")
    except Exception as e:
        print(f"  Error: {e}")

    # ---- 4. Sanity baseline: exact name query ----
    print("\n=== Sanity Baseline: query='read_file' vs filesystem-server:read_file ===")
    exact_query = "read_file"
    ev = model.encode([exact_query])[0].tolist()
    try:
        rec2 = tools_col.get(ids=["filesystem-server:read_file"], include=["embeddings"])
        if rec2["embeddings"]:
            tool_emb = rec2["embeddings"][0]
            dot = sum(a * b for a, b in zip(ev, tool_emb))
            import math
            na = math.sqrt(sum(a * a for a in ev))
            nb = math.sqrt(sum(b * b for b in tool_emb))
            sim = dot / (na * nb) if na and nb else 0.0
            print(f"  Cosine similarity (read_file query vs read_file embedding): {sim:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # ---- 5. Unconstrained top-10 across all tools ----
    print(f"\n=== Unconstrained Top-10 Tools (all servers, no filter) for '{query}' ===")
    all_results = tools_col.query(
        query_embeddings=[q_vec],
        n_results=min(10, tool_count),
        include=["distances", "metadatas"],
    )
    for i, (tid, dist, meta) in enumerate(zip(
        all_results["ids"][0],
        all_results["distances"][0],
        all_results["metadatas"][0],
    )):
        sim = max(0.0, 1.0 - dist)
        print(f"  [{i+1}] tool_id={tid}  cosine_sim={sim:.4f}")

    # ---- 6. Check filesystem-server embedding document text ----
    print("\n=== filesystem-server embedding document text ===")
    try:
        srec = servers_col.get(ids=["filesystem-server"], include=["documents"])
        if srec["ids"]:
            print(f"  {srec['documents'][0][:500]}")
        else:
            print("  NOT FOUND in ChromaDB!")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
