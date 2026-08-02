#!/usr/bin/env python
"""Sync Embeddings Script for MCP Orchestrator.

Reads all downstream MCP servers and tools from PostgreSQL and (re)embeds them
into ChromaDB vector collections. Useful for backfilling or recovering after a ChromaDB reset.
"""

import asyncio
import logging
from orchestrator.db import init_db, AsyncSessionLocal
from retrieval.embedder import sync_all_embeddings_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def main():
    print("=== Starting MCP Orchestrator Embedding Sync ===")
    await init_db()
    async with AsyncSessionLocal() as session:
        stats = await sync_all_embeddings_from_db(session)
        print("=== Embedding Sync Completed Successfully ===")
        print(f"Servers Synced: {stats['servers_synced']}")
        print(f"Tools Synced: {stats['tools_synced']}")


if __name__ == "__main__":
    asyncio.run(main())
