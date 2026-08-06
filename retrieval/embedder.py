"""Embedding pipeline using sentence-transformers (all-MiniLM-L6-v2) and ChromaDB."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from orchestrator.config import settings
from orchestrator.db import MCPServer, MCPTool

logger = logging.getLogger(__name__)

# Model Singleton
_MODEL: Optional[SentenceTransformer] = None

# Singleton ChromaDB client instance
_CHROMA_CLIENT: Optional[chromadb.ClientAPI] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and return the sentence-transformers model instance."""
    global _MODEL
    if _MODEL is None:
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def reset_chroma_client(new_client: Optional[chromadb.ClientAPI] = None):
    """Reset or set global ChromaDB client singleton (used for test isolation)."""
    global _CHROMA_CLIENT
    _CHROMA_CLIENT = new_client


def get_chroma_client(host: str = None, port: int = None, reset: bool = False) -> chromadb.ClientAPI:
    """Get ChromaDB HTTP client or fallback to EphemeralClient if server is unavailable."""
    global _CHROMA_CLIENT
    if reset:
        _CHROMA_CLIENT = None
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT

    h = host or settings.CHROMADB_HOST
    p = port or settings.CHROMADB_PORT
    try:
        client = chromadb.HttpClient(host=h, port=p)
        client.heartbeat()
        _CHROMA_CLIENT = client
        return client
    except Exception as e:
        logger.warning(
            f"Could not connect to ChromaDB HTTP client at {h}:{p} ({e}). Falling back to in-memory EphemeralClient."
        )
        _CHROMA_CLIENT = chromadb.EphemeralClient()
        return _CHROMA_CLIENT


def get_or_create_collections(client: Optional[chromadb.ClientAPI] = None):
    """Retrieve or create the 'servers' and 'tools' ChromaDB collections with cosine metric."""
    cli = client or get_chroma_client()
    servers_col = cli.get_or_create_collection(
        name="servers",
        metadata={"hnsw:space": "cosine"},
    )
    tools_col = cli.get_or_create_collection(
        name="tools",
        metadata={"hnsw:space": "cosine"},
    )
    return servers_col, tools_col


def extract_parameter_names(input_schema: Optional[Dict[str, Any]]) -> List[str]:
    """Extract key parameter names from a tool's JSON input_schema."""
    if not input_schema or not isinstance(input_schema, dict):
        return []
    properties = input_schema.get("properties")
    if isinstance(properties, dict):
        return list(properties.keys())
    return []


def format_tool_embedding_text(
    name: str, description: Optional[str], input_schema: Optional[Dict[str, Any]]
) -> str:
    """Format tool metadata into textual input for the embedding model."""
    desc_str = description.strip() if description else "No description provided."
    param_names = extract_parameter_names(input_schema)
    param_str = ", ".join(param_names) if param_names else "none"
    return f"Tool: {name}\nDescription: {desc_str}\nParameters: {param_str}"


def format_server_embedding_text(
    server_name: str, tools_data: List[Dict[str, Any]]
) -> str:
    """Format server metadata and tool descriptions into textual input for the embedding model."""
    lines = [f"Server: {server_name}"]
    if tools_data:
        lines.append("Tools:")
        for t in tools_data:
            t_name = t.get("name", "")
            t_desc = t.get("description", "") or ""
            lines.append(f"- {t_name}: {t_desc}")
    else:
        lines.append("Tools: none")
    return "\n".join(lines)


def upsert_server_and_tools_embeddings(
    server_id: str,
    server_name: str,
    tools: List[Dict[str, Any]],
    chroma_client: Optional[chromadb.ClientAPI] = None,
):
    """Generate embeddings and upsert server and tools into ChromaDB collections idempotently."""
    client = chroma_client or get_chroma_client()
    servers_col, tools_col = get_or_create_collections(client)
    model = get_embedding_model()

    # 1. Upsert Server Embedding
    server_text = format_server_embedding_text(server_name, tools)
    server_vector = model.encode([server_text])[0].tolist()

    servers_col.upsert(
        ids=[server_id],
        embeddings=[server_vector],
        documents=[server_text],
        metadatas=[
            {
                "server_id": server_id,
                "name": server_name,
                "tool_count": len(tools),
            }
        ],
    )
    logger.info(f"Upserted server embedding for '{server_id}' into ChromaDB.")

    # 2. Upsert Tool Embeddings
    if tools:
        tool_ids = []
        tool_texts = []
        tool_metadatas = []

        for t in tools:
            t_name = t["name"]
            t_desc = t.get("description", "")
            t_schema = t.get("input_schema", {})
            full_tool_id = f"{server_id}:{t_name}"

            text = format_tool_embedding_text(t_name, t_desc, t_schema)

            tool_ids.append(full_tool_id)
            tool_texts.append(text)
            tool_metadatas.append(
                {
                    "server_id": server_id,
                    "name": t_name,
                }
            )

        tool_vectors = model.encode(tool_texts).tolist()

        tools_col.upsert(
            ids=tool_ids,
            embeddings=tool_vectors,
            documents=tool_texts,
            metadatas=tool_metadatas,
        )
        logger.info(
            f"Upserted {len(tool_ids)} tool embeddings for server '{server_id}' into ChromaDB."
        )


async def sync_all_embeddings_from_db(
    session, chroma_client: Optional[chromadb.ClientAPI] = None
) -> Dict[str, int]:
    """Read all servers and tools from PostgreSQL and sync them into ChromaDB collections."""
    from sqlalchemy import select

    res_servers = await session.execute(select(MCPServer))
    servers = res_servers.scalars().all()

    servers_count = 0
    tools_count = 0

    for s in servers:
        res_tools = await session.execute(
            select(MCPTool).where(MCPTool.server_id == s.id)
        )
        tools = res_tools.scalars().all()

        tools_data = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

        upsert_server_and_tools_embeddings(
            server_id=s.id,
            server_name=s.name,
            tools=tools_data,
            chroma_client=chroma_client,
        )
        servers_count += 1
        tools_count += len(tools_data)

    logger.info(
        f"Completed embedding sync: {servers_count} servers, {tools_count} tools."
    )
    return {"servers_synced": servers_count, "tools_synced": tools_count}
