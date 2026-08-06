"""Tests for Phase 2: Embedding Pipeline.

Verifies:
1. Parameter extraction from tool JSON schemas.
2. Embedding text formatting for tools and servers.
3. Vector generation and upsert into ChromaDB collections ('servers' and 'tools').
4. Metadata tagging with server_id.
5. Semantic similarity queries ranking relevant tools and servers above unrelated ones.
6. Idempotent re-upserts.
"""

import pytest
import chromadb

from retrieval.embedder import (
    extract_parameter_names,
    format_tool_embedding_text,
    format_server_embedding_text,
    upsert_server_and_tools_embeddings,
    get_or_create_collections,
    get_embedding_model,
    reset_chroma_client,
)


def test_extract_parameter_names():
    """Verify property keys are extracted cleanly from JSON input schema."""
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["path"],
    }
    params = extract_parameter_names(schema)
    assert params == ["path", "content", "encoding"]

    assert extract_parameter_names(None) == []
    assert extract_parameter_names({}) == []


def test_format_embedding_texts():
    """Verify tool and server text formatting."""
    tool_text = format_tool_embedding_text(
        name="read_file",
        description="Read the complete contents of a file from disk.",
        input_schema={"properties": {"path": {}, "encoding": {}}},
    )
    assert "Tool: read_file" in tool_text
    assert "Description: Read the complete contents of a file from disk." in tool_text
    assert "Parameters: path, encoding" in tool_text

    server_text = format_server_embedding_text(
        server_name="Filesystem MCP Server",
        tools_data=[
            {"name": "read_file", "description": "Read file contents"},
            {"name": "write_file", "description": "Write contents to file"},
        ],
    )
    assert "Server: Filesystem MCP Server" in server_text
    assert "- read_file: Read file contents" in server_text
    assert "- write_file: Write contents to file" in server_text


def test_chroma_embedding_upsert_and_metadata():
    """Verify embeddings and metadata are written into ChromaDB collections."""
    chroma_client = chromadb.EphemeralClient()
    reset_chroma_client(chroma_client)
    try:
        chroma_client.delete_collection("servers")
    except Exception:
        pass
    try:
        chroma_client.delete_collection("tools")
    except Exception:
        pass

    servers_col, tools_col = get_or_create_collections(chroma_client)

    tools_data = [
        {
            "name": "read_file",
            "description": "Read contents from a specified file path.",
            "input_schema": {"properties": {"path": {}}},
        },
        {
            "name": "list_directory",
            "description": "List files and directories inside a target folder.",
            "input_schema": {"properties": {"path": {}}},
        },
    ]

    # First upsert
    upsert_server_and_tools_embeddings(
        server_id="filesystem-server",
        server_name="Filesystem MCP Server",
        tools=tools_data,
        chroma_client=chroma_client,
    )

    assert servers_col.count() == 1
    assert tools_col.count() == 2

    # Verify tool metadata contains server_id
    tool_entry = tools_col.get(ids=["filesystem-server:read_file"])
    assert len(tool_entry["ids"]) == 1
    assert tool_entry["metadatas"][0]["server_id"] == "filesystem-server"
    assert tool_entry["metadatas"][0]["name"] == "read_file"

    # Idempotency check: Re-upserting should update, not duplicate
    upsert_server_and_tools_embeddings(
        server_id="filesystem-server",
        server_name="Filesystem MCP Server",
        tools=tools_data,
        chroma_client=chroma_client,
    )
    assert servers_col.count() == 1
    assert tools_col.count() == 2


def test_semantic_similarity_search():
    """Verify cosine similarity search ranks relevant tools higher than unrelated ones."""
    chroma_client = chromadb.EphemeralClient()
    reset_chroma_client(chroma_client)
    try:
        chroma_client.delete_collection("servers")
    except Exception:
        pass
    try:
        chroma_client.delete_collection("tools")
    except Exception:
        pass

    servers_col, tools_col = get_or_create_collections(chroma_client)

    # Register filesystem tools and an unrelated github tool
    fs_tools = [
        {
            "name": "read_file",
            "description": "Read file content from local disk storage.",
            "input_schema": {"properties": {"path": {}}},
        },
        {
            "name": "create_github_issue",
            "description": "Create a new issue on GitHub repository tracker.",
            "input_schema": {"properties": {"title": {}, "body": {}}},
        },
    ]

    upsert_server_and_tools_embeddings(
        server_id="demo-server",
        server_name="Demo Multi Server",
        tools=fs_tools,
        chroma_client=chroma_client,
    )

    model = get_embedding_model()

    # Query 1: "read a document from disk"
    query1 = "read a document from disk"
    query1_vec = model.encode([query1]).tolist()

    res1 = tools_col.query(query_embeddings=query1_vec, n_results=2)
    top_tool_id_1 = res1["ids"][0][0]
    assert top_tool_id_1 == "demo-server:read_file"

    # Query 2: "report a bug or open an issue on github"
    query2 = "report a bug or open an issue on github"
    query2_vec = model.encode([query2]).tolist()

    res2 = tools_col.query(query_embeddings=query2_vec, n_results=2)
    top_tool_id_2 = res2["ids"][0][0]
    assert top_tool_id_2 == "demo-server:create_github_issue"
