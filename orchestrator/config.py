"""Configuration settings for MCP Orchestrator."""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class DownstreamServerConfig(BaseModel):
    """Configuration definition for a downstream MCP server connection."""

    id: str
    name: str
    transport_type: str = "stdio"  # "stdio" or "sse"
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    requires_auth: bool = False
    auth_env_var: str = ""


class Settings(BaseSettings):
    """Global configuration settings loaded from environment variables."""

    # Database Settings
    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL",
        "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/mcporchestrator",
    )

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Vector Storage Settings (ChromaDB on port 8001 externally / 8000 internally)
    CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "8001"))

    # Retrieval Pipeline Parameters (Phase 3)
    TOP_N_SERVERS: int = int(os.getenv("TOP_N_SERVERS", "3"))
    TOP_M_TOOLS: int = int(os.getenv("TOP_M_TOOLS", "10"))

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Downstream MCP Server Definitions (Filesystem, Memory, GitHub)
DEFAULT_DOWNSTREAM_SERVERS: List[DownstreamServerConfig] = [
    DownstreamServerConfig(
        id="filesystem-server",
        name="Filesystem MCP Server",
        transport_type="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    ),
    DownstreamServerConfig(
        id="memory-server",
        name="Memory Knowledge Graph MCP Server",
        transport_type="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
    ),
    DownstreamServerConfig(
        id="github-server",
        name="GitHub MCP Server",
        transport_type="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        requires_auth=True,
        auth_env_var="GITHUB_PERSONAL_ACCESS_TOKEN",
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")},
    ),
]
