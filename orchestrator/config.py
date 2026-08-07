"""Configuration settings for MCP Orchestrator loaded from environment or defaults."""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file automatically at startup
load_dotenv()


class MCPServerEntry(BaseModel):
    """Pydantic model representing a single downstream MCP server entry in mcp_orc.json."""

    transport: str = Field(
        default="stdio", description="Transport layer: stdio or sse"
    )
    command: Optional[Any] = Field(
        default=None, description="Executable command string or array for stdio transport"
    )
    args: List[str] = Field(
        default_factory=list, description="Command arguments"
    )
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    url: Optional[str] = Field(
        default=None, description="Target URL for sse transport"
    )
    type: Optional[str] = Field(
        default=None, description="Optional type field (local -> stdio)"
    )
    enabled: Optional[bool] = Field(
        default=True, description="Whether server is enabled"
    )

    @model_validator(mode="after")
    def normalize_and_validate(self):
        # Support type: "local" as alias for transport: "stdio"
        if self.type == "local":
            self.transport = "stdio"

        # If command is provided as a list e.g. ["uvx", "mcp-server-fetch"]
        if isinstance(self.command, list):
            if len(self.command) > 0:
                cmd_list = self.command
                self.command = cmd_list[0]
                if len(cmd_list) > 1 and not self.args:
                    self.args = cmd_list[1:]
            else:
                self.command = None

        if self.transport == "stdio" and not self.command:
            raise ValueError("Command is required for stdio transport.")
        return self


class MCPOrcConfig(BaseModel):
    """Pydantic model representing the full mcp_orc.json structure."""

    mcpServers: Dict[str, MCPServerEntry]


class DownstreamServerConfig(BaseModel):
    """Internal downstream server config representation."""

    id: str
    name: str
    transport_type: str  # "stdio" or "sse"
    command: str
    args: List[str]
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    requires_auth: bool = False
    auth_env_var: Optional[str] = None


class Settings(BaseSettings):
    """System-wide configuration settings."""

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database configuration
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/mcporchestrator"

    # ChromaDB configuration
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8001

    # Two-Stage Retrieval configuration
    TOP_N_SERVERS: int = 3
    TOP_M_TOOLS: int = 10

    # Phase 4 Ranking Weights (w1 + w2 + w3 = 1.0)
    RANKING_W1_QUERY_SIM: float = 0.33
    RANKING_W2_PROJ_MATCH: float = 0.33
    RANKING_W3_PAST_SUCCESS: float = 0.34

    # Phase 6 Bypass Mode Toggle (Naive Baseline)
    ORCHESTRATOR_BYPASS_MODE: bool = False

    # Minimum Relevance Threshold for query_sim (suppresses irrelevant/conversational queries)
    MIN_RELEVANCE_THRESHOLD: float = 0.15

    # Dynamic MCP Server Registry Config File Path
    MCP_ORC_CONFIG_PATH: str = "mcp_orc.json"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()


def resolve_env_vars(text: str) -> tuple[str, bool]:
    """Interpolate ${ENV_VAR} placeholders with environment values.

    Returns:
        tuple[str, bool]: (resolved_text, has_unresolved_env_var)
    """

    def replace_var(match):
        var_name = match.group(1)
        val = os.getenv(var_name, "")
        return val

    # Match ${VAR_NAME}
    pattern = r"\$\{([A-Za-z0-9_]+)\}"

    # Find if any referenced env var is missing or empty
    has_unresolved = False
    for m in re.finditer(pattern, text):
        var_name = m.group(1)
        val = os.getenv(var_name, "")
        if not val or not val.strip():
            has_unresolved = True

    resolved_str = re.sub(pattern, replace_var, text)
    return resolved_str, has_unresolved


def load_downstream_servers(
    config_path: Optional[str] = None,
    raise_on_error: bool = False,
) -> List[DownstreamServerConfig]:
    """Parse downstream servers from mcp_orc.json with environment variable interpolation.

    Falls back to DEFAULT_DOWNSTREAM_SERVERS if config file does not exist.
    """
    path = config_path or settings.MCP_ORC_CONFIG_PATH

    if not os.path.exists(path):
        if raise_on_error:
            raise ValueError(f"Config file does not exist: {path}")
        return DEFAULT_DOWNSTREAM_SERVERS

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        resolved_json_str, has_unresolved = resolve_env_vars(raw_content)

        try:
            raw_dict = json.loads(resolved_json_str)
        except Exception as json_err:
            if raise_on_error:
                raise ValueError(f"Failed to parse config file '{path}': {json_err}") from json_err
            return DEFAULT_DOWNSTREAM_SERVERS

        try:
            config_obj = MCPOrcConfig.model_validate(raw_dict)
        except Exception as val_err:
            if raise_on_error:
                raise ValueError(f"Config schema validation failed for '{path}': {val_err}") from val_err
            return DEFAULT_DOWNSTREAM_SERVERS

        result: List[DownstreamServerConfig] = []
        for server_id, entry in config_obj.mcpServers.items():
            requires_auth = False
            auth_var = None
            for k, v in entry.env.items():
                if v == "" or not v.strip():
                    requires_auth = True
                    auth_var = k

            cfg = DownstreamServerConfig(
                id=server_id,
                name=server_id.replace("-", " ").title(),
                transport_type=entry.transport,
                command=entry.command or "",
                args=entry.args,
                env=entry.env,
                url=entry.url,
                requires_auth=requires_auth,
                auth_env_var=auth_var,
            )
            result.append(cfg)

        return result
    except Exception as e:
        if raise_on_error:
            raise ValueError(f"Failed to load downstream servers from '{path}': {e}") from e
        return DEFAULT_DOWNSTREAM_SERVERS


# Hardcoded Fallback Registry
DEFAULT_DOWNSTREAM_SERVERS = [
    DownstreamServerConfig(
        id="filesystem-server",
        name="Filesystem MCP Server",
        transport_type="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
]
