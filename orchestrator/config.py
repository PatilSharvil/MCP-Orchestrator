"""Configuration settings for MCP Orchestrator loaded from environment or defaults."""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file automatically at startup
load_dotenv()
logger = logging.getLogger(__name__)


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
        # Support type: "local" -> transport: "stdio", type: "remote"/"sse" -> transport: "sse"
        if self.type == "local":
            self.transport = "stdio"
        elif self.type in ("remote", "sse"):
            self.transport = "sse"

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


def ensure_global_user_config() -> str:
    """Ensure ~/.mcp-orchestrator/mcp_orc.json exists with default template if missing."""
    user_dir = os.path.expanduser("~/.mcp-orchestrator")
    os.makedirs(user_dir, exist_ok=True)
    global_config_path = os.path.join(user_dir, "mcp_orc.json")

    if not os.path.exists(global_config_path):
        sample_template = {
            "_info": "MCP Orchestrator Global Downstream Server Registry",
            "_guide_option_A": "Option A (Simplicity): Paste raw secret strings directly in 'env' (e.g. 'BRAVE_API_KEY': 'BSA_xyz123')",
            "_guide_option_B": "Option B (Security): Use '${VAR_NAME}' template syntax (e.g. 'GITHUB_TOKEN': '${GITHUB_TOKEN}') and keep secrets in .env",
            "mcpServers": {
                "filesystem-server": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                    "env": {}
                },
                "memory-server": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                    "env": {}
                },
                "github-server": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
                    }
                }
            }
        }
        try:
            with open(global_config_path, "w", encoding="utf-8") as f:
                json.dump(sample_template, f, indent=2)
            logger.info(f"Created default global MCP Orchestrator config at '{global_config_path}'")
        except Exception as e:
            logger.warning(f"Could not write default global config to '{global_config_path}': {e}")

    return global_config_path


def load_downstream_servers(
    config_path: Optional[str] = None,
    raise_on_error: bool = False,
) -> List[DownstreamServerConfig]:
    """Parse downstream servers from mcp_orc.json with environment variable interpolation.

    Resolution order:
    1. Explicit config_path parameter or custom MCP_ORC_CONFIG_PATH env var
    2. Global AppData config (~/.mcp-orchestrator/mcp_orc.json)
    3. Local mcp_orc.json in CWD
    4. DEFAULT_DOWNSTREAM_SERVERS fallback
    """
    # 1. Always ensure global user config directory and mcp_orc.json exist
    global_user_path = ensure_global_user_config()

    # Determine file path
    path = config_path
    if not path:
        if settings.MCP_ORC_CONFIG_PATH != "mcp_orc.json" and os.path.exists(settings.MCP_ORC_CONFIG_PATH):
            path = settings.MCP_ORC_CONFIG_PATH
        elif os.path.exists(global_user_path):
            path = global_user_path
        elif os.path.exists("mcp_orc.json"):
            path = "mcp_orc.json"
        else:
            path = "mcp_orc.json"

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
            if entry.enabled is False:
                continue
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
