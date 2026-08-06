"""Configuration settings and downstream MCP server loader for MCP Orchestrator."""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

# Load environment variables from .env file into os.environ
load_dotenv()

logger = logging.getLogger(__name__)


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


class MCPServerEntry(BaseModel):
    """Pydantic schema for individual server entries in mcp_orc.json."""

    transport: str = "stdio"
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    name: Optional[str] = None


class MCPOrcConfig(BaseModel):
    """Pydantic schema for the root mcp_orc.json file format."""

    mcpServers: Dict[str, MCPServerEntry]


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

    # 3-Term Ranking Weights (Phase 4)
    RANKING_W1_QUERY_SIM: float = float(os.getenv("RANKING_W1_QUERY_SIM", "0.33"))
    RANKING_W2_PROJ_MATCH: float = float(os.getenv("RANKING_W2_PROJ_MATCH", "0.33"))
    RANKING_W3_PAST_SUCCESS: float = float(os.getenv("RANKING_W3_PAST_SUCCESS", "0.34"))

    # Path to external MCP config file
    MCP_CONFIG_PATH: str = os.getenv("MCP_CONFIG_PATH", "mcp_orc.json")

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()


def _resolve_env_value(val: str) -> Tuple[str, bool, str]:
    """Interpolate ${ENV_VAR} patterns in string values.

    Returns:
        Tuple of (resolved_value, is_missing_secret, missing_var_name)
    """
    pattern = re.compile(r"\$\{([^}]+)\}")
    matches = pattern.findall(val)
    if not matches:
        return val, False, ""

    missing_var = ""
    is_missing = False

    def replace_var(m):
        nonlocal missing_var, is_missing
        var_name = m.group(1)
        env_val = os.getenv(var_name)
        if env_val is None or env_val == "":
            is_missing = True
            missing_var = var_name
            return f"${{{var_name}}}"
        return env_val

    resolved = pattern.sub(replace_var, val)
    return resolved, is_missing, missing_var


def load_downstream_servers(config_path: Optional[str] = None) -> List[DownstreamServerConfig]:
    """Load and validate downstream MCP servers from mcp_orc.json or specified path.

    Supports ${ENV_VAR} secret interpolation. Marks missing secrets for graceful skip.
    """
    load_dotenv()
    target_path = config_path or settings.MCP_CONFIG_PATH

    # Fallback to mcp_orc.example.json if target path does not exist
    if not os.path.exists(target_path):
        fallback_path = "mcp_orc.example.json"
        if os.path.exists(fallback_path):
            logger.info(
                f"Config file '{target_path}' not found. Falling back to template '{fallback_path}'."
            )
            target_path = fallback_path
        else:
            raise FileNotFoundError(
                f"MCP Orchestrator config file not found at '{target_path}' or '{fallback_path}'."
            )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        logger.error(f"Malformed JSON in config file '{target_path}': {err}")
        raise ValueError(f"Failed to parse config file '{target_path}': {err}") from err

    try:
        validated_config = MCPOrcConfig.model_validate(raw_data)
    except ValidationError as err:
        logger.error(f"Schema validation error in config file '{target_path}': {err}")
        raise ValueError(f"Config schema validation failed for '{target_path}': {err}") from err

    server_configs: List[DownstreamServerConfig] = []

    for server_id, entry in validated_config.mcpServers.items():
        resolved_env: Dict[str, str] = {}
        requires_auth = False
        auth_var = ""

        for k, v in entry.env.items():
            res_val, missing, missing_var = _resolve_env_value(v)
            resolved_env[k] = res_val
            if missing:
                requires_auth = True
                auth_var = missing_var

        # Determine human readable server name
        readable_name = entry.name or server_id.replace("-", " ").title()

        server_configs.append(
            DownstreamServerConfig(
                id=server_id,
                name=readable_name,
                transport_type=entry.transport,
                command=entry.command,
                args=entry.args,
                env=resolved_env,
                requires_auth=requires_auth,
                auth_env_var=auth_var,
            )
        )

    logger.info(f"Loaded {len(server_configs)} downstream server configurations from '{target_path}'.")
    return server_configs


# Global Default Server Configurations loaded from mcp_orc.json
DEFAULT_DOWNSTREAM_SERVERS: List[DownstreamServerConfig] = load_downstream_servers()
