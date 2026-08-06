"""Tests for Phase 3.5: External MCP Server Config Loader (mcp_orc.json).

Verifies:
1. Pydantic schema validation for JSON configuration.
2. ${ENV_VAR} secret interpolation against environment variables.
3. Graceful handling of missing auth environment variables.
4. Exception raising on malformed JSON or invalid schema.
5. Deterministic test fixture isolation.
"""

import os
import json
import tempfile
import pytest

from orchestrator.config import load_downstream_servers, MCPOrcConfig, DownstreamServerConfig


def test_valid_config_loading_and_interpolation(monkeypatch):
    """Test loading valid config JSON with env var interpolation."""
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_1234567890abcdef")

    test_json = {
        "mcpServers": {
            "test-filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "env": {}
            },
            "test-github": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "${TEST_GITHUB_TOKEN}"
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(test_json, f)
        temp_path = f.name

    try:
        servers = load_downstream_servers(config_path=temp_path)
        assert len(servers) == 2

        fs_cfg = next(s for s in servers if s.id == "test-filesystem")
        assert fs_cfg.command == "npx"
        assert fs_cfg.requires_auth is False

        gh_cfg = next(s for s in servers if s.id == "test-github")
        assert gh_cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_1234567890abcdef"
        assert gh_cfg.requires_auth is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_missing_env_var_flagging(monkeypatch):
    """Test missing environment variable marks requires_auth=True."""
    monkeypatch.delenv("MISSING_SECRET_VAR", raising=False)

    test_json = {
        "mcpServers": {
            "auth-server": {
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
                "env": {
                    "API_KEY": "${MISSING_SECRET_VAR}"
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(test_json, f)
        temp_path = f.name

    try:
        servers = load_downstream_servers(config_path=temp_path)
        assert len(servers) == 1
        cfg = servers[0]
        assert cfg.requires_auth is True
        assert cfg.auth_env_var == "API_KEY"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_malformed_json_raises_value_error():
    """Test malformed JSON raises descriptive ValueError."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{ invalid json content }")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Failed to parse config file"):
            load_downstream_servers(config_path=temp_path, raise_on_error=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_schema_validation_error():
    """Test missing required field (command) raises ValueError."""
    invalid_schema = {
        "mcpServers": {
            "invalid-server": {
                "transport": "stdio"
                # Missing required 'command' field
            }
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(invalid_schema, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Config schema validation failed"):
            load_downstream_servers(config_path=temp_path, raise_on_error=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
