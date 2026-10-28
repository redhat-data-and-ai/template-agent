"""Atlan uses ATLAN_MCP_API_KEY when AI Factory materialized a DCR server."""

import json

from deep_agent.src.agent.config import AgentConfig

_PROMPT = """---
name: dva-lite-v1
model: claude-sonnet-4-6@default
mcps:
  - dataverse-mcp
  - atlan-mcp
---

Test prompt.
"""

_PROMPT_WITHOUT_ATLAN = """---
name: dva-lite-v1
model: claude-sonnet-4-6@default
mcps:
  - dataverse-mcp
---

Test prompt.
"""


def _write_config(tmp_path, prompt: str, servers: dict) -> None:
    config_dir = tmp_path / "agent"
    config_dir.mkdir()
    (config_dir / "PROMPT.md").write_text(prompt)
    (config_dir / "mcp.json").write_text(json.dumps({"mcpServers": servers}))


def _load(tmp_path) -> AgentConfig:
    AgentConfig._instance = None
    return AgentConfig(tmp_path / "agent")


class TestAtlanApiKeyFromEnv:
    def setup_method(self):
        AgentConfig._instance = None

    def teardown_method(self):
        AgentConfig._instance = None

    def test_dcr_server_switches_to_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAN_MCP_API_KEY", "test-key")
        _write_config(
            tmp_path,
            _PROMPT,
            {
                "atlan-mcp": {
                    "url": "https://redhat.atlan.com/mcp",
                    "transport": "streamable_http",
                    "enabled": True,
                    "auth": True,
                    "auth_mode": "dcr",
                    "oauth": {
                        "registration_endpoint": "https://redhat.atlan.com/register",
                        "authorization_endpoint": "https://redhat.atlan.com/authorize",
                        "token_endpoint": "https://redhat.atlan.com/token",
                    },
                }
            },
        )

        servers = _load(tmp_path).get_mcp_servers()

        assert servers["atlan-mcp"]["auth_mode"] == "api_key"
        assert servers["atlan-mcp"]["auth_env_var"] == "ATLAN_MCP_API_KEY"
        assert servers["atlan-mcp"]["url"] == "https://redhat.atlan.com/mcp"
        assert "oauth" not in servers["atlan-mcp"]

    async def test_dcr_config_connects_with_api_key_not_user_token(
        self, tmp_path, monkeypatch
    ):
        """A mounted DCR Atlan server still sends ATLAN_MCP_API_KEY."""
        monkeypatch.setenv("ATLAN_MCP_API_KEY", "test-key")
        _write_config(
            tmp_path,
            _PROMPT,
            {
                "atlan-mcp": {
                    "url": "https://mcp.atlan.com/mcp",
                    "transport": "streamable_http",
                    "enabled": True,
                    "auth": True,
                    "auth_mode": "dcr",
                    "oauth": {
                        "registration_endpoint": "https://mcp.atlan.com/oauth/register",
                        "authorization_endpoint": "https://mcp.atlan.com/oauth/authorize",
                        "token_endpoint": "https://mcp.atlan.com/oauth/token",
                        "scopes": ["openid", "profile", "email"],
                    },
                },
                "dataverse-mcp": {
                    "url": "https://mcp.dataverse.redhat.com/mcp/",
                    "transport": "streamable_http",
                    "enabled": True,
                    "auth": True,
                    "auth_mode": "sso",
                },
            },
        )

        from deep_agent.aegra.mcp import _build_server_config, _resolve_connection_token
        from deep_agent.aegra.mcp_oauth_handlers import _interactive_oauth_servers

        cfg = _load(tmp_path)
        entry = cfg.get_mcp_servers()["atlan-mcp"]
        monkeypatch.setattr(
            "deep_agent.aegra.mcp_oauth_handlers.agent_config",
            cfg,
        )

        token = await _resolve_connection_token(
            "atlan-mcp",
            entry,
            sso_token="user-sso-token",
            user_id="user-1",
        )
        headers = _build_server_config(entry, token)["headers"]

        assert entry["auth_mode"] == "api_key"
        assert token == "test-key"
        assert headers["Authorization"] == "Bearer test-key"
        assert "atlan-mcp" not in [name for name, _ in _interactive_oauth_servers()]

    def test_missing_key_leaves_dcr_in_place(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ATLAN_MCP_API_KEY", raising=False)
        _write_config(
            tmp_path,
            _PROMPT,
            {
                "atlan-mcp": {
                    "url": "https://mcp.atlan.com/mcp",
                    "auth_mode": "dcr",
                    "oauth": {"token_endpoint": "https://mcp.atlan.com/oauth/token"},
                }
            },
        )

        servers = _load(tmp_path).get_mcp_servers()

        assert servers["atlan-mcp"]["auth_mode"] == "dcr"
        assert "oauth" in servers["atlan-mcp"]

    def test_adds_atlan_when_package_was_not_selected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAN_MCP_API_KEY", "test-key")
        monkeypatch.setenv("ATLAN_MCP_URL", "https://mcp.atlan.com/mcp")
        _write_config(
            tmp_path,
            _PROMPT_WITHOUT_ATLAN,
            {
                "dataverse-mcp": {
                    "url": "https://mcp.dataverse.redhat.com/mcp/",
                    "auth_mode": "sso",
                    "enabled": True,
                }
            },
        )

        cfg = _load(tmp_path)

        assert cfg.get_mcp_servers()["atlan-mcp"]["auth_mode"] == "api_key"
        assert cfg.get_mcp_servers()["atlan-mcp"]["url"] == "https://mcp.atlan.com/mcp"
        assert cfg.get_orchestrator_config()["mcps"] == ["dataverse-mcp", "atlan-mcp"]

    def test_uses_redhat_atlan_url_when_url_env_is_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAN_MCP_API_KEY", "test-key")
        monkeypatch.delenv("ATLAN_MCP_URL", raising=False)
        _write_config(
            tmp_path,
            _PROMPT_WITHOUT_ATLAN,
            {
                "dataverse-mcp": {
                    "url": "https://mcp.dataverse.redhat.com/mcp/",
                    "auth_mode": "sso",
                    "enabled": True,
                }
            },
        )

        servers = _load(tmp_path).get_mcp_servers()

        assert servers["atlan-mcp"]["url"] == "https://redhat.atlan.com/mcp"
        assert servers["atlan-mcp"]["auth_mode"] == "api_key"

    def test_url_env_overrides_mcp_json_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAN_MCP_API_KEY", "test-key")
        monkeypatch.setenv("ATLAN_MCP_URL", "https://redhat.atlan.com/mcp")
        _write_config(
            tmp_path,
            _PROMPT,
            {
                "atlan-mcp": {
                    "url": "https://mcp.atlan.com/mcp",
                    "transport": "streamable_http",
                    "enabled": True,
                    "auth": True,
                    "auth_mode": "dcr",
                    "oauth": {
                        "registration_endpoint": "https://mcp.atlan.com/oauth/register",
                        "authorization_endpoint": "https://mcp.atlan.com/oauth/authorize",
                        "token_endpoint": "https://mcp.atlan.com/oauth/token",
                    },
                }
            },
        )

        servers = _load(tmp_path).get_mcp_servers()

        assert servers["atlan-mcp"]["url"] == "https://redhat.atlan.com/mcp"
        assert servers["atlan-mcp"]["auth_mode"] == "api_key"
        assert "oauth" not in servers["atlan-mcp"]
