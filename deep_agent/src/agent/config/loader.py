"""Agent configuration loader and singleton.

This module provides the main AgentConfig singleton class that orchestrates loading
agent configurations from the config/agent/ directory at the repository root. It
loads the unified runtime/agent.yaml once, then extracts sections for providers,
middleware, and filesystem config. Orchestrator, subagents, skills, and MCP server
configurations are loaded eagerly at initialization time.

Why this exists:
    All agent configurations need to be loaded once and made available throughout
    the application. This singleton ensures configs are loaded only once and
    provides convenient access methods.

Classes:
    AgentConfig: Singleton for managing all agent configuration loading
"""

import json
from pathlib import Path
from typing import Any

import yaml

from deep_agent.src.exceptions import AppException, ErrorCodes
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

from .cache import CacheFileConfig
from .filesystem import FilesystemFileConfig
from .middleware import (
    MiddlewareFileConfig,
    ResolvedMiddlewareConfig,
    resolve_middleware,
)
from .parser import inject_runtime_values, parse_frontmatter
from .providers import ProvidersFileConfig
from .resolver import resolve_skill_paths, resolve_tools

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Default config directory path (repo-root config/agent/)
_AGENT_CONFIG_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "config" / "agent"
)


class AgentConfig:
    """Singleton class for managing agent configuration operations.

    This class provides centralized access to all config/ directory
    operations including loading configurations, resolving paths, and
    managing runtime values.
    """

    _instance: "AgentConfig | None" = None
    _initialized: bool
    _configs_loaded: bool
    _base_dir: Path
    _orchestrator: dict[str, Any]
    _subagents: dict[str, dict[str, Any]]
    _mcp_servers: dict[str, Any]
    _available_skills: dict[str, Path]
    _middleware_config: MiddlewareFileConfig
    _filesystem_config: FilesystemFileConfig
    _providers_config: ProvidersFileConfig
    _cache_config: CacheFileConfig
    _name: str

    def __new__(cls, base_dir: Path | None = None) -> "AgentConfig":
        """Create or return the singleton instance.

        Args:
            base_dir: Optional base directory for config. Only used on first instantiation.

        Returns:
            The singleton AgentConfig instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Path | None = None):
        """Initialize the AgentConfig singleton.

        Args:
            base_dir: Optional base directory for config. Defaults to
                config/ at the repository root relative to this module.
        """
        if self._initialized:
            return

        self._base_dir = base_dir if base_dir is not None else _AGENT_CONFIG_DIR
        self._initialized = True
        self._configs_loaded = False

    def _load_agent_yaml(self) -> dict[str, Any]:
        """Load the unified runtime/agent.yaml once.

        Returns:
            Raw dict from agent.yaml, or empty dict if missing.
        """
        agent_yaml = self._base_dir / "runtime" / "agent.yaml"
        if not agent_yaml.is_file():
            logger.warning("No runtime/agent.yaml found — using defaults")
            return {}

        try:
            raw = yaml.safe_load(agent_yaml.read_text()) or {}
            logger.info("Loaded runtime/agent.yaml")
            return raw
        except Exception as e:
            logger.warning("Failed to parse runtime/agent.yaml, using defaults: %s", e)
            return {}

    def _ensure_loaded(self) -> None:
        """Lazy load configurations on first access.

        This ensures logging is properly configured before we try to log.
        """
        if self._configs_loaded:
            return

        logger.info("Loading agent configurations...")

        raw = self._load_agent_yaml()

        # Extract middleware section (defaults + harness_profiles as profiles)
        self._middleware_config = MiddlewareFileConfig.model_validate(
            {
                "defaults": raw.get("middleware", {}),
                "profiles": raw.get("harness_profiles", {}),
            }
        )

        # Extract filesystem section
        self._filesystem_config = FilesystemFileConfig.model_validate(
            raw.get("filesystem", {})
        )

        # Extract providers section (shares harness_profiles with middleware)
        self._providers_config = ProvidersFileConfig.model_validate(
            {
                "resolve_strategy": raw.get("resolve_strategy", "legacy"),
                "providers": raw.get("providers", {}),
                "harness_profiles": raw.get("harness_profiles", {}),
                "async_tasks": raw.get("async_tasks", {}),
            }
        )

        # Extract cache section
        self._cache_config = CacheFileConfig.model_validate(raw.get("cache", {}))

        # Extract top-level identity
        self._name = raw.get("name", "Agent")
        # Scan skills first, as orchestrator and subagents need them for resolution
        self._available_skills: dict[str, Path] = self._scan_available_skills()
        self._orchestrator: dict[str, Any] = self._load_orchestrator()
        self._subagents: dict[str, dict[str, Any]] = self._load_all_subagents()
        self._mcp_servers: dict[str, Any] = self._load_mcp_servers()

        self._configs_loaded = True
        logger.info(
            f"Agent config loaded: orchestrator={self._orchestrator.get('name')}, "
            f"subagents={len(self._subagents)}, skills={len(self._available_skills)}"
        )

    @property
    def base_dir(self) -> Path:
        """Get the config base directory path."""
        return self._base_dir

    @staticmethod
    def _validate_mcps_field(mcps: Any, agent_name: str) -> None:
        """Validate the ``mcps`` frontmatter field is a list of strings.

        Args:
            mcps: The raw value from frontmatter.
            agent_name: Agent name for error messages.

        Raises:
            AppException: If ``mcps`` is not a list of strings.
        """
        if not isinstance(mcps, list) or not all(
            isinstance(s, str) for s in mcps
        ):
            raise AppException(
                f"Agent '{agent_name}': 'mcps' must be a list of strings",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )

    def _load_orchestrator(self) -> dict[str, Any]:
        """Load orchestrator configuration at initialization.

        Returns:
            Orchestrator config dict with injected runtime values and resolved skill paths.

        Raises:
            AppException: If orchestrator/main.md is missing or invalid.
        """
        orchestrator_path = self._base_dir / "PROMPT.md"
        try:
            config = parse_frontmatter(orchestrator_path)
            if "body" in config:
                config["body"] = inject_runtime_values(config["body"])

            if "mcps" in config:
                self._validate_mcps_field(
                    config["mcps"], config.get("name", "orchestrator")
                )

            # Resolve skill names to paths eagerly
            skill_names = config.get("skills", [])
            if skill_names:
                config["skill_paths"] = resolve_skill_paths(
                    skill_names,
                    self._available_skills,
                    agent_name=config.get("name", "orchestrator"),
                )

            return config
        except FileNotFoundError:
            raise AppException(
                f"Orchestrator config not found at {orchestrator_path}",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )
        except Exception as e:
            raise AppException(
                f"Failed to load orchestrator config: {e}",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )

    def _load_all_subagents(self) -> dict[str, dict[str, Any]]:
        """Load all subagent configurations at initialization.

        Returns:
            Dict mapping subagent name to config dict with resolved skill paths.
        """
        subagents_dir = self._base_dir / "subagents"
        if not subagents_dir.is_dir():
            logger.warning(f"Subagents directory not found at {subagents_dir}")
            return {}

        subagents = {}
        for agent_file in sorted(subagents_dir.glob("*.md")):
            try:
                config = parse_frontmatter(agent_file)
                if "body" in config:
                    config["body"] = inject_runtime_values(config["body"])

                name = config.get("name", agent_file.stem)

                if "mcps" in config:
                    self._validate_mcps_field(config["mcps"], name)

                # Resolve skill names to paths eagerly
                skill_names = config.get("skills", [])
                if skill_names:
                    config["skill_paths"] = resolve_skill_paths(
                        skill_names, self._available_skills, agent_name=name
                    )

                subagents[name] = config
                logger.info(f"Loaded subagent config: {name}")
            except Exception as e:
                logger.error(f"Failed to load subagent {agent_file}: {e}")

        return subagents

    def _load_mcp_servers(self) -> dict[str, Any]:
        """Load MCP server configuration at initialization.

        Returns:
            Dict of MCP server configurations.
        """
        mcp_path = self._base_dir / "mcp.json"
        if not mcp_path.is_file():
            logger.warning(f"MCP config not found at {mcp_path}")
            return {}

        try:
            data = json.loads(mcp_path.read_bytes())
            servers: dict[str, Any] = data.get("mcpServers", {})
            logger.info(f"Loaded {len(servers)} MCP server config(s)")
            return servers
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}

    def _scan_available_skills(self) -> dict[str, Path]:
        """Scan and index all available skills at initialization.

        Returns:
            Dict mapping skill name to skill directory path.
        """
        skills_dir = self._base_dir / "skills"
        if not skills_dir.is_dir():
            logger.warning(f"Skills directory not found at {skills_dir}")
            return {}

        skills = {}
        for skill_path in skills_dir.iterdir():
            if skill_path.is_dir() and not skill_path.name.startswith("."):
                skills[skill_path.name] = skill_path
                logger.debug(f"Found skill: {skill_path.name}")

        logger.info(f"Scanned {len(skills)} available skill(s)")
        return skills

    def get_orchestrator_config(self) -> dict[str, Any]:
        """Get the pre-loaded orchestrator configuration.

        Returns:
            Orchestrator config dict with all fields and injected runtime values.
        """
        self._ensure_loaded()
        return self._orchestrator

    def get_all_subagent_configs(self) -> dict[str, dict[str, Any]]:
        """Get all subagent configurations.

        Returns:
            Dict mapping subagent name to config dict.
        """
        self._ensure_loaded()
        return self._subagents

    @staticmethod
    def resolve_tools(
        tool_names: list[str],
        available_tools: list[Any],
        agent_name: str = "agent",
    ) -> list[Any]:
        """Resolve tool names to actual tool objects.

        This is a static method that delegates to the resolver module.

        Args:
            tool_names: List of tool names from frontmatter.
            available_tools: List of available tool objects.
            agent_name: Name of the agent (for logging).

        Returns:
            List of resolved tool objects.
        """
        return resolve_tools(tool_names, available_tools, agent_name)

    def get_mcp_servers(self) -> dict[str, Any]:
        """Get the pre-loaded MCP server configurations.

        Returns:
            Dict of MCP server configurations.
        """
        self._ensure_loaded()
        return self._mcp_servers

    def get_providers_config(self) -> ProvidersFileConfig:
        """Get the pre-loaded providers configuration.

        Returns:
            The parsed providers.yaml config (strategy, profiles, async tasks).
        """
        self._ensure_loaded()
        return self._providers_config

    def get_filesystem_config(self) -> FilesystemFileConfig:
        """Get the pre-loaded filesystem configuration.

        Returns:
            The parsed filesystem.yaml config (backend, permissions, settings).
        """
        self._ensure_loaded()
        return self._filesystem_config

    def get_cache_config(self) -> CacheFileConfig:
        """Get the pre-loaded cache configuration.

        Returns:
            The parsed cache section (TTLs, feature flags, size limits).
        """
        self._ensure_loaded()
        return self._cache_config

    def get_name(self) -> str:
        """Get the agent display name from config.

        Returns:
            The agent name as configured in agent.yaml (top-level `name` field).
        """
        self._ensure_loaded()
        return self._name

    def get_middleware_config(self) -> MiddlewareFileConfig:
        """Get the pre-loaded middleware file configuration.

        Returns:
            The parsed middleware.yaml config (defaults + profiles).
        """
        self._ensure_loaded()
        return self._middleware_config

    def resolve_agent_middleware(
        self,
        model_name: str,
        agent_overrides: dict[str, Any] | None = None,
    ) -> ResolvedMiddlewareConfig:
        """Resolve middleware config for a specific agent.

        Merges: global defaults → profile (from model) → per-agent overrides.

        Args:
            model_name: Model name from agent frontmatter.
            agent_overrides: Optional middleware: block from frontmatter.

        Returns:
            Fully resolved middleware configuration.
        """
        self._ensure_loaded()
        return resolve_middleware(self._middleware_config, model_name, agent_overrides)

    def get_pyproject_path(self) -> Path:
        """Get the skill sandbox pyproject.toml path.

        Returns:
            Path to config/skills/pyproject.toml for skill sandbox dependencies.
        """
        return self._base_dir / "skills" / "pyproject.toml"


# Singleton instance
agent_config = AgentConfig(_AGENT_CONFIG_DIR)
