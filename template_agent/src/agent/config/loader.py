"""Agent configuration loader and singleton.

This module provides the main AgentConfig singleton class that orchestrates loading
agent configurations from the agent_config/ directory. It eagerly loads orchestrator,
subagents, skills, and MCP server configurations at initialization time.

Why this exists:
    All agent configurations (orchestrator, subagents, skills, MCP servers) need
    to be loaded once and made available throughout the application. This singleton
    ensures configs are loaded only once and provides convenient access methods.

Classes:
    AgentConfig: Singleton for managing all agent configuration loading
"""

import json
from pathlib import Path
from typing import Any

from template_agent.src.exceptions import AppException, ErrorCodes
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

from .parser import inject_runtime_values, parse_frontmatter
from .resolver import resolve_skill_paths, resolve_tools

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Default agent_config directory path
_AGENT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "agent_config"


class AgentConfig:
    """Singleton class for managing agent configuration operations.

    This class provides centralized access to all agent_config directory
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

    def __new__(cls, base_dir: Path | None = None) -> "AgentConfig":
        """Create or return the singleton instance.

        Args:
            base_dir: Optional base directory for agent_config. Only used on first instantiation.

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
            base_dir: Optional base directory for agent_config. Defaults to
                template_agent/agent_config relative to this module.
        """
        if self._initialized:
            return

        self._base_dir = base_dir if base_dir is not None else _AGENT_CONFIG_DIR
        self._initialized = True
        self._configs_loaded = False

    def _ensure_loaded(self):
        """Lazy load configurations on first access.

        This ensures logging is properly configured before we try to log.
        """
        if self._configs_loaded:
            return

        logger.info("Loading agent configurations...")
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
        """Get the agent_config base directory path."""
        return self._base_dir

    def _load_orchestrator(self) -> dict[str, Any]:
        """Load orchestrator configuration at initialization.

        Returns:
            Orchestrator config dict with injected runtime values and resolved skill paths.

        Raises:
            AppException: If orchestrator/main.md is missing or invalid.
        """
        orchestrator_path = self._base_dir / "orchestrator" / "main.md"
        try:
            config = parse_frontmatter(orchestrator_path)
            if "body" in config:
                config["body"] = inject_runtime_values(config["body"])

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
            servers = data.get("mcpServers", {})
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

    def get_pyproject_path(self) -> Path:
        """Get the backend pyproject.toml path.

        Returns:
            Path to agent_config/pyproject.toml for backend dependencies.
        """
        return self._base_dir / "pyproject.toml"


# Singleton instance
agent_config = AgentConfig(_AGENT_CONFIG_DIR)
