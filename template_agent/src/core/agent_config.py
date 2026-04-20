"""Agent configuration utilities.

Central module for loading and processing agent configurations from
the agent_config directory. Handles frontmatter parsing, skill/tool
resolution, and runtime value injection.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from template_agent.src.core.exceptions import AppException, ErrorCodes
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Default agent_config directory path
_AGENT_CONFIG_DIR = Path(__file__).parent.parent.parent / "agent_config"


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

    @staticmethod
    def _get_current_date() -> str:
        """Get the current date in a formatted string.

        Returns:
            The current date formatted as "Month Day, Year" (e.g., "December 25, 2024").
        """
        return datetime.now().strftime("%B %d, %Y")

    @staticmethod
    def _inject_runtime_values(content: str) -> str:
        """Inject runtime values into content.

        Args:
            content: String content with template variables.

        Returns:
            Content with template variables replaced.
        """
        return content.replace("{{current_date}}", AgentConfig._get_current_date())

    def _load_orchestrator(self) -> dict[str, Any]:
        """Load orchestrator configuration at initialization.

        Returns:
            Orchestrator config dict with injected runtime values and resolved skill paths.

        Raises:
            AppException: If orchestrator/main.md is missing or invalid.
        """
        orchestrator_path = self._base_dir / "orchestrator" / "main.md"
        try:
            config = AgentConfig._parse_frontmatter(orchestrator_path)
            if "body" in config:
                config["body"] = AgentConfig._inject_runtime_values(config["body"])

            # Resolve skill names to paths eagerly
            skill_names = config.get("skills", [])
            if skill_names:
                config["skill_paths"] = self._resolve_skill_paths(
                    skill_names, agent_name=config.get("name", "orchestrator")
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
                config = AgentConfig._parse_frontmatter(agent_file)
                if "body" in config:
                    config["body"] = AgentConfig._inject_runtime_values(config["body"])

                name = config.get("name", agent_file.stem)

                # Resolve skill names to paths eagerly
                skill_names = config.get("skills", [])
                if skill_names:
                    config["skill_paths"] = self._resolve_skill_paths(
                        skill_names, agent_name=name
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
            import json

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

    @staticmethod
    def _parse_frontmatter(path: Path) -> dict[str, Any]:
        r"""Parse a markdown file with YAML frontmatter.

        Expects the format: ``--- \n <yaml> \n --- \n <markdown body>``.
        The markdown body is returned under the ``"body"`` key.

        Args:
            path: Path to the ``.md`` file.

        Returns:
            A dict of frontmatter fields plus ``body`` (the markdown content).
        """
        content = path.read_text()
        if not content.startswith("---"):
            return {"body": content.strip()}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {"body": content.strip()}

        frontmatter: dict[str, Any] = yaml.safe_load(parts[1]) or {}
        frontmatter["body"] = parts[2].strip()
        return frontmatter

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

    def _resolve_skill_paths(
        self,
        skill_names: list[str],
        agent_name: str = "agent",
    ) -> list[str]:
        """Resolve skill names to skill directory paths using cached skill index.

        Args:
            skill_names: List of skill names from frontmatter.
            agent_name: Name of the agent (for logging).

        Returns:
            List of skill directory paths as strings.
        """
        skill_paths: list[str] = []
        missing: list[str] = []

        for skill_name in skill_names:
            if skill_name in self._available_skills:
                skill_path = self._available_skills[skill_name]
                skill_paths.append(str(skill_path))
                logger.debug(f"Agent '{agent_name}' resolved skill: {skill_name}")
            else:
                missing.append(skill_name)

        if missing:
            logger.warning(f"Agent '{agent_name}' references unknown skills: {missing}")

        return skill_paths

    @staticmethod
    def resolve_tools(
        tool_names: list[str],
        available_tools: list[Any],
        agent_name: str = "agent",
    ) -> list[Any]:
        """Resolve tool names to actual tool objects.

        Args:
            tool_names: List of tool names from frontmatter.
            available_tools: List of available tool objects.
            agent_name: Name of the agent (for logging).

        Returns:
            List of resolved tool objects.
        """
        tool_by_name = {t.name: t for t in available_tools}
        resolved = [tool_by_name[n] for n in tool_names if n in tool_by_name]
        missing = [n for n in tool_names if n not in tool_by_name]

        if missing:
            logger.warning(f"Agent '{agent_name}' references unknown tools: {missing}")

        return resolved

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
