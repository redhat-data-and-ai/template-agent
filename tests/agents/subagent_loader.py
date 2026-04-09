"""Load subagent configurations for testing."""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from deepagents import SubAgent

from mock_tools import MOCK_TOOLS


def parse_agent_frontmatter(path: Path) -> Dict[str, Any]:
    """Parse markdown file with YAML frontmatter."""
    content = path.read_text()
    if not content.startswith("---"):
        return {"body": content.strip()}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {"body": content.strip()}

    frontmatter = yaml.safe_load(parts[1]) or {}
    frontmatter["body"] = parts[2].strip()
    return frontmatter


def load_subagents(
    agents_dir: Path,
    skills_dir: Path,
) -> List[SubAgent]:
    """
    Load subagent configurations from agents/*.md files.

    Args:
        agents_dir: Path to agents directory
        skills_dir: Path to skills directory

    Returns:
        List of configured SubAgent objects
    """
    subagents = []

    for agent_file in sorted(agents_dir.glob("*.md")):
        config = parse_agent_frontmatter(agent_file)
        name = config.get("name", agent_file.stem)

        # Create base subagent
        subagent = SubAgent(
            name=name,
            description=config.get("description", ""),
            system_prompt=config.get("body", ""),
        )

        # Add tools
        tool_names = config.get("tools", [])
        if tool_names:
            tools = [MOCK_TOOLS[name] for name in tool_names if name in MOCK_TOOLS]
            if tools:
                subagent["tools"] = tools

        # Add skills
        skill_names = config.get("skills", [])
        if skill_names:
            skill_paths = []
            for skill_name in skill_names:
                skill_path = skills_dir / skill_name
                if skill_path.exists():
                    skill_paths.append(str(skill_path.resolve()))
            if skill_paths:
                subagent["skills"] = skill_paths

        # Add model if specified
        model_name = config.get("model")
        if model_name:
            subagent["model"] = model_name

        subagents.append(subagent)

    return subagents
