"""Load subagent configurations for testing."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.auth
import httpx
from deepagents import SubAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from mock_tools import MOCK_TOOLS

from template_agent.src.core.frontmatter import parse_agent_frontmatter


def load_subagents(
    agents_dir: Path,
    skills_dir: Path,
    default_model: Optional[Any] = None,
) -> List[SubAgent]:
    """
    Load subagent configurations from agents/*.md files.

    Args:
        agents_dir: Path to agents directory
        skills_dir: Path to skills directory
        default_model: Default model to use if subagent doesn't specify one

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

        # Add model if specified in config
        model_name = config.get("model")
        if model_name:
            # Create ChatGoogleGenerativeAI instance for the specified model
            # Skip if Google credentials not available
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                try:
                    credentials, project = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    # Disable HTTP keepalive to prevent stale TLS connections
                    _no_keepalive = httpx.Limits(max_keepalive_connections=0)

                    subagent_model = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0,
                        credentials=credentials,
                        project=project,
                        client_args={"limits": _no_keepalive},
                    )
                    subagent["model"] = subagent_model
                except Exception:
                    # If model creation fails, use default model if provided
                    if default_model:
                        subagent["model"] = default_model
        elif default_model:
            # Use default model if no model specified in config
            subagent["model"] = default_model

        subagents.append(subagent)

    return subagents
