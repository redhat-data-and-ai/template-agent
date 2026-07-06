"""Template Agent package.

This package provides a template-based agent system for MCP (Model Context Protocol) servers.
"""

from template_agent.src.settings import settings
from template_agent.utils.langfuse_mock import (
    install_langfuse_mock,
    is_langfuse_configured,
)

if not is_langfuse_configured(
    settings.LANGFUSE_PUBLIC_KEY,
    settings.LANGFUSE_SECRET_KEY,
    settings.LANGFUSE_BASE_URL,
):
    install_langfuse_mock()
