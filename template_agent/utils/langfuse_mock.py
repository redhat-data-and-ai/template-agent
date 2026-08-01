"""No-op Langfuse stand-ins when credentials are not configured."""

import sys
import types
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def is_langfuse_configured(
    public_key: Optional[str],
    secret_key: Optional[str],
    base_url: Optional[str],
) -> bool:
    """Return True when all Langfuse credential values are non-empty."""
    return all((value or "").strip() for value in (public_key, secret_key, base_url))


class NoOpLangfuse:
    """Langfuse client stub that accepts calls without sending observability data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept Langfuse client initialization arguments without side effects."""
        pass

    def score(self, *args: Any, **kwargs: Any) -> None:
        """No-op score call."""
        return None

    def flush(self, *args: Any, **kwargs: Any) -> None:
        """No-op flush call."""
        return None

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        """No-op shutdown call."""
        return None


class NoOpCallbackHandler(BaseCallbackHandler):
    """LangChain callback stub with default no-op handler methods."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept callback handler initialization arguments without side effects."""
        super().__init__()


def install_langfuse_mock() -> None:
    """Register mock Langfuse modules so the real SDK is never imported."""
    if "langfuse" in sys.modules:
        return

    langfuse_module = types.ModuleType("langfuse")
    langfuse_module.Langfuse = NoOpLangfuse  # type: ignore[attr-defined]

    callback_module = types.ModuleType("langfuse.callback")
    callback_module.CallbackHandler = NoOpCallbackHandler  # type: ignore[attr-defined]

    sys.modules["langfuse"] = langfuse_module
    sys.modules["langfuse.callback"] = callback_module

    logger.info("Langfuse credentials not configured; using no-op mock implementations")
