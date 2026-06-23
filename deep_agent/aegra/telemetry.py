"""Langfuse and token budget observability for aegra deployment.

Provides:
- Langfuse callback handler factory for LangChain tracing (v4 SDK)
- Langfuse client accessor via ``get_langfuse_client()``
- Token budget LangChain callback registration and metadata provider

Environment variables (Langfuse — auto-read by v4 SDK):
    LANGFUSE_PUBLIC_KEY: Langfuse public key
    LANGFUSE_SECRET_KEY: Langfuse secret key
    LANGFUSE_BASE_URL: Langfuse server URL
    LANGFUSE_TRACING_ENVIRONMENT: Environment tag (e.g. development, production)
"""

import contextvars
import os
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# Langfuse v4 integration
# ---------------------------------------------------------------------------

_langfuse_tracing_initialized = False
_token_budget_tracing_initialized = False


def _get_trace_name() -> str:
    """Resolve trace name: agent.yaml name > env var > fallback."""
    try:
        from deep_agent.src.agent.config import agent_config

        return agent_config.get_name()
    except Exception:
        return os.environ.get("LANGFUSE_TRACE_NAME", "template-agent")


def _langfuse_configured() -> bool:
    """Return True if the minimum Langfuse credentials are present."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def setup_langfuse_tracing() -> None:
    """Register Langfuse as a global LangChain callback and Aegra observability provider.

    Two mechanisms work together:

    1. ``register_configure_hook`` — the same mechanism LangSmith uses to
       auto-inject its tracer. Creates a fresh ``CallbackHandler()`` per run.
    2. ``LangfuseObservabilityProvider`` — plugs into Aegra's
       ``ObservabilityManager`` so that ``create_run_config`` injects
       ``langfuse_user_id``, ``langfuse_session_id``, and
       ``langfuse_trace_name`` into ``RunnableConfig.metadata``.
       The CallbackHandler reads these automatically.

    Must be called **once** at process startup. Subsequent calls are no-ops.
    """
    global _langfuse_tracing_initialized
    if _langfuse_tracing_initialized:
        return
    _langfuse_tracing_initialized = True

    if not _langfuse_configured():
        logger.info("Langfuse credentials not set — auto-tracing disabled")
        return

    try:
        from langchain_core.tracers.context import register_configure_hook
        from langfuse.langchain import CallbackHandler

        _langfuse_ctx_var: contextvars.ContextVar = contextvars.ContextVar(
            "langfuse_handler", default=None
        )

        register_configure_hook(
            _langfuse_ctx_var,
            True,
            CallbackHandler,
            env_var="LANGFUSE_PUBLIC_KEY",
        )
        logger.info("Langfuse auto-tracing registered for all LangChain runs")
    except ImportError:
        logger.warning(
            "langfuse or langchain_core not available — auto-tracing disabled"
        )
        return
    except Exception:
        logger.warning("Failed to register Langfuse tracing hook", exc_info=True)
        return

    try:
        from aegra_api.observability.base import get_observability_manager

        manager = get_observability_manager()
        manager.register_provider(LangfuseObservabilityProvider())
        logger.info("Langfuse observability provider registered with Aegra")
    except ImportError:
        logger.debug("aegra_api not available — skipping provider registration")
    except Exception:
        logger.warning(
            "Failed to register Langfuse observability provider", exc_info=True
        )


class LangfuseObservabilityProvider:
    """Aegra ObservabilityProvider that injects Langfuse metadata into RunnableConfig.

    The Langfuse v4 ``CallbackHandler`` auto-reads these keys from
    ``RunnableConfig.metadata``:

    - ``langfuse_user_id`` — who triggered the run
    - ``langfuse_session_id`` — groups traces by conversation (thread)
    - ``langfuse_trace_name`` — human-readable trace name in the UI
    """

    def get_callbacks(self) -> list[Any]:
        """Return empty list — callbacks are handled by register_configure_hook."""
        return []

    def get_metadata(
        self, run_id: str, thread_id: str, user_identity: str | None = None
    ) -> dict[str, Any]:
        """Return Langfuse metadata keys for RunnableConfig injection."""
        metadata: dict[str, Any] = {
            "langfuse_trace_name": _get_trace_name(),
        }
        if user_identity:
            metadata["langfuse_user_id"] = user_identity
        if thread_id:
            metadata["langfuse_session_id"] = thread_id
        return metadata

    def is_enabled(self) -> bool:
        """Return True if Langfuse credentials are configured."""
        return _langfuse_configured()


# ---------------------------------------------------------------------------
# Token budget callback integration
# ---------------------------------------------------------------------------


class TokenBudgetObservabilityProvider:
    """Inject thread_id into RunnableConfig metadata for the token budget callback."""

    def get_callbacks(self) -> list[Any]:
        """Return empty list — callbacks are handled by register_configure_hook."""
        return []

    def get_metadata(
        self, run_id: str, thread_id: str, user_identity: str | None = None
    ) -> dict[str, Any]:
        from deep_agent.src.token_budget.callback import (
            THREAD_ID_METADATA_KEY,
            USER_ID_METADATA_KEY,
        )

        metadata: dict[str, Any] = {}
        if thread_id:
            metadata[THREAD_ID_METADATA_KEY] = thread_id
        if user_identity:
            metadata[USER_ID_METADATA_KEY] = user_identity
        return metadata

    def is_enabled(self) -> bool:
        try:
            from deep_agent.src.agent.config import agent_config

            return agent_config.get_token_budget_config().is_active
        except Exception:
            return False


def setup_token_budget_tracking() -> None:
    """Register token budget LangChain callback and Aegra metadata provider."""
    global _token_budget_tracing_initialized
    if _token_budget_tracing_initialized:
        return
    _token_budget_tracing_initialized = True

    try:
        from deep_agent.src.agent.config import agent_config

        if not agent_config.get_token_budget_config().is_active:
            logger.info("Token budget disabled — callback registration skipped")
            return
    except Exception:
        logger.debug("Token budget config unavailable — skipping callback registration")
        return

    try:
        from langchain_core.tracers.context import register_configure_hook

        from deep_agent.src.token_budget.callback import TokenBudgetCallbackHandler

        _token_budget_ctx_var: contextvars.ContextVar = contextvars.ContextVar(
            "token_budget_handler", default=None
        )
        os.environ.setdefault("TOKEN_BUDGET_TRACKING", "1")
        register_configure_hook(
            _token_budget_ctx_var,
            True,
            TokenBudgetCallbackHandler,
            env_var="TOKEN_BUDGET_TRACKING",
        )
        logger.info("Token budget callback registered for all LangChain runs")
    except ImportError:
        logger.warning("langchain_core not available — token budget callback disabled")
        return
    except Exception:
        logger.warning("Failed to register token budget callback", exc_info=True)
        return

    try:
        from aegra_api.observability.base import get_observability_manager

        manager = get_observability_manager()
        manager.register_provider(TokenBudgetObservabilityProvider())
        logger.info("Token budget observability provider registered with Aegra")
    except ImportError:
        logger.debug("aegra_api not available — skipping token budget provider")
    except Exception:
        logger.warning(
            "Failed to register token budget observability provider", exc_info=True
        )


def get_langfuse_client() -> Any:
    """Return the Langfuse singleton client (v4), or None if unconfigured.

    Uses ``get_client()`` which auto-reads ``LANGFUSE_PUBLIC_KEY``,
    ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_BASE_URL`` from the environment.
    """
    if not _langfuse_configured():
        return None

    try:
        from langfuse import get_client

        return get_client()
    except ImportError:
        logger.warning("langfuse package not installed — Langfuse tracing disabled")
        return None
    except Exception:
        logger.warning("Failed to initialize Langfuse client", exc_info=True)
        return None
