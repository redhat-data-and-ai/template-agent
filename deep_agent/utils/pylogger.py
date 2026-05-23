"""Structured logger utility for the template-agent.

Provides a single ``get_python_logger()`` entry point that returns a
structlog ``BoundLogger``.  All log output is structured JSON by default
(production), with an optional human-readable console renderer for
local development.

Environment variables:
    LOG_FORMAT: ``json`` (default) or ``console``
    PYTHON_LOG_LEVEL: standard level name (default: INFO)

Context binding:
    ``bind_request_context(trace_id, user_id, thread_id)`` adds
    per-request fields to every subsequent log line in the same
    async context.  Call at request entry; structlog's context-var
    support auto-clears on context exit.
"""

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Third-party logger noise suppression
# ---------------------------------------------------------------------------

HTTP_CLIENT_LOGGERS = {
    "urllib3",
    "urllib3.connectionpool",
    "urllib3.util",
    "urllib3.util.retry",
    "requests",
    "httpx",
}

AWS_LOGGERS = {
    "botocore",
    "botocore.client",
    "botocore.credentials",
    "botocore.httpsession",
    "boto3",
    "boto3.resources",
}

MCP_LOGGERS = {
    "fastmcp",
    "fastmcp.server",
    "fastmcp.server.http",
    "fastmcp.utilities",
    "fastmcp.utilities.logging",
    "fastmcp.client",
    "fastmcp.transports",
}

ML_AI_LOGGERS = {
    "sentence_transformers",
    "transformers",
    "transformers.models",
    "transformers.tokenization_utils",
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "huggingface_hub",
    "huggingface_hub.utils",
    "langchain_huggingface",
    "torch",
    "torch.nn",
}

OBSERVABILITY_LOGGERS = {
    "langfuse",
    "langfuse.client",
    "langfuse.api",
    "langfuse.callback",
}

SILENT_LOGGERS: set[str] = set()

THIRD_PARTY_LOGGERS: set[str] = (
    HTTP_CLIENT_LOGGERS
    | AWS_LOGGERS
    | MCP_LOGGERS
    | ML_AI_LOGGERS
    | OBSERVABILITY_LOGGERS
    | SILENT_LOGGERS
)

ERROR_ONLY_LOGGERS: set[str] = ML_AI_LOGGERS | OBSERVABILITY_LOGGERS

_LOGGING_CONFIGURED = False

SERVICE_NAME = os.environ.get("SERVICE_NAME", "template-agent")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json").lower()

for _name in SILENT_LOGGERS:
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Request context (per-request fields via contextvars)
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
_thread_id_var: ContextVar[str | None] = ContextVar("thread_id", default=None)


def bind_request_context(
    trace_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> None:
    """Bind per-request identifiers into the logging context.

    Call this once at request entry. The values are automatically
    injected into every log line within the same async context.
    """
    if trace_id:
        _trace_id_var.set(trace_id)
    if user_id:
        _user_id_var.set(user_id)
    if thread_id:
        _thread_id_var.set(thread_id)


def clear_request_context() -> None:
    """Reset request context (called at request exit)."""
    _trace_id_var.set(None)
    _user_id_var.set(None)
    _thread_id_var.set(None)


def _inject_request_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor: inject request context vars into every log event."""
    rid = _trace_id_var.get()
    uid = _user_id_var.get()
    tid = _thread_id_var.get()
    if rid:
        event_dict["trace_id"] = rid
    if uid:
        event_dict["user_id"] = uid
    if tid:
        event_dict["thread_id"] = tid
    event_dict["service"] = SERVICE_NAME
    return event_dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clear_handlers(logger: logging.Logger) -> None:
    logger.handlers.clear()
    logger.filters.clear()


def _setup_logger(logger_name: str, level: str) -> None:
    lgr = logging.getLogger(logger_name)
    _clear_handlers(lgr)
    if logger_name in SILENT_LOGGERS:
        lgr.setLevel(logging.CRITICAL)
    elif logger_name in ERROR_ONLY_LOGGERS:
        lgr.setLevel(logging.ERROR)
    else:
        lgr.setLevel(level)
    lgr.propagate = True


def _configure_third_party_loggers(log_level: str) -> None:
    """Apply structured logging to selected third-party loggers."""
    logging.getLogger().handlers.clear()
    for name in THIRD_PARTY_LOGGERS:
        _setup_logger(name, log_level)


def _get_renderer() -> Any:
    """Return the appropriate structlog renderer based on LOG_FORMAT."""
    if LOG_FORMAT == "console":
        return structlog.dev.ConsoleRenderer(colors=True)
    return structlog.processors.JSONRenderer()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def force_reconfigure_all_loggers(log_level: str = "INFO") -> None:
    """Force logger reconfiguration, even if already initialized."""
    global _LOGGING_CONFIGURED  # noqa: PLW0603
    _LOGGING_CONFIGURED = False
    get_python_logger(log_level)


def get_python_logger(log_level: str = "INFO") -> structlog.BoundLogger:
    """Get a configured structlog logger.

    First call configures the entire logging pipeline. Subsequent
    calls return cached loggers from structlog.
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603
    log_level = log_level.upper()

    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
        )

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                _inject_request_context,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                _get_renderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        _LOGGING_CONFIGURED = True

    _configure_third_party_loggers(log_level)
    return structlog.get_logger()


def get_uvicorn_log_config(log_level: str = "INFO") -> dict[str, Any]:
    """Return a Uvicorn-compatible logging config that integrates with structlog."""
    log_level = log_level.upper()
    renderer = _get_renderer()

    default_formatter = {
        "()": "structlog.stdlib.ProcessorFormatter",
        "processor": renderer,
        "foreign_pre_chain": [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _inject_request_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ],
    }

    def make_logger_config(names: list[str], level: str) -> dict[str, Any]:
        return {
            name: {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            }
            for name in names
        }

    passthrough_formatter = {"format": "%(message)s"}

    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.asgi", "uvicorn.protocols"]
    access_loggers = ["uvicorn.access"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": default_formatter,
            "access": default_formatter,
            "passthrough": passthrough_formatter,
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "passthrough": {
                "formatter": "passthrough",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {
                "handlers": ["passthrough"],
                "level": log_level,
                "propagate": False,
            },
            **make_logger_config(uvicorn_loggers, log_level),
            **make_logger_config(access_loggers, log_level),
            **make_logger_config(
                list(THIRD_PARTY_LOGGERS - ERROR_ONLY_LOGGERS - SILENT_LOGGERS),
                log_level,
            ),
            **make_logger_config(list(ERROR_ONLY_LOGGERS), "ERROR"),
            **make_logger_config(list(SILENT_LOGGERS), "CRITICAL"),
        },
    }
