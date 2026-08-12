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
from collections import OrderedDict
from typing import Any

from deep_agent.aegra.auth import encrypt_user_id
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# Langfuse v4 integration
# ---------------------------------------------------------------------------

_langfuse_tracing_initialized = False
_token_budget_tracing_initialized = False
_guardian_initialized = False
_pii_initialized = False

# Cap abandoned HITL open roots (never resumed / never replaced by a new message).
# Mirrors Langfuse's MAX_PENDING_RESUME_TRACE_CONTEXTS bound.
_MAX_OPEN_HITL_ROOTS = 1024

# Process-wide Langfuse handler so HITL interrupt->resume keeps the same trace.
# Langfuse stores pending resume context on the handler instance; a fresh
# CallbackHandler() per run (register_configure_hook default) drops that state.
#
# NOTE: thread-safety limitation -- _open_hitl_roots, _keep_open_root_run_ids,
# and _runs are plain dicts/sets with no locking.  Concurrent LangChain runs
# sharing this handler may race on those structures.  A proper fix would use
# per-thread or per-asyncio-task state, but that is a larger refactor.
_shared_langfuse_handler: Any | None = None
HitlAwareCallbackHandler: Any | None = None


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


def setup_pii_middleware() -> None:
    """Initialise the PII scrubber from agent.yaml (custom_pii section).

    Must be called before setup_langfuse_tracing() so that the global
    scrubber is available when the Langfuse handler activates.
    No-op when pii.enabled is false in agent.yaml or already initialised.
    """
    global _pii_initialized  # noqa: PLW0603
    if _pii_initialized:
        return
    _pii_initialized = True

    try:
        from deep_agent.src.agent.config import agent_config
        from deep_agent.src.pii import init_pii_middleware
        from deep_agent.src.pii.config import ActionType, PIIConfig, PIIRule
        from deep_agent.src.settings import settings

        pii_cfg = agent_config.get_custom_pii_config()
        if not pii_cfg.enabled or not pii_cfg.rules:
            logger.info("PII middleware: no rules defined in agent.yaml pii section")
            return

        # Only non-default rules go to the token-map scrubber;
        # provider: default rules are handled by the stock PIIMiddleware.
        rules = [
            PIIRule(
                name=r.name,
                regex=r.regex,
                strategy=ActionType(r.strategy),
                provider=r.provider,
                label=r.label,
            )
            for r in pii_cfg.rules
            if r.provider != "default"
        ]
        config = PIIConfig(
            enabled=True,
            trace_strategy=pii_cfg.trace_strategy,
            rules=rules,
        )
        hash_key = settings.PII_HASH_KEY.encode() if settings.PII_HASH_KEY else b""
        init_pii_middleware(config, hash_key)
        logger.info("PII middleware initialised (%d rules from agent.yaml)", len(rules))
    except Exception:
        logger.warning("Failed to initialise PII middleware", exc_info=True)


def _build_hitl_aware_handler_class() -> type:
    """Build CallbackHandler that keeps one orchestrator span across HITL resume.

    LangGraph soft-interrupts end the root chain run, so upstream Langfuse opens
    a new ``orchestrator`` root on every approve. We instead:

    1. On nested ``GraphInterrupt``, remember the root observation by thread_id
       and mark that root run to stay open.
    2. On root ``on_chain_end`` for an interrupted run, update output but do
       **not** call ``span.end()``.
    3. On ``Command(resume=...)`` root ``on_chain_start``, rebind the new
       LangChain run_id to the open observation (no new root span). Children
       then attach under the original orchestrator.
    """
    from uuid import UUID

    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    try:
        from langgraph.errors import GraphInterrupt as _GraphInterrupt
    except ImportError:  # pragma: no cover - langgraph always present in runtime
        _GraphInterrupt = None

    class _HitlAwareCallbackHandler(CallbackHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # thread_id → still-open root observation across HITL pauses (LRU)
            self._open_hitl_roots: OrderedDict[str, Any] = OrderedDict()
            # root run_ids that must skip span.end() on on_chain_end
            self._keep_open_root_run_ids: set[UUID] = set()

        def _store_open_hitl_root(self, resume_key: str, root_obs: Any) -> None:
            """Remember an open root, evicting the oldest if over the cap."""
            previous = self._open_hitl_roots.get(resume_key)
            if previous is not None and previous is not root_obs:
                try:
                    previous.end()
                except Exception:
                    logger.warning(
                        "Failed to end replaced HITL open root", exc_info=True
                    )
            if resume_key in self._open_hitl_roots:
                self._open_hitl_roots.move_to_end(resume_key)
            self._open_hitl_roots[resume_key] = root_obs
            while len(self._open_hitl_roots) > _MAX_OPEN_HITL_ROOTS:
                _, evicted = self._open_hitl_roots.popitem(last=False)
                try:
                    evicted.end()
                except Exception:
                    logger.warning(
                        "Failed to end evicted HITL open root", exc_info=True
                    )

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            root_run_id: UUID | None = None
            root_obs: Any | None = None
            resume_key: str | None = None
            # Capture before super() — root hard errors clear resume_key upstream.
            stale_resume_key: str | None = None
            if parent_run_id is None:
                root_state = self._get_root_run_state(run_id)
                if root_state is not None:
                    stale_resume_key = root_state.resume_key

            # Only HITL GraphInterrupt — not ParentCommand / other GraphBubbleUp.
            if (
                parent_run_id is not None
                and _GraphInterrupt is not None
                and isinstance(error, _GraphInterrupt)
            ):
                state = self._get_run_state(run_id)
                if state is not None:
                    root_run_id = state.root_run_id
                    root_obs = self._runs.get(root_run_id)
                    root_state = self._root_run_states.get(root_run_id)
                    if root_state is not None:
                        resume_key = root_state.resume_key

            super().on_chain_error(
                error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

            if (
                root_run_id is not None
                and root_obs is not None
                and resume_key is not None
            ):
                self._store_open_hitl_root(resume_key, root_obs)
                self._keep_open_root_run_ids.add(root_run_id)
            elif stale_resume_key is not None:
                # Root hard failure: never leave a dead obs for the next resume.
                self._open_hitl_roots.pop(stale_resume_key, None)

        def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            # Interrupted root: keep the Langfuse observation open for resume.
            if parent_run_id is None and run_id in self._keep_open_root_run_ids:
                try:
                    span = self._detach_observation(run_id)
                    if span is not None:
                        span.update(
                            output=outputs,
                            input=kwargs.get("inputs"),
                        )
                        self._deregister_langfuse_prompt(run_id)
                    self._clear_root_run_resume_key(run_id)
                    self._exit_propagation_context(run_id)
                except Exception:
                    logger.warning("HITL open-root on_chain_end failed", exc_info=True)
                finally:
                    self._keep_open_root_run_ids.discard(run_id)
                    if parent_run_id is None:
                        try:
                            self._reset(run_id)
                        except Exception:
                            logger.warning(
                                "HITL open-root _reset failed", exc_info=True
                            )
                return None

            # Final (or non-HITL) root end — drop open-root bookkeeping first.
            if parent_run_id is None:
                root_state = self._get_root_run_state(run_id)
                if root_state is not None and root_state.resume_key is not None:
                    self._open_hitl_roots.pop(root_state.resume_key, None)
                self._keep_open_root_run_ids.discard(run_id)

            return super().on_chain_end(
                outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            # New user message while a HITL root is still open → close it.
            if parent_run_id is None and not self._is_langgraph_resume(inputs):
                stale_key = self._get_langgraph_resume_key(metadata)
                if stale_key is not None:
                    stale = self._open_hitl_roots.pop(stale_key, None)
                    if stale is not None:
                        try:
                            stale.end()
                        except Exception:
                            logger.warning(
                                "Failed to end stale HITL open root", exc_info=True
                            )

            # Resume: reuse the open orchestrator observation (no new root span).
            if parent_run_id is None and self._is_langgraph_resume(inputs):
                resume_key = self._get_langgraph_resume_key(metadata)
                open_obs = (
                    self._open_hitl_roots.get(resume_key)
                    if resume_key is not None
                    else None
                )
                if open_obs is not None:
                    self._track_run(
                        run_id=run_id, parent_run_id=None, metadata=metadata
                    )
                    try:
                        parsed = self._parse_langfuse_trace_attributes(
                            metadata=metadata, tags=tags
                        )
                        propagation_context_manager = propagate_attributes(
                            user_id=parsed.get("user_id", None),
                            session_id=parsed.get("session_id", None),
                            tags=parsed.get("tags", None),
                            metadata=parsed.get("metadata", None),
                            trace_name=parsed.get("trace_name", None),
                        )
                        root_run_state = self._get_root_run_state(run_id)
                        if root_run_state is not None:
                            root_run_state.propagation_context_manager = (
                                propagation_context_manager
                            )
                        propagation_context_manager.__enter__()
                        self._attach_observation(run_id, open_obs)
                        trace_id = getattr(open_obs, "trace_id", None)
                        if isinstance(trace_id, str):
                            self.last_trace_id = trace_id
                        # Ownership moves to _runs for this invoke; map stays empty
                        # until a later nested GraphInterrupt re-stores.
                        if resume_key is not None:
                            self._open_hitl_roots.pop(resume_key, None)
                        return None
                    except Exception:
                        self._exit_propagation_context(run_id)
                        self._reset(run_id)
                        if resume_key is not None:
                            failed = self._open_hitl_roots.pop(resume_key, None)
                            if failed is not None:
                                try:
                                    failed.end()
                                except Exception:
                                    logger.warning(
                                        "Failed to end open root after rebind failure",
                                        exc_info=True,
                                    )
                        logger.warning(
                            "HITL open-root resume rebind failed; "
                            "falling back to new root span",
                            exc_info=True,
                        )
                        # Fall through to super().on_chain_start.

            return super().on_chain_start(
                serialized,
                inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )

    _HitlAwareCallbackHandler.__name__ = "HitlAwareCallbackHandler"
    _HitlAwareCallbackHandler.__qualname__ = "HitlAwareCallbackHandler"
    return _HitlAwareCallbackHandler


def setup_langfuse_tracing() -> None:
    """Register Langfuse as a global LangChain callback and Aegra observability provider.

    Two mechanisms work together:

    1. ``register_configure_hook`` — injects a **process-wide** HITL-aware
       ``CallbackHandler`` so interrupt→resume keeps the same Langfuse trace.
    2. ``LangfuseObservabilityProvider`` — plugs into Aegra's
       ``ObservabilityManager`` so that ``create_run_config`` injects
       ``langfuse_user_id``, ``langfuse_session_id``, ``thread_id``, and
       ``langfuse_trace_name`` into ``RunnableConfig.metadata``.
       The CallbackHandler reads these automatically.

    Must be called **once** at process startup. Subsequent calls are no-ops.
    """
    global _langfuse_tracing_initialized
    global _shared_langfuse_handler
    global HitlAwareCallbackHandler
    if _langfuse_tracing_initialized:
        return
    _langfuse_tracing_initialized = True

    if not _langfuse_configured():
        logger.info("Langfuse credentials not set — auto-tracing disabled")
        return

    try:
        from langchain_core.tracers.context import register_configure_hook

        from deep_agent.src.pii import get_scrubber as _get_scrubber

        if _get_scrubber() is not None:
            try:
                from langfuse import Langfuse
                from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

                def _mask_otel_spans(*, params: Any) -> Any:
                    s = _get_scrubber()
                    if s is None:
                        return None
                    use_hash = getattr(s._config, "trace_strategy", "redact") == "hash"
                    scrub_fn = s.scrub_for_trace_hash if use_hash else s.scrub_one_way
                    patches: dict = {}
                    for identifier, span in params.spans.items():
                        replacements: dict = {}
                        for key, value in span.attributes.items():
                            if isinstance(value, str):
                                scrubbed = scrub_fn(value)
                                if scrubbed != value:
                                    replacements[key] = scrubbed
                        if replacements:
                            patches[identifier] = OtelSpanPatch(
                                set_attributes=replacements
                            )
                    return MaskOtelSpansResult(span_patches=patches)

                Langfuse(mask_otel_spans=_mask_otel_spans)
                logger.info(
                    "Langfuse: PII mask_otel_spans registered — all span attributes scrubbed before export"
                )
            except Exception:
                logger.warning(
                    "Failed to register Langfuse mask_otel_spans", exc_info=True
                )

        HitlAwareCallbackHandler = _build_hitl_aware_handler_class()
        _shared_langfuse_handler = HitlAwareCallbackHandler()

        # ContextVar default makes every request context see the same handler
        # without needing env-var-gated fresh construction.
        _langfuse_ctx_var: contextvars.ContextVar = contextvars.ContextVar(
            "langfuse_handler", default=_shared_langfuse_handler
        )

        register_configure_hook(_langfuse_ctx_var, True)
        logger.info(
            "Langfuse auto-tracing registered (shared HITL-aware handler for interrupt/resume)"
        )
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
    - ``thread_id`` — Langfuse resume-key for HITL interrupt→resume stitching
    - ``langfuse_trace_name`` — human-readable trace name in the UI
    """

    def get_callbacks(self) -> list[Any]:
        """Return empty list — callbacks are handled by register_configure_hook."""
        return []

    def get_metadata(
        self, run_id: str, thread_id: str, user_identity: str | None = None
    ) -> dict[str, Any]:
        """Return Langfuse metadata keys for RunnableConfig injection."""
        from deep_agent.utils.pylogger import _trace_id_var

        metadata: dict[str, Any] = {
            "langfuse_trace_name": _get_trace_name(),
        }
        if user_identity:
            metadata["langfuse_user_id"] = encrypt_user_id(user_identity)
        if thread_id:
            metadata["langfuse_session_id"] = thread_id
            # Langfuse keys pending resume context by metadata["thread_id"].
            metadata["thread_id"] = thread_id
        trace_id = _trace_id_var.get()
        if trace_id:
            metadata["langfuse_tags"] = [f"trace_id:{trace_id}"]
        return metadata

    def is_enabled(self) -> bool:
        """Return True if Langfuse credentials are configured."""
        return _langfuse_configured()


# ---------------------------------------------------------------------------
# Token budget callback integration
# ---------------------------------------------------------------------------


class TokenBudgetObservabilityProvider:
    """Inject thread_id and trace_id into RunnableConfig metadata for the token budget callback."""

    def get_callbacks(self) -> list[Any]:
        """Return empty list — callbacks are handled by register_configure_hook."""
        return []

    def get_metadata(
        self, run_id: str, thread_id: str, user_identity: str | None = None
    ) -> dict[str, Any]:
        """Return token-budget metadata keys for RunnableConfig injection."""
        from deep_agent.src.token_budget.callback import (
            THREAD_ID_METADATA_KEY,
            TRACE_ID_METADATA_KEY,
            USER_ID_METADATA_KEY,
        )
        from deep_agent.utils.pylogger import _trace_id_var

        metadata: dict[str, Any] = {}
        if thread_id:
            metadata[THREAD_ID_METADATA_KEY] = thread_id
        if user_identity:
            metadata[USER_ID_METADATA_KEY] = user_identity
        trace_id = _trace_id_var.get()
        if trace_id:
            metadata[TRACE_ID_METADATA_KEY] = trace_id
        return metadata

    def is_enabled(self) -> bool:
        """Return True if token budget tracking is active."""
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


def setup_guardian_guardrails() -> None:
    """Register Granite Guardian LangChain callback for input/output safety checks."""
    global _guardian_initialized  # noqa: PLW0603
    if _guardian_initialized:
        return
    _guardian_initialized = True

    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.guardrails import init_guardrails
    from deep_agent.src.settings import settings

    guardian_cfg = agent_config.get_guardrails_config()
    if not guardian_cfg.enabled:
        logger.info(
            "Granite Guardian disabled in agent.yaml — skipping callback registration"
        )
        return

    if not settings.GUARDIAN_API_BASE:
        logger.info("Granite Guardian disabled — set GUARDIAN_API_BASE to enable")
        return

    init_guardrails(guardian_cfg)

    try:
        from langchain_core.tracers.context import register_configure_hook

        from deep_agent.src.guardrails.callback import GraniteGuardianCallbackHandler

        _guardian_ctx_var: contextvars.ContextVar = contextvars.ContextVar(
            "guardian_handler", default=None
        )
        os.environ.setdefault("GUARDIAN_ACTIVE", "true")
        register_configure_hook(
            _guardian_ctx_var,
            True,
            GraniteGuardianCallbackHandler,
            env_var="GUARDIAN_ACTIVE",
        )
        logger.info(
            "Granite Guardian callback registered (model=%s)",
            guardian_cfg.model,
        )
    except ImportError:
        logger.warning("langchain_core not available — Guardian callback disabled")
    except Exception:
        logger.warning("Failed to register Guardian callback", exc_info=True)


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
