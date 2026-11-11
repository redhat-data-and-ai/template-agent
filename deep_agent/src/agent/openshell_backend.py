"""An OpenShell-backed code sandbox for the agent's filesystem routes.

The agent builds its backend from agent.yaml alone, which never produces
one that can execute: `FilesystemMiddleware` only offers the `execute` tool
when the backend is a sandbox. This supplies one.

The sandbox itself is `langchain_nvidia_openshell.OpenShellSandbox`, which
already implements the `BaseSandbox` contract against OpenShell -- exec,
chunked upload and chunked download. What is added here is the part
specific to serving many callers from one long-lived agent process:

  * a gateway client built from settings rather than from the OpenShell
    CLI's on-disk state, because an agent in a container has no CLI config
    and no browser to log in with. Note that the SDK keeps channel TLS
    (`SandboxClient(tls=...)`) and OIDC issuer verification
    (`ClientCredentialsAuth(insecure=...)`) separate, and refuses client
    credentials over a plaintext channel to a non-loopback gateway;
  * **one sandbox per caller per runtime**. The compiled graph -- and so
    this backend -- is a process-wide singleton shared by every request,
    so the sandbox cannot be. Sessions are keyed by the calling
    conversation (or user, see `OPENSHELL_SESSION_SCOPE`) taken from the
    LangGraph run context, which `asyncio.to_thread` propagates into the
    worker thread `BaseSandbox`'s sync methods run on. A caller that
    cannot be identified gets no sandbox at all unless
    `OPENSHELL_ALLOW_SHARED_SANDBOX` is set;
  * lifecycle the SDK deliberately leaves to the caller: ready-waits,
    idle reaping, an LRU cap, recovery from a sandbox that died
    underneath us, and deletion on shutdown;
  * the two tools the filesystem middleware cannot derive, since choosing
    a runtime and recreating a sandbox are not filesystem operations.

Configuration lives in `deep_agent.src.settings` under `OPENSHELL_*`;
without `OPENSHELL_ENDPOINT` `wrap_backend` is a no-op and the agent runs
exactly as it did before, minus `execute`.
"""

from __future__ import annotations

import atexit
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from langchain_core.tools import tool
from langchain_nvidia_openshell import OpenShellSandbox

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

RUNTIMES: dict[str, str] = {
    "python": settings.OPENSHELL_IMAGE_PYTHON,
    "node": settings.OPENSHELL_IMAGE_NODE,
    "go": settings.OPENSHELL_IMAGE_GO,
}

# Red Hat Project Hummingbird builder variants, used as published. The
# builder variant matters: the default variants are distroless, and this
# backend is shell-first.
#
# Only hi/python ships python3, and BaseSandbox implements read, ls, grep,
# glob, edit and write as python3 scripts piped through the shell -- so on
# node and go, those file tools fail and only `execute` works. PROMPT.md
# says so, and openshell_use_runtime repeats it, so the model reaches for
# shell commands there instead of failing tool calls.
NO_PYTHON3 = ("node", "go")

# Substrings that mean "this sandbox is gone", as opposed to "your command
# failed", used to decide whether a retry on a freshly created sandbox is
# worth attempting. Deliberately narrow: a false positive silently
# destroys a caller's working state.
#
# Matched against text rather than a gRPC status code because the
# langchain wrapper catches the SDK's exception and renders it into an
# `ExecuteResponse`, so by the time `execute` sees a failure the code is
# gone. `GatewayError` stringifies the underlying `grpc.RpcError`, which
# names its status, hence the "statuscode." entries.
_DEAD_SANDBOX_MARKERS = (
    "statuscode.not_found",
    "statuscode.unavailable",
    "statuscode.failed_precondition",
    "not found",
    "notfound",
    "no such sandbox",
    "sandbox does not exist",
    "unavailable",
    "terminated",
    "has been deleted",
    "connection refused",
    "connection reset",
)

# Error text is logged, and gateway errors can quote the request that
# produced them. Keep credentials out of the log.
_SECRET_PATTERN = re.compile(
    r"(?i)\b(client[_-]?secret|authorization|bearer|access[_-]?token"
    r"|refresh[_-]?token|api[_-]?key|password)\b"
    # Whatever separates the name from its value, an optional opening
    # quote, an optional auth scheme, then the value itself.
    r"[\s:=]*\"?(?:bearer\s+|basic\s+)?[\w.\-+/=]+\"?"
)


def _redact(text: str, limit: int = 300) -> str:
    """Strip credential-looking values from error text and clip it.

    Deliberately greedy about what counts as a credential: over-redacting
    a log line costs nothing, and gateway errors quote the request that
    produced them.
    """
    return _SECRET_PATTERN.sub(r"\1=***", text)[:limit]


class OpenShellNotConfigured(RuntimeError):
    """Raised when the gateway endpoint or its credentials are missing.

    Its own type so the tools can report it as a readable message rather
    than a traceback: an agent deployed without a gateway should still
    answer, just without the ability to run code.
    """


class OpenShellOwnerUnknown(RuntimeError):
    """Raised when the calling conversation cannot be identified.

    Falling back to a process-wide sandbox here would hand every caller
    the same filesystem, so this fails closed instead. See
    `OPENSHELL_ALLOW_SHARED_SANDBOX` for the single-user case.
    """


def is_configured() -> bool:
    """Whether a gateway is configured for this deployment."""
    return bool(
        settings.OPENSHELL_ENDPOINT
        and settings.OPENSHELL_CLIENT_ID
        and settings.OPENSHELL_CLIENT_SECRET
        and settings.OPENSHELL_ISSUER
    )


# ----------------------------------------------------------------------
# Caller identity
# ----------------------------------------------------------------------

_SAFE_KEY = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize(value: str, limit: int = 48) -> str:
    """Reduce an identifier to something safe to use as a Kubernetes label.

    Truncation happens before the final strip, because a value cut at the
    limit can end on a separator and a label value must begin and end
    alphanumeric -- the gateway rejects the whole create otherwise.
    """
    return _SAFE_KEY.sub("-", value)[:limit].strip("-._") or "unknown"


def _config_bags() -> list[dict[str, Any]]:
    """Return the run config's `configurable` and `metadata` dicts, if any.

    Read through `langgraph.config.get_config()`, which is backed by a
    context variable. `asyncio.to_thread` copies the context into the
    worker thread, so this resolves inside `BaseSandbox`'s sync methods as
    well as inside async tool calls.
    """
    try:
        from langgraph.config import get_config

        config = get_config() or {}
    except Exception:  # noqa: BLE001 - outside a runnable context
        return []
    return [
        bag
        for bag in (config.get("configurable"), config.get("metadata"))
        if isinstance(bag, dict)
    ]


def _user_identity() -> str | None:
    """Best-effort identity of the authenticated caller.

    Mirrors the resolution order the Store namespaces already use
    (`_safe_namespace_user`): the BFF user id the UI proxy sets, then the
    authenticated principal from the run config, then the request-scoped
    id the logger binds.
    """
    for bag in _config_bags():
        for key in ("user_id", "x_user_id", "owner"):
            value = bag.get(key)
            if value:
                return str(value)
        user = bag.get("langgraph_auth_user")
        identity = getattr(user, "identity", None) if user is not None else None
        if identity:
            return str(identity)

    try:
        from deep_agent.utils.pylogger import _user_id_var

        bound = _user_id_var.get()
    except Exception:  # noqa: BLE001
        bound = None
    return str(bound) if bound else None


def _owner_key() -> str:
    """Identify whose sandbox the current call should use.

    The returned key is the isolation boundary: two calls that produce
    different keys can never see each other's files. Under the default
    "thread" scope that is one sandbox per conversation; "user" shares a
    sandbox across one user's conversations. The user component is always
    included, so a guessed or reused thread id still cannot reach another
    user's sandbox.
    """
    user = _user_identity()
    thread: str | None = None
    for bag in _config_bags():
        value = bag.get("thread_id") or bag.get("run_id")
        if value:
            thread = str(value)
            break

    if settings.OPENSHELL_SESSION_SCOPE == "user" and user:
        return f"u.{_sanitize(user)}"
    if user and thread:
        return f"u.{_sanitize(user)}.t.{_sanitize(thread)}"
    if thread:
        return f"t.{_sanitize(thread)}"
    if user:
        return f"u.{_sanitize(user)}"

    if settings.OPENSHELL_ALLOW_SHARED_SANDBOX:
        return "shared"
    raise OpenShellOwnerUnknown(
        "the sandbox could not tell which conversation this call belongs to, "
        "and refuses to use one shared by every caller; this is a deployment "
        "problem, not something to retry"
    )


# ----------------------------------------------------------------------
# Gateway client
# ----------------------------------------------------------------------

_client_singleton: Any = None
_client_lock = threading.Lock()


def _tls_config() -> Any:
    """Build the gRPC channel's TLS material, or None for plaintext.

    `TlsConfig()` with no paths means the OS trust store, which is what a
    gateway behind a public or cluster CA needs; paths narrow that to a
    custom CA and, with cert+key, full mTLS.

    Returning None is a plaintext channel. The SDK refuses to carry OAuth
    client credentials over one unless the endpoint is loopback, and
    `validate_config` rejects the same combination at startup, so this
    only happens where it is safe.
    """
    if not settings.OPENSHELL_TLS_ENABLED:
        return None
    import pathlib

    from openshell import TlsConfig

    def _path(value: str | None) -> Any:
        return pathlib.Path(value) if value else None

    return TlsConfig(
        ca_path=_path(settings.OPENSHELL_TLS_CA_PATH),
        cert_path=_path(settings.OPENSHELL_TLS_CERT_PATH),
        key_path=_path(settings.OPENSHELL_TLS_KEY_PATH),
    )


def _client() -> Any:
    """Return the gateway client, built once per process.

    Built from settings rather than with
    `SandboxClient.from_active_cluster()`, which reads the OpenShell CLI's
    on-disk gateway state. A container has no such state and no way to
    complete an interactive login, so a machine identity is used instead.

    Imported lazily so that importing this module -- which graph.py does
    at startup -- cannot fail the agent on a missing SDK.
    """
    global _client_singleton
    if not is_configured():
        raise OpenShellNotConfigured(
            "OPENSHELL_ENDPOINT and a machine identity "
            "(OPENSHELL_CLIENT_ID / _CLIENT_SECRET / _ISSUER) are required "
            "before code can be run in a sandbox"
        )
    # Always locked rather than double-checked: construction is once per
    # process, every call after that is an uncontended acquire, and two
    # concurrent first calls would otherwise each open a gRPC channel and
    # leak the loser's.
    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton
        import openshell

        # `insecure` here is about the OIDC issuer's certificate, not the
        # gateway channel -- the SDK passes it to httpx as `verify`. The
        # channel's own security is `tls=` below, and conflating the two
        # silently downgrades one of them.
        auth = openshell.ClientCredentialsAuth(
            client_secret=settings.OPENSHELL_CLIENT_SECRET,
            client_id=settings.OPENSHELL_CLIENT_ID,
            issuer=settings.OPENSHELL_ISSUER,
            audience=settings.OPENSHELL_AUDIENCE,
            scopes=(
                settings.OPENSHELL_SCOPES.split() if settings.OPENSHELL_SCOPES else None
            ),
            insecure=settings.OPENSHELL_OIDC_INSECURE,
        )
        _client_singleton = openshell.SandboxClient(
            endpoint=settings.OPENSHELL_ENDPOINT,
            tls=_tls_config(),
            client_credentials=auth,
            cluster_name=settings.OPENSHELL_NAME,
        )
        logger.info(
            "openshell_client_ready",
            endpoint=settings.OPENSHELL_ENDPOINT,
            workspace=settings.OPENSHELL_WORKSPACE,
            tls=settings.OPENSHELL_TLS_ENABLED,
        )
        return _client_singleton


def _runtime_spec(runtime: str) -> Any:
    """Build the inline sandbox spec naming this runtime's image.

    The image can be supplied either by a stored workload template or
    inline on the sandbox itself, and this takes the inline route
    deliberately: creating a workload template needs the `admin` role in
    the workspace, while creating a sandbox needs only `user`. Templates
    are also shared, mutable, workspace-level state -- two agents on one
    workspace would fight over the same name -- whereas an inline spec is
    scoped to the sandbox that uses it.
    """
    from openshell._proto import openshell_pb2

    spec = openshell_pb2.SandboxSpec()
    spec.template.image = RUNTIMES[runtime]
    return spec


def _grpc_status(exc: BaseException) -> Any:
    """Return an exception's gRPC status code, if it carries one.

    The SDK wraps RPC failures in `GatewayError`, which keeps `code()`
    from the underlying error. Reading the code beats matching on message
    text, which is server-worded and free to change.
    """
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        return code()
    except Exception:  # noqa: BLE001 - not an RPC error after all
        return None


# ----------------------------------------------------------------------
# Session registry
# ----------------------------------------------------------------------


@dataclass
class _Entry:
    """One sandbox, plus the bookkeeping needed to retire it."""

    owner: str
    runtime: str
    handle: Any = None
    sandbox: OpenShellSandbox | None = None
    last_used: float = field(default_factory=time.monotonic)
    # Per-entry, so that one caller's three-minute image pull does not
    # stall every other caller's commands.
    lock: threading.Lock = field(default_factory=threading.Lock)


_entries: dict[tuple[str, str], _Entry] = {}
# Guards `_entries` and `_runtime_choice` only -- never held across a
# gateway call.
_registry_lock = threading.RLock()
_runtime_choice: dict[str, str] = {}
_reaper: threading.Thread | None = None
_reaper_stop = threading.Event()


def _active_runtime(owner: str) -> str:
    """The runtime this owner last selected, or the configured default."""
    with _registry_lock:
        return _runtime_choice.get(owner, settings.OPENSHELL_RUNTIME)


def _set_active_runtime(owner: str, runtime: str) -> None:
    with _registry_lock:
        _runtime_choice[owner] = runtime


def _delete_remote(entry: _Entry) -> None:
    """Delete an entry's sandbox at the gateway, logging any failure.

    Never raises: every caller is already on a path where the sandbox has
    been dropped locally, and a leaked pod is the reaper's problem, not
    something to fail a user's turn over.
    """
    handle = entry.handle
    if handle is None:
        return
    name = getattr(getattr(handle, "sandbox", None), "name", None)
    if not name:
        return
    try:
        _client().delete(
            name,
            workspace=settings.OPENSHELL_WORKSPACE,
            allow_missing=True,
            # The SDK's 30s default is shorter than a real pod teardown, so
            # the call raised DEADLINE_EXCEEDED while the delete went on to
            # succeed -- a warning for a sandbox that was in fact removed.
            timeout_seconds=settings.OPENSHELL_DELETE_TIMEOUT,
        )
        logger.info(
            "openshell_session_deleted",
            owner=entry.owner,
            runtime=entry.runtime,
            sandbox=name,
        )
    except Exception as exc:  # noqa: BLE001 - best effort teardown
        logger.warning(
            "openshell_session_delete_failed",
            owner=entry.owner,
            runtime=entry.runtime,
            sandbox=name,
            error=_redact(f"{type(exc).__name__}: {exc}"),
        )


def _evict_over_cap() -> list[_Entry]:
    """Drop least-recently-used entries until the cap is satisfied.

    A cap is what stops a busy day from filling the namespace with pods.
    Eviction only removes the entry from the registry; the caller deletes
    the sandboxes outside the lock.
    """
    doomed: list[_Entry] = []
    with _registry_lock:
        while len(_entries) > settings.OPENSHELL_MAX_SESSIONS:
            key = min(_entries, key=lambda k: _entries[k].last_used)
            doomed.append(_entries.pop(key))
    for entry in doomed:
        logger.info(
            "openshell_session_evicted",
            owner=entry.owner,
            runtime=entry.runtime,
            reason="max_sessions",
        )
    return doomed


def _reap_idle() -> list[_Entry]:
    """Remove entries unused for longer than the idle TTL."""
    cutoff = time.monotonic() - settings.OPENSHELL_SESSION_IDLE_TTL
    doomed: list[_Entry] = []
    with _registry_lock:
        for key in [k for k, e in _entries.items() if e.last_used < cutoff]:
            doomed.append(_entries.pop(key))
        live_owners = {owner for owner, _ in _entries}
        for owner in [o for o in _runtime_choice if o not in live_owners]:
            del _runtime_choice[owner]
    for entry in doomed:
        logger.info(
            "openshell_session_reaped",
            owner=entry.owner,
            runtime=entry.runtime,
            idle_ttl=settings.OPENSHELL_SESSION_IDLE_TTL,
        )
    return doomed


def _reaper_loop() -> None:
    """Delete idle sandboxes until the process stops.

    A sandbox is a pod with a cost; without this, every conversation the
    agent ever served keeps one alive forever.
    """
    while not _reaper_stop.wait(settings.OPENSHELL_REAPER_INTERVAL):
        try:
            for entry in _reap_idle():
                _delete_remote(entry)
        except Exception as exc:  # noqa: BLE001 - the reaper must not die
            logger.warning(
                "openshell_reaper_error",
                error=_redact(f"{type(exc).__name__}: {exc}"),
            )


def _ensure_reaper() -> None:
    """Start the idle reaper on first use."""
    global _reaper
    if _reaper is not None and _reaper.is_alive():
        return
    with _registry_lock:
        if _reaper is not None and _reaper.is_alive():
            return
        _reaper_stop.clear()
        _reaper = threading.Thread(
            target=_reaper_loop, name="openshell-reaper", daemon=True
        )
        _reaper.start()


def _create_sandbox(entry: _Entry) -> OpenShellSandbox:
    """Create this entry's sandbox and wait for it to accept commands."""
    client = _client()
    logger.info(
        "openshell_session_create",
        owner=entry.owner,
        runtime=entry.runtime,
        image=RUNTIMES[entry.runtime],
    )
    handle = client.create_session(
        workspace=settings.OPENSHELL_WORKSPACE,
        spec=_runtime_spec(entry.runtime),
        labels={
            "app": settings.OPENSHELL_NAME,
            "runtime": entry.runtime,
            # Enough to find orphans left by a hard process kill.
            "owner": _sanitize(entry.owner, limit=63),
        },
    )
    try:
        # The first command in a runtime waits for an image pull, not just
        # a scheduling decision: these are multi-hundred-MiB builder images.
        client.wait_ready(
            handle.sandbox.name,
            workspace=settings.OPENSHELL_WORKSPACE,
            timeout_seconds=settings.OPENSHELL_READY_TIMEOUT,
        )
    except Exception:
        # A sandbox that never became ready is still a pod. Delete it
        # rather than let the ready-timeout leak one per attempt.
        entry.handle = handle
        _delete_remote(entry)
        entry.handle = None
        raise
    entry.handle = handle
    entry.sandbox = OpenShellSandbox(
        sandbox=handle, timeout=settings.OPENSHELL_EXEC_TIMEOUT
    )
    logger.info(
        "openshell_session_ready",
        owner=entry.owner,
        runtime=entry.runtime,
        sandbox=handle.sandbox.name,
    )
    return entry.sandbox


def _acquire(owner: str, runtime: str) -> OpenShellSandbox:
    """Return this owner's sandbox for `runtime`, creating it on first use."""
    _ensure_reaper()
    key = (owner, runtime)
    with _registry_lock:
        entry = _entries.get(key)
        if entry is None:
            entry = _Entry(owner=owner, runtime=runtime)
            _entries[key] = entry
        entry.last_used = time.monotonic()

    for doomed in _evict_over_cap():
        _delete_remote(doomed)

    # Taken outside the registry lock: creation can block for the whole
    # ready timeout, and only concurrent calls for this same key need to
    # wait on it.
    with entry.lock:
        if entry.sandbox is None:
            try:
                _create_sandbox(entry)
            except Exception:
                with _registry_lock:
                    if _entries.get(key) is entry:
                        del _entries[key]
                raise
        return entry.sandbox


def _discard(owner: str, runtime: str) -> _Entry | None:
    """Drop an owner's sandbox from the registry and return it for deletion."""
    with _registry_lock:
        return _entries.pop((owner, runtime), None)


def _looks_dead(response: ExecuteResponse) -> bool:
    """Whether a failed response means the sandbox itself is gone.

    The SDK turns transport errors into an `ExecuteResponse` rather than
    raising, so a vanished pod is indistinguishable from a failing command
    except by its text.
    """
    if response.exit_code == 0:
        return False
    text = (response.output or "").lower()
    if not text.startswith("error:"):
        return False
    return any(marker in text for marker in _DEAD_SANDBOX_MARKERS)


def shutdown_all() -> None:
    """Delete every sandbox this process created.

    Registered with `atexit` so an ordinary restart does not strand a pod
    per conversation. A hard kill still can, which is what the `owner` and
    `app` labels are for.
    """
    global _client_singleton
    _reaper_stop.set()
    with _registry_lock:
        doomed = list(_entries.values())
        _entries.clear()
        _runtime_choice.clear()
    for entry in doomed:
        _delete_remote(entry)

    # Closed last: the deletes above need the channel. `SandboxClient`
    # owns a gRPC channel and, with client credentials, an httpx client
    # for token exchange; neither is released by garbage collection
    # alone.
    with _client_lock:
        client, _client_singleton = _client_singleton, None
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - teardown is best effort
            logger.warning(
                "openshell_client_close_failed",
                error=_redact(f"{type(exc).__name__}: {exc}"),
            )


def _trace_metadata(**extra: Any) -> None:
    """Tag the enclosing tool span with where the command actually ran.

    An `execute` span says a command ran; it does not say the sandbox was
    OpenShell rather than Daytona or this cluster, which runtime served
    it, or which sandbox it landed in. On a deployment where more than one
    sandbox backend is in play, that is the first thing anybody reading a
    trace wants to know.

    This runs inside the LangChain tool span -- `aexecute` reaches the
    sync path through `asyncio.to_thread`, which copies the context, so
    the span the callback handler opened is still current. Best effort:
    observability must never be the reason a command fails, and outside a
    traced run there is no span to update at all.
    """
    # Resolved separately: a call with no owner is refused, and that is
    # the one most worth labelling -- so losing the key must not cost the
    # whole tag.
    try:
        owner: str | None = _owner_key()
    except Exception:  # noqa: BLE001 - unattributable call; tag it anyway
        owner = None
    runtime = _active_runtime(owner) if owner else None
    entry = _entries.get((owner, runtime)) if owner and runtime else None
    handle = getattr(entry, "handle", None)

    try:
        from langfuse import get_client

        get_client().update_current_span(
            metadata={
                "sandbox.provider": "openshell",
                "sandbox.runtime": runtime,
                "sandbox.image": RUNTIMES.get(runtime) if runtime else None,
                "sandbox.session": owner,
                "sandbox.id": getattr(getattr(handle, "sandbox", None), "name", None),
                "sandbox.workspace": settings.OPENSHELL_WORKSPACE,
                "sandbox.gateway": settings.OPENSHELL_ENDPOINT,
                **extra,
            }
        )
    except Exception:  # noqa: BLE001 - never fail a run for a trace tag
        pass


atexit.register(shutdown_all)


# ----------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------


class OpenShellRuntimeSandbox(OpenShellSandbox):
    """A `BaseSandbox` that resolves a per-caller sandbox on every call.

    `OpenShellSandbox` wraps one session for its lifetime, which is the
    right contract for the SDK and the wrong one here: this backend is
    built once when the graph is assembled and is then shared by every
    request the process serves, across any number of runtime switches.

    So the inherited `execute`, `upload_files` and `download_files` are
    overridden to look up the calling conversation's own
    `OpenShellSandbox` and delegate to it. Everything else `BaseSandbox`
    derives -- read, write, edit, ls, glob, grep -- is built on those
    three and needs no change.
    """

    def __init__(self) -> None:
        """Build a backend with no session of its own."""
        # `sandbox` is required by the base constructor and is never read:
        # every method that would touch `self._sandbox` is overridden
        # below to resolve the caller's sandbox instead.
        super().__init__(sandbox=None, timeout=settings.OPENSHELL_EXEC_TIMEOUT)

    # -- per-call resolution ------------------------------------------

    def _current(self) -> tuple[str, str, OpenShellSandbox]:
        """Return (owner, runtime, sandbox) for the in-flight call."""
        owner = _owner_key()
        runtime = _active_runtime(owner)
        return owner, runtime, _acquire(owner, runtime)

    @property
    def id(self) -> str:
        """Return the calling conversation's sandbox id.

        Informational only, and reachable from diagnostics outside a run,
        so it degrades to a placeholder rather than raising or -- worse --
        creating a sandbox nobody asked for.
        """
        try:
            owner = _owner_key()
            runtime = _active_runtime(owner)
        except OpenShellOwnerUnknown:
            return "openshell:unidentified"
        with _registry_lock:
            entry = _entries.get((owner, runtime))
        if entry is None or entry.handle is None:
            return f"openshell:{runtime}:not-started"
        return str(getattr(entry.handle.sandbox, "name", "unknown"))

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Run `command` in the calling conversation's sandbox.

        Retried once against a fresh sandbox when the first attempt fails
        because the sandbox is gone -- pods get evicted, OOM-killed and
        drained, and without this the entry stays cached and every later
        command in that conversation fails forever.
        """
        try:
            owner, runtime, sandbox = self._current()
        except (OpenShellNotConfigured, OpenShellOwnerUnknown) as exc:
            # Tagged too, and it is the tag worth having: a refused call
            # leaves no sandbox behind to explain itself afterwards.
            _trace_metadata(status=type(exc).__name__)
            # Non-zero so the base class does not read the message as
            # command output.
            return ExecuteResponse(
                output=f"error: {exc}", exit_code=127, truncated=False
            )

        response = sandbox.execute(command, timeout=timeout)
        if not _looks_dead(response):
            _trace_metadata(status="ok", exit_code=response.exit_code)
            return response

        logger.warning(
            "openshell_session_lost",
            owner=owner,
            runtime=runtime,
            detail=_redact(response.output or ""),
        )
        entry = _discard(owner, runtime)
        if entry is not None:
            _delete_remote(entry)
        try:
            retry_sandbox = _acquire(owner, runtime)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            _trace_metadata(status="unavailable")
            return ExecuteResponse(
                output=f"error: sandbox is unavailable: {_redact(str(exc))}",
                exit_code=1,
                truncated=False,
            )
        retried = retry_sandbox.execute(command, timeout=timeout)
        # `recreated` distinguishes a span whose sandbox id is not the one
        # the command started on -- otherwise the trace silently attributes
        # output to a pod that never ran it.
        _trace_metadata(status="ok", exit_code=retried.exit_code, recreated=True)
        return retried

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload into the calling conversation's sandbox."""
        try:
            _, _, sandbox = self._current()
        except (OpenShellNotConfigured, OpenShellOwnerUnknown) as exc:
            logger.warning("openshell_upload_unavailable", error=str(exc))
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _ in files
            ]
        uploaded: list[FileUploadResponse] = sandbox.upload_files(files)
        return uploaded

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download from the calling conversation's sandbox."""
        try:
            _, _, sandbox = self._current()
        except (OpenShellNotConfigured, OpenShellOwnerUnknown) as exc:
            logger.warning("openshell_download_unavailable", error=str(exc))
            return [
                FileDownloadResponse(path=path, content=None, error="permission_denied")
                for path in paths
            ]
        downloaded: list[FileDownloadResponse] = sandbox.download_files(paths)
        return downloaded

    # -- lifecycle -----------------------------------------------------

    def reset(self, runtime: str | None = None) -> str:
        """Destroy the caller's own sandbox; the next command makes a new one.

        Scoped to the caller: a reset must never reach into another
        conversation's filesystem.
        """
        owner = _owner_key()
        name = runtime or _active_runtime(owner)
        entry = _discard(owner, name)
        if entry is None:
            return f"no {name} sandbox was running; the next command will start one"
        _delete_remote(entry)
        return f"{name} sandbox destroyed; the next command will start a fresh one"


_sandbox_singleton: OpenShellRuntimeSandbox | None = None


def get_sandbox() -> OpenShellRuntimeSandbox:
    """Return the process-wide backend that dispatches to per-caller sandboxes."""
    global _sandbox_singleton
    if _sandbox_singleton is None:
        _sandbox_singleton = OpenShellRuntimeSandbox()
    return _sandbox_singleton


def wrap_backend(backend: BackendProtocol) -> BackendProtocol:
    """Put the sandbox underneath the agent's configured routes.

    The agent's own `get_configured_backend()` builds a `CompositeBackend`
    from agent.yaml: /skills/ and /data/ read-only, /memories/ and
    /reports/ in the LangGraph store, everything else in checkpointed
    state. Only that last part changes here -- the sandbox takes over as
    the default, which is both where loose paths go and what the
    middleware inspects to decide whether to offer the `execute` tool.

    The displaced backend is kept, routed at `SANDBOX_ARTIFACTS_ROOT`, so
    the middleware's offloaded tool results and evicted messages stay in
    checkpointed state rather than in a sandbox that `reset()` can delete.

    With no gateway configured this returns `backend` untouched. Installing
    a sandbox that cannot run would otherwise take the agent's ordinary
    checkpointed filesystem down with it, since the sandbox is the default
    route for every loose path.
    """
    if not is_configured():
        logger.info("openshell_disabled", reason="OPENSHELL_ENDPOINT is not set")
        return backend

    artifacts_root = settings.SANDBOX_ARTIFACTS_ROOT
    sandbox = get_sandbox()
    if isinstance(backend, CompositeBackend):
        routes = dict(backend.routes)
        displaced = backend.default
    else:
        routes, displaced = {}, backend

    if artifacts_root in routes:
        # agent.yaml already claims this prefix, so the state backend has
        # nowhere to go and offloaded messages would land in the sandbox.
        raise ValueError(
            f"SANDBOX_ARTIFACTS_ROOT {artifacts_root!r} is already routed by "
            "agent.yaml; point it at an unused prefix so offloaded messages "
            "survive a sandbox reset"
        )
    routes[artifacts_root] = displaced
    return CompositeBackend(
        default=sandbox, routes=routes, artifacts_root=artifacts_root
    )


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------


def _tool_error(exc: Exception) -> str:
    """Render an exception as a message the model can act on."""
    if isinstance(exc, OpenShellNotConfigured | OpenShellOwnerUnknown):
        return f"error: {exc}"
    return f"error: {type(exc).__name__}: {_redact(str(exc))}"


@tool
def openshell_use_runtime(name: str) -> str:
    """Choose which language runtime the sandbox runs commands in.

    Each runtime is a separate sandbox with its own filesystem, so files
    written under one are NOT visible from another. The first command
    after switching waits for the image to be pulled; later ones do not.
    The choice applies to this conversation only.

    Args:
        name: One of "python", "node", "go".
    """
    name = (name or "").strip().lower()
    if name not in RUNTIMES:
        return f"error: unknown runtime {name!r}; choose one of {sorted(RUNTIMES)}"
    try:
        owner = _owner_key()
    except OpenShellOwnerUnknown as exc:
        return _tool_error(exc)
    _set_active_runtime(owner, name)
    logger.info("openshell_runtime_selected", owner=owner, runtime=name)
    note = (
        " This image has no python3, so read_file, write_file, edit_file, ls,"
        " glob and grep will NOT work here -- use `execute` with shell"
        " commands instead."
        if name in NO_PYTHON3
        else ""
    )
    return (
        f"runtime is now {name} ({RUNTIMES[name]}); its filesystem is "
        f"separate from the others.{note}"
    )


@tool
def openshell_reset_sandbox() -> str:
    """Destroy this conversation's sandbox for the current runtime.

    Use when previous state is getting in the way, or to discard anything
    an earlier command left behind. Files, installed packages and the
    working directory all go. Other runtimes, and other conversations,
    are untouched.
    """
    try:
        return get_sandbox().reset()
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the model
        return _tool_error(exc)


def get_builtin_tools() -> list:
    """Return the two tools the filesystem middleware does not provide.

    Everything else the model needs -- execute, read_file, write_file,
    edit_file, ls, glob, grep -- `FilesystemMiddleware` derives from the
    backend itself. Choosing a runtime and recreating the sandbox are not
    filesystem operations, so they have no equivalent there.

    Returned even when no gateway is configured, because PROMPT.md
    declares them and `enforce_capability` would otherwise drop a declared
    tool; they report that they are unconfigured when called.
    """
    return [openshell_use_runtime, openshell_reset_sandbox]
