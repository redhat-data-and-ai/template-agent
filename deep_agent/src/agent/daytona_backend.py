"""A Daytona-backed code sandbox for the agent's filesystem routes.

The agent builds its backend from agent.yaml alone, which never produces
one that can execute: `FilesystemMiddleware` only offers the `execute` tool
when the backend is a sandbox. This supplies one.

The sandbox itself is `langchain_daytona.DaytonaSandbox`, which already
implements the `BaseSandbox` contract against Daytona -- session-based
exec with a polling budget, and batched file upload and download. What is
added here is the part specific to running several language runtimes
behind one agent:

  * a client built from the environment, and treated as optional: the
    provider is off-cluster and can legitimately be unconfigured;
  * one long-lived sandbox per conversation per runtime, created on first
    use and reused across calls, so installed packages and written files
    persist for that conversation;
  * the two tools the filesystem middleware cannot derive, since choosing
    a runtime and recreating a sandbox are not filesystem operations.

**Sandboxes are per conversation, never per process.** Every entry point
resolves a session key from the LangGraph run -- the chatting user's id
and the thread id -- and all state hangs off that key. Two people talking
to the same pod get different sandboxes, cannot read each other's files,
and cannot reset or re-runtime each other's session. Module-global sandbox
state would hand one user's scratch files to the next.

Configuration:

    DAYTONA_API_KEY     Required; without it the tools report that they
                        are not configured rather than raising.
    DAYTONA_API_URL     Control plane. Defaults to the SDK's own default,
                        https://app.daytona.io/api. Point this at a
                        self-hosted control plane to keep traffic internal.
    DAYTONA_TARGET      Region, when the organisation has more than one.
    DAYTONA_IMAGE_{PYTHON,NODE,GO}
                        Per-runtime images.
    DAYTONA_LOG_COMMANDS
                        "true" to log command text and output previews.
                        Off by default: a command carries whatever the
                        user pasted into the chat, credentials included.
    DAYTONA_AUTO_DELETE_MINUTES, DAYTONA_IDLE_REAP_MINUTES,
    DAYTONA_MAX_SANDBOXES
                        Lifecycle limits; see below.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, cast

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendProtocol
from langchain_core.tools import tool
from langchain_daytona import DaytonaSandbox

from deep_agent.src.token_budget.identity import resolve_thread_id, resolve_user_id
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


API_KEY = os.environ.get("DAYTONA_API_KEY") or ""
API_URL = os.environ.get("DAYTONA_API_URL") or ""
TARGET = os.environ.get("DAYTONA_TARGET") or ""

# Used as published, with nothing added. BaseSandbox implements read, ls,
# grep, glob, edit and write as python3 scripts piped through the shell,
# and only `execute` is pure shell, so seven of the eight file tools fail
# on an image without python3. The python image has it; the node and go
# ones do not, and are left as their publishers ship them. Override the
# matching env var to supply an image that does have it.
RUNTIMES = {
    "python": os.environ.get("DAYTONA_IMAGE_PYTHON", "python:3.13-slim"),
    "node": os.environ.get("DAYTONA_IMAGE_NODE", "node:22-bookworm-slim"),
    "go": os.environ.get("DAYTONA_IMAGE_GO", "golang:1.25-bookworm"),
}
# Runtimes where only `execute` works, for the warning in use_runtime.
NO_PYTHON3 = ("node", "go")

DEFAULT_RUNTIME = os.environ.get("DAYTONA_RUNTIME", "python")
DEFAULT_TIMEOUT = int(os.environ.get("DAYTONA_EXEC_TIMEOUT", "120"))
CREATE_TIMEOUT = float(os.environ.get("DAYTONA_CREATE_TIMEOUT", "180"))
# Daytona stops an idle sandbox on its own; this is what we ask for. Zero
# disables it, which is not wanted here -- a preview that forgets a
# sandbox should not bill for it forever.
AUTO_STOP_MINUTES = int(os.environ.get("DAYTONA_AUTO_STOP_MINUTES", "30"))
# auto_stop only *stops* a sandbox: its disk survives and keeps costing.
# This is the one that makes it go away. Asked for at create time, so a
# pod killed before it can clean up still strands nothing permanently.
AUTO_DELETE_MINUTES = int(os.environ.get("DAYTONA_AUTO_DELETE_MINUTES", "120"))
# Belt to the provider's braces: sessions this process has not touched in
# this long have their sandboxes deleted on the next create. Daytona's own
# timer covers a pod that dies; this is what stops a long-lived pod from
# holding a sandbox open for every conversation it has ever served.
IDLE_REAP_SECONDS = int(os.environ.get("DAYTONA_IDLE_REAP_MINUTES", "60")) * 60
# Hard cap on concurrent sandboxes, least-recently-used evicted first. One
# sandbox per conversation on a metered provider is an unbounded bill
# without it.
MAX_SANDBOXES = int(os.environ.get("DAYTONA_MAX_SANDBOXES", "50"))
# Command text and output are user data -- whatever was pasted into the
# chat, credentials included. Off unless someone turns it on deliberately.
LOG_COMMANDS = (os.environ.get("DAYTONA_LOG_COMMANDS") or "").lower() in (
    "1",
    "true",
    "yes",
)
# Where the middleware offloads large tool results and evicted messages.
# It defaults to "/", which under this topology is the sandbox -- so a
# reset would throw away conversation history the model still needs to
# rehydrate. Routed back to the agent's own checkpointed backend below.
ARTIFACTS_ROOT = os.environ.get("SANDBOX_ARTIFACTS_ROOT", "/artifacts/")
# What to do when no thread id resolves. Isolation is keyed on it, so
# without one every caller would land in the same sandbox -- a silent
# cross-tenant leak, and silent is the worst way for this to fail. Default
# is to refuse. Set true for a single-user local script, never in a
# deployment that serves more than one person.
# What one conversation may take. The sandbox runs shell the model wrote,
# so a fork loop or a runaway download is an ordinary Tuesday, not an
# attack -- and without a cap it is the provider's bill and the other
# tenants' capacity.
CPU = int(os.environ.get("DAYTONA_CPU", "1"))
MEMORY_GB = int(os.environ.get("DAYTONA_MEMORY_GB", "2"))
DISK_GB = int(os.environ.get("DAYTONA_DISK_GB", "5"))
# Egress. Unset means the sandbox reaches the whole internet, which is
# what `pip install` needs and what a locked-down deployment will not
# want. A comma-separated allow list closes it to those domains; block-all
# closes it entirely and breaks package installs with it.
DOMAIN_ALLOW_LIST = [
    d.strip()
    for d in (os.environ.get("DAYTONA_DOMAIN_ALLOW_LIST") or "").split(",")
    if d.strip()
]
NETWORK_BLOCK_ALL = (os.environ.get("DAYTONA_NETWORK_BLOCK_ALL") or "").lower() in (
    "1",
    "true",
    "yes",
)
ALLOW_ANONYMOUS_SESSION = (
    os.environ.get("DAYTONA_ALLOW_ANONYMOUS_SESSION") or ""
).lower() in ("1", "true", "yes")

_client_singleton: Any = None
_lock = threading.Lock()

# The SDK's own default control plane. Named so the startup check can
# tell "operator chose the SaaS" from "operator chose nothing".
DAYTONA_SAAS_URL = "https://app.daytona.io/api"


def _warn_on_startup_posture() -> None:
    """Say out loud, once, what this configuration actually does.

    Two settings are safe to get wrong silently and expensive to discover
    later: sending user code to a third party, and collapsing every
    caller into one shared sandbox. Both are legitimate choices for a
    single-user preview and neither should be made by accident, so each
    is stated at startup rather than inferred from a missing env var.
    """
    if not API_KEY:
        return
    if not API_URL or API_URL.rstrip("/") == DAYTONA_SAAS_URL:
        logger.warning(
            "daytona_egress_leaves_cluster",
            api_url=API_URL or DAYTONA_SAAS_URL,
            detail=(
                "user files and model-authored shell commands are sent to "
                "Daytona's SaaS control plane; set DAYTONA_API_URL to a "
                "self-hosted control plane to keep this traffic internal"
            ),
        )
    if ALLOW_ANONYMOUS_SESSION:
        logger.warning(
            "daytona_anonymous_sessions_enabled",
            detail=(
                "every caller arriving without a thread id shares ONE "
                "sandbox; safe only where a single person can reach this "
                "agent"
            ),
        )
    if LOG_COMMANDS:
        logger.warning(
            "daytona_command_logging_enabled",
            detail=(
                "command text and output are written to the pod log, "
                "including any credential a user pasted into the chat"
            ),
        )


_warn_on_startup_posture()


def default_runtime() -> str:
    """The runtime a conversation starts in."""
    return DEFAULT_RUNTIME if DEFAULT_RUNTIME in RUNTIMES else "python"


@dataclass
class _Session:
    """One conversation's sandbox state.

    `runtime` is this conversation's choice and nobody else's. `sandboxes`
    holds what it has created, one per runtime, because switching runtime
    switches filesystem.
    """

    runtime: str
    sandboxes: dict[str, Any] = field(default_factory=dict)
    last_used: float = field(default_factory=time.monotonic)


_sessions: dict[str, _Session] = {}


class SessionUnavailable(RuntimeError):
    """Raised when the call cannot be attributed to a conversation.

    Its own type so the tools can refuse in a readable way. This is a
    safety stop, not a provider outage: without a thread id there is no
    key to isolate on, and handing the caller a shared sandbox would leak
    one conversation's files into another's without anything in the logs
    to show it happened.
    """


def session_key() -> str:
    """Identify the conversation the current call belongs to.

    The chatting user plus the thread, both read off the LangGraph run --
    Aegra's ``create_thread_config`` puts ``thread_id`` in every run
    config and ``inject_user_context`` adds ``user_id`` whenever a user is
    authenticated. Every sandbox and every runtime choice hangs off this,
    which is what stops one conversation reading, resetting or
    re-runtiming another.

    An unauthenticated run still has its thread id, so sessions stay
    separate under the ``anon`` user. No thread id at all means the call
    reached here outside a run -- or through an async path that dropped
    the context var -- and that is refused rather than quietly collapsed
    into one shared sandbox. ``DAYTONA_ALLOW_ANONYMOUS_SESSION`` lifts the
    refusal for a genuinely single-user process.
    """
    thread = resolve_thread_id()
    if not thread:
        if not ALLOW_ANONYMOUS_SESSION:
            raise SessionUnavailable(
                "no thread id on this run, so the sandbox cannot be isolated "
                "to one conversation; refusing rather than sharing one"
            )
        thread = "local"
    return f"{resolve_user_id() or 'anon'}/{thread}"


def _log(event: str, **fields: Any) -> None:
    """Write one structured line per sandbox event.

    Through the agent's own structlog pipeline, so these land with the
    trace, user and thread ids every other log line carries, at a level an
    operator can turn down. Command text and output stay out of `fields`
    unless `DAYTONA_LOG_COMMANDS` is set: the sandbox runs whatever the
    user typed, and that routinely includes a token.
    """
    try:
        logger.info(event, **fields)
    except Exception:  # noqa: BLE001 - logging must never break a run
        pass


class DaytonaNotConfigured(RuntimeError):
    """Raised when no API key is set.

    Its own type so the tools can report it as a readable message rather
    than a traceback: an unconfigured provider is an expected state for an
    off-cluster sandbox, and the agent should still answer without one.
    """


def _client() -> Any:
    """Return the Daytona client, built once per process.

    Imported lazily so that importing this module -- which graph.py does
    at startup -- cannot fail the agent on a missing SDK.
    """
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    if not API_KEY:
        raise DaytonaNotConfigured(
            "DAYTONA_API_KEY is not set, so there is no sandbox to run code in"
        )
    from daytona import Daytona, DaytonaConfig

    config: dict[str, str] = {"api_key": API_KEY}
    if API_URL:
        config["api_url"] = API_URL
    if TARGET:
        config["target"] = TARGET
    _client_singleton = Daytona(DaytonaConfig(**config))
    return _client_singleton


def _create_params(runtime: str, key: str) -> Any:
    """Build the create request for one conversation's sandbox.

    Three things beyond the image. `auto_delete_interval` is what actually
    reclaims the disk, since `auto_stop_interval` only stops the sandbox.
    `resources` caps what a single conversation can take -- the sandbox
    runs model-authored shell, so an unbounded fork loop or a runaway
    download is a thing that happens. `domain_allow_list` closes egress
    when an operator sets it; left unset the sandbox reaches the whole
    internet, which is what package installs need and what a locked-down
    deployment will not want.
    """
    from daytona import CreateSandboxFromImageParams, Resources

    kwargs: dict[str, Any] = {
        "image": RUNTIMES[runtime],
        "labels": {"app": "template-agent", "runtime": runtime, "session": key},
        "auto_stop_interval": AUTO_STOP_MINUTES,
        "auto_delete_interval": AUTO_DELETE_MINUTES,
        "resources": Resources(cpu=CPU, memory=MEMORY_GB, disk=DISK_GB),
    }
    if DOMAIN_ALLOW_LIST:
        kwargs["domain_allow_list"] = DOMAIN_ALLOW_LIST
    if NETWORK_BLOCK_ALL:
        kwargs["network_block_all"] = True
    return CreateSandboxFromImageParams(**kwargs)


def _delete_quietly(sandbox: Any, key: str, runtime: str) -> bool:
    """Delete one sandbox, reporting rather than raising on failure."""
    try:
        sandbox.delete()
    except Exception as exc:  # noqa: BLE001 - a leaked sandbox auto-deletes
        _log(
            "daytona_session_delete_failed",
            session=key,
            runtime=runtime,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
        return False
    return True


def _evict_idle(now: float) -> list[tuple[str, str, Any]]:
    """Drop sessions nothing has touched in a while.

    Call under `_lock`. Returns what was dropped so the caller can delete
    it outside the lock: a delete is an HTTPS round trip, and holding the
    registry across one would serialise every other conversation behind
    it.
    """
    doomed: list[tuple[str, str, Any]] = []
    for key in [
        k for k, s in _sessions.items() if now - s.last_used > IDLE_REAP_SECONDS
    ]:
        session = _sessions.pop(key)
        doomed.extend((key, rt, sb) for rt, sb in session.sandboxes.items())
    return doomed


def _evict_over_cap(protect: str) -> list[tuple[str, str, Any]]:
    """Drop least-recently-used sessions until the cap is met.

    Call under `_lock`, straight after a create -- the cap has to be
    enforced on the way out, not on the next call in, or the pod holds
    one sandbox over the limit for as long as nobody else asks.
    `protect` is the session being served right now, which is the one
    thing eviction must not take.
    """
    doomed: list[tuple[str, str, Any]] = []
    live = sum(len(s.sandboxes) for s in _sessions.values())
    for key in sorted(_sessions, key=lambda k: _sessions[k].last_used):
        if live <= MAX_SANDBOXES:
            break
        if key == protect:
            continue
        session = _sessions.pop(key)
        doomed.extend((key, rt, sb) for rt, sb in session.sandboxes.items())
        live -= len(session.sandboxes)
    return doomed


def _delete_all(doomed: list[tuple[str, str, Any]]) -> None:
    """Delete evicted sandboxes, outside the lock."""
    for key, runtime, sandbox in doomed:
        _log("daytona_session_reaped", session=key, runtime=runtime)
        _delete_quietly(sandbox, key, runtime)


def _shutdown() -> None:
    """Delete every sandbox this process still holds.

    Without this a rolling restart strands one sandbox per live
    conversation, and a stopped sandbox still bills for its disk.
    """
    with _lock:
        sessions = dict(_sessions)
        _sessions.clear()
    for key, session in sessions.items():
        for runtime, sandbox in session.sandboxes.items():
            _delete_quietly(sandbox, key, runtime)


atexit.register(_shutdown)


def _session(key: str) -> Any:
    """Return this conversation's sandbox, creating it on first use.

    Serialised: two concurrent calls on one conversation would otherwise
    both find no sandbox and create one, and the loser is leaked -- which
    on a metered provider is a bill, not just a stray object.
    """
    now = time.monotonic()
    with _lock:
        idle = _evict_idle(now)
    _delete_all(idle)

    with _lock:
        session = _sessions.get(key)
        if session is None:
            session = _Session(runtime=default_runtime())
            _sessions[key] = session
        session.last_used = now
        runtime = session.runtime

        existing = session.sandboxes.get(runtime)
        if existing is not None:
            return existing

        _log(
            "daytona_session_create",
            session=key,
            runtime=runtime,
            image=RUNTIMES[runtime],
        )
        started = time.time()
        sandbox = _client().create(_create_params(runtime, key), timeout=CREATE_TIMEOUT)
        _log(
            "daytona_session_ready",
            session=key,
            runtime=runtime,
            sandbox_id=getattr(sandbox, "id", None),
            duration_s=round(time.time() - started, 3),
        )
        session.sandboxes[runtime] = sandbox
        over_cap = _evict_over_cap(protect=key)

    _delete_all(over_cap)
    return sandbox


def active_runtime(key: str | None = None) -> str:
    """Return the runtime a conversation is currently running in."""
    key = key or session_key()
    with _lock:
        session = _sessions.get(key)
        return session.runtime if session else default_runtime()


class DaytonaRuntimeSandbox(DaytonaSandbox):
    """A `DaytonaSandbox` whose sandbox follows the calling conversation.

    `DaytonaSandbox` wraps one sandbox for its lifetime, but this backend
    is built once when the graph is assembled and has to serve every
    conversation the pod handles, across any number of runtime switches.
    Overriding the sandbox as a property is enough: the base class only
    ever reads it, and reading it per call is what keeps two users'
    filesystems apart.
    """

    def __init__(self) -> None:
        """Build a backend that resolves its sandbox per call.

        The base class wants a real `daytona.Sandbox`; the setter below
        drops whatever it is handed, so `None` is passed through `cast`
        rather than an ignore comment -- the comment reads as unused
        wherever the SDK's types do not resolve.
        """
        super().__init__(sandbox=cast(Any, None), timeout=DEFAULT_TIMEOUT)

    @property
    def _sandbox(self) -> Any:
        """Return the calling conversation's sandbox, created on first use."""
        return _session(session_key())

    @_sandbox.setter
    def _sandbox(self, value: Any) -> None:
        """Discard the base class's assignment.

        `DaytonaSandbox.__init__` stores the sandbox it was handed. The
        sandbox here is resolved per call instead, so the constructor's
        value -- deliberately None -- is dropped rather than kept.
        """

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        """Run a command, reporting setup failures instead of raising.

        An unconfigured provider and an unattributable call are both
        states the model should be told about plainly: one means nobody
        wired up a sandbox, the other means this call could not be tied to
        a conversation and will not be given a shared one. Neither is a
        bug in the turn, and a traceback here just makes the model try to
        debug the platform. A non-zero exit code keeps the base class from
        reading the message as command output.
        """
        from deepagents.backends.protocol import ExecuteResponse

        try:
            return super().execute(command, timeout=timeout)
        except (DaytonaNotConfigured, SessionUnavailable) as exc:
            _log("daytona_run_refused", reason=type(exc).__name__)
            return ExecuteResponse(
                output=f"error: {exc}", exit_code=127, truncated=False
            )

    def reset(self, runtime: str | None = None) -> str:
        """Delete this conversation's sandbox; the next command makes one.

        Scoped to the caller's session, so one user asking for a clean
        slate cannot wipe the files another user is working in.
        """
        key = session_key()
        with _lock:
            session = _sessions.get(key)
            name = runtime or (session.runtime if session else default_runtime())
            sandbox = session.sandboxes.pop(name, None) if session else None
        if sandbox is None:
            return f"no {name} sandbox was running; the next command will start one"
        if not _delete_quietly(sandbox, key, name):
            return (
                f"could not delete the {name} sandbox; it is no longer in use "
                f"here and will auto-stop after {AUTO_STOP_MINUTES}m"
            )
        _log("daytona_session_reset", session=key, runtime=name)
        return f"{name} sandbox destroyed; the next command will start a fresh one"


_sandbox_singleton: DaytonaRuntimeSandbox | None = None


def get_sandbox() -> DaytonaRuntimeSandbox:
    """Return the process-wide backend object.

    One object is fine and one sandbox is not: the object holds no
    sandbox of its own, and every method resolves its session per call.
    """
    global _sandbox_singleton
    if _sandbox_singleton is None:
        _sandbox_singleton = DaytonaRuntimeSandbox()
    return _sandbox_singleton


def wrap_backend(backend: BackendProtocol) -> BackendProtocol:
    """Put the sandbox underneath the agent's configured routes.

    The agent's own `get_configured_backend()` builds a `CompositeBackend`
    from agent.yaml: /skills/ and /data/ read-only, /memories/ and
    /reports/ in the LangGraph store, everything else in checkpointed
    state. Only that last part changes here -- the sandbox takes over as
    the default, which is both where loose paths go and what the
    middleware inspects to decide whether to offer the `execute` tool.

    The displaced backend is kept, routed at `ARTIFACTS_ROOT`, so the
    middleware's offloaded tool results and evicted messages stay in
    checkpointed state rather than in a sandbox that `reset()` can delete.
    """
    sandbox = get_sandbox()
    if isinstance(backend, CompositeBackend):
        routes = dict(backend.routes)
        displaced = backend.default
    else:
        routes, displaced = {}, backend
    routes.setdefault(ARTIFACTS_ROOT, displaced)
    return CompositeBackend(
        default=sandbox, routes=routes, artifacts_root=ARTIFACTS_ROOT
    )


@tool
def daytona_use_runtime(name: str) -> str:
    """Choose which language runtime the sandbox runs commands in.

    The choice applies to this conversation only. Each runtime is a
    separate Daytona sandbox with its own filesystem, so files written
    under one are NOT visible from another. The first command after
    switching waits for the sandbox to be created.

    Args:
        name: One of "python", "node", "go".
    """
    name = (name or "").strip().lower()
    if name not in RUNTIMES:
        return f"error: unknown runtime {name!r}; choose one of {sorted(RUNTIMES)}"
    try:
        key = session_key()
    except SessionUnavailable as exc:
        return f"error: {exc}"
    with _lock:
        session = _sessions.get(key)
        if session is None:
            session = _Session(runtime=name)
            _sessions[key] = session
        session.runtime = name
        session.last_used = time.monotonic()
    _log("daytona_runtime_selected", session=key, runtime=name, image=RUNTIMES[name])
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
def daytona_reset_sandbox() -> str:
    """Destroy this conversation's sandbox for a clean one.

    Use when previous state is getting in the way, or to discard anything
    an earlier command left behind. Files, installed packages and the
    working directory all go. Other runtimes, and everybody else's
    sandboxes, are untouched.
    """
    try:
        return get_sandbox().reset()
    except DaytonaNotConfigured as exc:
        return f"error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def get_builtin_tools() -> list:
    """Return the two tools the filesystem middleware does not provide.

    Everything else the model needs -- execute, read_file, write_file,
    edit_file, ls, glob, grep -- `FilesystemMiddleware` derives from the
    backend itself. Choosing a runtime and recreating the sandbox are not
    filesystem operations, so they have no equivalent there.
    """
    return [daytona_use_runtime, daytona_reset_sandbox]
