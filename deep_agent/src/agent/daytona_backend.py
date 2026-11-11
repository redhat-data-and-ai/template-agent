"""A Daytona-backed code sandbox for the agent's filesystem routes.

The same shape as the OpenShell backend on the sibling branch, against a
different sandbox provider. `BaseSandbox` turns read, write, edit, ls, glob
and grep into shell commands, so a backend only has to supply `execute()`
plus file transfer; everything else is derived. `wrap_backend()` then puts
this underneath the agent's configured routes as the default backend of its
`CompositeBackend`, which is also what makes `FilesystemMiddleware` offer
the `execute` tool at all.

Differences from the OpenShell backend, all of them consequences of the
sandbox living outside the cluster:

  * there is no pod to exec into, so `execute()` is one HTTPS round trip to
    the Daytona API rather than a websocket to the kubelet;
  * file transfer is a first-class SDK call instead of base64 smuggled
    through a command line;
  * the provider can be unreachable or unconfigured. That is a normal
    state here, not a bug, so it is reported as a tool error the model can
    read rather than an exception that fails the turn.

Configuration (all optional; without a key the tools report that they are
not configured):

    DAYTONA_API_KEY     API key. Required for the backend to do anything.
    DAYTONA_API_URL     Control plane. Defaults to the SDK's own default,
                        https://app.daytona.io/api. Point this at a
                        self-hosted control plane to keep traffic internal.
    DAYTONA_TARGET      Region, when the organisation has more than one.
    DAYTONA_*           The timeouts and runtime images below.
"""

from __future__ import annotations

import json
import os
import posixpath
import shlex
import threading
import time
import uuid
from typing import Any

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from langchain_core.tools import tool


def _emit(event: str, **fields: Any) -> None:
    """One JSON line per run event, keyed by run_id so runs stay separable.

    Written straight to stdout rather than through `logging`, for the same
    reason as the OpenShell backend: the agent configures the root logger at
    WARNING with no handlers, so logger.info() never reaches Loki. Here it
    matters more -- the sandbox is off-cluster, so these lines are the only
    record of a run that exists anywhere in the platform.
    """
    try:
        print(
            json.dumps(
                {"event": event, "logger": "daytona.run", "level": "info", **fields}
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 - logging must never break a run
        pass


API_KEY = os.environ.get("DAYTONA_API_KEY") or ""
API_URL = os.environ.get("DAYTONA_API_URL") or ""
TARGET = os.environ.get("DAYTONA_TARGET") or ""

# The same three runtimes the OpenShell branch offers, so a conversation
# behaves the same whichever backend is deployed.
#
# Resolved by Daytona, not by this cluster: the locally built
# openshell-sandbox-* images do not exist over there, and neither does the
# platform CA, so these are public registry references.
#
# Used as published, with nothing added -- the same trade the OpenShell
# branch makes with the Hummingbird images. BaseSandbox implements read,
# ls, grep, glob, edit and write as python3 scripts piped through the
# shell, and only `execute` is pure shell, so seven of the eight file tools
# fail on an image without python3. The python image has it; the node and
# go ones do not, and are left as their publishers ship them.
#
# Override any of these with the matching env var to supply an image that
# does have python3, if you want the file tools in that runtime.
RUNTIMES = {
    "python": os.environ.get("DAYTONA_IMAGE_PYTHON", "python:3.13-slim"),
    "node": os.environ.get("DAYTONA_IMAGE_NODE", "node:22-bookworm-slim"),
    "go": os.environ.get("DAYTONA_IMAGE_GO", "golang:1.25-bookworm"),
}
# Runtimes where only `execute` works, for the warning in use_runtime.
NO_PYTHON3 = ("node", "go")
DEFAULT_RUNTIME = os.environ.get("DAYTONA_RUNTIME", "python")

MAX_OUTPUT = int(os.environ.get("DAYTONA_MAX_OUTPUT", "16000"))
DEFAULT_TIMEOUT = int(os.environ.get("DAYTONA_EXEC_TIMEOUT", "120"))
CREATE_TIMEOUT = float(os.environ.get("DAYTONA_CREATE_TIMEOUT", "180"))
# Daytona stops an idle sandbox on its own; this is what we ask for. Zero
# disables it, which is not wanted here -- a preview that forgets a sandbox
# should not bill for it forever.
AUTO_STOP_MINUTES = int(os.environ.get("DAYTONA_AUTO_STOP_MINUTES", "30"))

WORKDIR = os.environ.get("DAYTONA_WORKDIR", "/tmp/session")
# Where the middleware offloads large tool results and evicted messages. It
# defaults to "/", which under this topology is the sandbox -- so a reset
# would throw away conversation history the model still needs. Routed back
# to the agent's own checkpointed backend by wrap_backend().
ARTIFACTS_ROOT = os.environ.get("SANDBOX_ARTIFACTS_ROOT", "/artifacts/")

_active_runtime = DEFAULT_RUNTIME if DEFAULT_RUNTIME in RUNTIMES else "python"
_daytona: Any = None
_sandboxes: dict[str, Any] = {}
_session_lock = threading.Lock()


class DaytonaNotConfigured(RuntimeError):
    """Raised when no API key is set.

    Its own type so `execute()` can turn it into a readable tool result
    instead of a traceback: an unconfigured provider is an expected state
    for this preview, and the model should be told plainly rather than
    shown a stack trace it will try to debug.
    """


def _client() -> Any:
    """The Daytona client, built once per process.

    Imported lazily so that merely importing this module -- which
    graph.py does at startup -- cannot fail the agent on a missing or
    broken SDK. An agent that cannot run code should still answer.
    """
    global _daytona
    if _daytona is not None:
        return _daytona
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
    _daytona = Daytona(DaytonaConfig(**config))
    return _daytona


class DaytonaSandbox(BaseSandbox):
    """A Daytona sandbox driven through the Daytona SDK.

    One sandbox per runtime, created on first use and reused across calls,
    so state persists the way it does on the OpenShell branch. Switching
    runtime switches sandbox; the filesystems are separate.
    """

    # The capture wrapper assumes a shell and coreutils the base class does
    # not guarantee, and the SDK already returns exit code and output
    # together, so there is nothing to gain by capturing at the source.
    enable_capture_offload = False

    @property
    def id(self) -> str:
        """Identify the active sandbox by its runtime."""
        return f"daytona/{_active_runtime}"

    # -- session -------------------------------------------------------

    def _session(self) -> Any:
        """Return the active runtime's sandbox, creating it if needed.

        Serialised: two concurrent calls would otherwise both find no
        sandbox and create one, and the loser's sandbox is leaked -- which
        on a metered provider is a bill, not just a stray object.
        """
        runtime = _active_runtime
        with _session_lock:
            existing = _sandboxes.get(runtime)
            if existing is not None:
                return existing

            from daytona import CreateSandboxFromImageParams

            _emit("daytona_session_create", runtime=runtime, image=RUNTIMES[runtime])
            started = time.time()
            sandbox = _client().create(
                CreateSandboxFromImageParams(
                    image=RUNTIMES[runtime],
                    labels={"app": "template-agent", "runtime": runtime},
                    auto_stop_interval=AUTO_STOP_MINUTES,
                ),
                timeout=CREATE_TIMEOUT,
            )
            _emit(
                "daytona_session_ready",
                runtime=runtime,
                sandbox_id=getattr(sandbox, "id", None),
                duration_s=round(time.time() - started, 3),
            )
            _sandboxes[runtime] = sandbox

        # Outside the lock: a slow mkdir should not hold up another runtime.
        sandbox.process.exec(f"mkdir -p {shlex.quote(WORKDIR)}", timeout=60)
        return sandbox

    def reset(self, runtime: str | None = None) -> str:
        """Delete a sandbox; the next command creates a fresh one."""
        name = runtime or _active_runtime
        with _session_lock:
            sandbox = _sandboxes.pop(name, None)
        if sandbox is None:
            return f"no {name} sandbox was running; the next command will start one"
        try:
            sandbox.delete()
        except Exception as exc:  # noqa: BLE001 - a leaked sandbox auto-stops
            _emit(
                "daytona_session_reset_failed",
                runtime=name,
                error=f"{type(exc).__name__}: {exc}",
            )
            return (
                f"could not delete the {name} sandbox ({exc}); it is no longer "
                f"in use here and will auto-stop after {AUTO_STOP_MINUTES}m"
            )
        _emit("daytona_session_reset", runtime=name)
        return f"{name} sandbox destroyed; the next command will start a fresh one"

    # -- the exec primitive --------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the sandbox's working directory.

        Every sandbox command passes through here, including the ones the
        base class generates for `read_file`, `grep` and the rest, so this
        is the one place a run can be recorded.

        Args:
            command: Full shell command string.
            timeout: Seconds to allow. Defaults to DAYTONA_EXEC_TIMEOUT.

        Returns:
            `ExecuteResponse` with the command's combined output.
        """
        timeout_s = max(1, min(int(timeout or DEFAULT_TIMEOUT), 600))
        run_id = uuid.uuid4().hex[:8]
        started = time.time()
        _emit(
            "daytona_run_started",
            run_id=run_id,
            runtime=_active_runtime,
            timeout_s=timeout_s,
            command_bytes=len(command),
            command_preview=command[:400],
        )
        try:
            sandbox = self._session()
            result = sandbox.process.exec(command, cwd=WORKDIR, timeout=timeout_s)
            body = result.result or ""
            code = result.exit_code
        except DaytonaNotConfigured as exc:
            # Deliberately not raised: an unconfigured provider is a state
            # the model can report, not a turn-ending failure. A non-zero
            # exit code keeps the base class from parsing this as output.
            _emit("daytona_run_finished", run_id=run_id, status="unconfigured")
            return ExecuteResponse(
                output=f"error: {exc}", exit_code=127, truncated=False
            )
        except Exception as exc:  # noqa: BLE001
            _emit(
                "daytona_run_finished",
                run_id=run_id,
                status="error",
                duration_s=round(time.time() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        truncated = len(body) > MAX_OUTPUT
        if truncated:
            body = f"{body[:MAX_OUTPUT]}\n...[truncated, {len(body) - MAX_OUTPUT} more bytes]"
        _emit(
            "daytona_run_finished",
            run_id=run_id,
            status="ok" if code == 0 else "error",
            exit_code=code,
            duration_s=round(time.time() - started, 3),
            output_preview=body[:800],
        )
        return ExecuteResponse(output=body, exit_code=code, truncated=truncated)

    # -- file transfer -------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write files into the sandbox, creating parent directories.

        Unlike the OpenShell backend there is no base64-through-the-command-
        line dance here: the SDK has a real upload endpoint. Parent
        directories still have to be made first -- upload_file does not
        create them.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            target = _abs(path)
            try:
                sandbox = self._session()
                parent = posixpath.dirname(target) or "/"
                sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}", timeout=60)
                sandbox.fs.upload_file(content, target)
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:  # noqa: BLE001 - partial success is the contract
                responses.append(
                    FileUploadResponse(
                        path=path, error=_file_error(f"{type(exc).__name__}: {exc}")
                    )
                )
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read files out of the sandbox as bytes."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            target = _abs(path)
            try:
                sandbox = self._session()
                responses.append(
                    FileDownloadResponse(
                        path=path, content=sandbox.fs.download_file(target)
                    )
                )
            except Exception as exc:  # noqa: BLE001 - partial success is the contract
                responses.append(
                    FileDownloadResponse(
                        path=path, error=_file_error(f"{type(exc).__name__}: {exc}")
                    )
                )
        return responses


def _file_error(detail: str) -> str:
    """Map a provider failure onto the standard FileOperationError codes."""
    low = (detail or "").lower()
    if "not found" in low or "no such file" in low or "404" in low:
        return "file_not_found"
    if "permission denied" in low or "forbidden" in low or "403" in low:
        return "permission_denied"
    if "is a directory" in low:
        return "is_directory"
    return (detail or "unknown error").strip()[:200]


def _abs(path: str) -> str:
    """Resolve a path the model supplied against the sandbox working dir."""
    path = (path or ".").strip()
    return (
        path
        if posixpath.isabs(path)
        else posixpath.normpath(posixpath.join(WORKDIR, path))
    )


_sandbox: DaytonaSandbox | None = None


def get_sandbox() -> DaytonaSandbox:
    """The process-wide sandbox backend."""
    global _sandbox
    if _sandbox is None:
        _sandbox = DaytonaSandbox()
    return _sandbox


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

    Each runtime is a separate Daytona sandbox with its own filesystem, so
    files written under one are NOT visible from another. The first command
    after switching waits for the sandbox to be created; later ones do not.

    Args:
        name: One of "python", "node", "go".
    """
    global _active_runtime
    name = (name or "").strip().lower()
    if name not in RUNTIMES:
        return f"error: unknown runtime {name!r}; choose one of {sorted(RUNTIMES)}"
    _active_runtime = name
    _emit("daytona_runtime_selected", runtime=name, image=RUNTIMES[name])
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
    """Destroy the current runtime's sandbox for a clean one.

    Use when previous state is getting in the way, or to discard anything
    an earlier command left behind. Files, installed packages and the
    working directory all go. Other runtimes are untouched.
    """
    try:
        return get_sandbox().reset()
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def get_builtin_tools() -> list:
    """The two tools the filesystem middleware does not already provide.

    Everything else the model needs -- execute, read_file, write_file,
    edit_file, ls, glob, grep -- `FilesystemMiddleware` derives from the
    backend itself. Choosing a runtime and recreating the sandbox are not
    filesystem operations, so they have no equivalent there.
    """
    return [daytona_use_runtime, daytona_reset_sandbox]
