"""Agent backend for state management and skill execution.

This module provides the backend infrastructure for agents to execute skills
in isolated Python environments. It creates dedicated virtual environments for
skill execution, manages dependencies from config/skills/pyproject.toml, and
provides a safe execution sandbox.

Why this exists:
    Skills need to run Python code with specific dependencies without polluting
    the main application environment. This backend creates isolated venvs for
    safe execution of agent skills.

Functions:
    get_backend: Get or create the configured backend instance
    initialize_backend: One-time backend initialization at app startup
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult, FileUploadResponse, WriteResult

from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
_PASSTHROUGH_VARS = ("HOME", "USER", "LANG", "LC_ALL", "TZ", "TERM")


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_backend: LocalShellBackend | None = None


def _backend_accepts_runtime(cls: type) -> bool:
    """Return True if *cls.__init__* takes a positional ``runtime`` argument.

    PyPI deepagents 0.7.6 uses no-arg StateBackend/StoreBackend constructors.
    Later builds require ``ToolRuntime`` as the first argument. Detect at
    runtime so the same code works against both.
    """
    try:
        param = inspect.signature(cls).parameters.get("runtime")
    except (TypeError, ValueError):
        return False
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def _make_state_backend(runtime: Any) -> Any:
    """Instantiate StateBackend, passing runtime only when the ctor requires it."""
    from deepagents.backends.state import StateBackend

    if _backend_accepts_runtime(StateBackend):
        return StateBackend(runtime)
    return StateBackend()


def _make_store_backend(runtime: Any, namespace: Any) -> Any:
    """Instantiate StoreBackend, passing runtime only when the ctor requires it."""
    from deepagents.backends.store import StoreBackend

    if _backend_accepts_runtime(StoreBackend):
        return StoreBackend(runtime, namespace=namespace)
    return StoreBackend(namespace=namespace)


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """FilesystemBackend that rejects all write operations."""

    def write(self, file_path: str, content: str) -> WriteResult:
        """Reject write operations."""
        return WriteResult(error="Read-only backend: writes not permitted")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Reject edit operations."""
        return EditResult(error="Read-only backend: edits not permitted")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject upload operations."""
        return [
            FileUploadResponse(path=p, error="Read-only backend: uploads not permitted")
            for p, _ in files
        ]


def _text_from_read(result: Any) -> str | None:
    """Return file text from a backend ``read``/``aread`` result.

    deepagents 0.7.x returns :class:`ReadResult`, not a string. Callers that
    treat the result as ``str`` (e.g. ``.strip()``) crash after a successful
    ``edit_file``.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if getattr(result, "error", None):
        return None
    file_data = getattr(result, "file_data", None)
    if isinstance(file_data, dict):
        content = file_data.get("content")
        if isinstance(content, str):
            return content
    return None


class DeduplicatingStoreBackend:
    """Wrapper around StoreBackend that deduplicates memory content on write.

    Intercepts write/edit calls to memory files and removes near-duplicate
    lines before persisting. All other operations are proxied unchanged.
    """

    def __init__(self, inner: Any, memory_prefix: str = "/memories/") -> None:
        """Wrap *inner* backend, deduplicating writes under *memory_prefix*."""
        self._inner = inner
        self._memory_prefix = memory_prefix

    def _deduplicate_content(self, content: str) -> str:
        """Remove near-duplicate lines from memory file content."""
        import re

        from deep_agent.src.memory.clustering import near_duplicate_groups

        lines = content.strip().split("\n")
        facts: list[str] = []
        non_fact_lines: list[str] = []

        for line in lines:
            cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
            if cleaned:
                facts.append(cleaned)
            elif line.strip():
                non_fact_lines.append(line)

        if len(facts) < 2:
            return content

        # Only drop restatements of the same fact (70kg vs 70 kg). Never
        # drop "joining date" because it looks a bit like "date of birth".
        clusters = near_duplicate_groups(facts)
        indices_to_remove: set[int] = set()
        for group in clusters:
            longest_idx = max(group, key=lambda i: len(facts[i]))
            for idx in group:
                if idx != longest_idx:
                    indices_to_remove.add(idx)

        if not indices_to_remove:
            return content

        deduped_facts = [f for i, f in enumerate(facts) if i not in indices_to_remove]
        result_lines = non_fact_lines + [f"- {f}" for f in deduped_facts]
        logger.debug(
            "Deduplicated memory: %d facts → %d (removed %d)",
            len(facts),
            len(deduped_facts),
            len(indices_to_remove),
        )
        return "\n".join(result_lines) + "\n"

    def _is_memory_path(self, file_path: str) -> bool:
        return True

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write *content* to *file_path*, deduplicating memory files."""
        if self._is_memory_path(file_path):
            content = self._deduplicate_content(content)
        return self._inner.write(file_path, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async write with memory deduplication."""
        if self._is_memory_path(file_path):
            content = self._deduplicate_content(content)
        return await self._inner.awrite(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit file, then deduplicate if it is a memory file."""
        result = self._inner.edit(file_path, old_string, new_string, replace_all)
        if self._is_memory_path(file_path) and not getattr(result, "error", None):
            current = _text_from_read(self._inner.read(file_path))
            if current:
                deduped = self._deduplicate_content(current)
                if deduped != current:
                    self._inner.write(file_path, deduped)
        return result

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async edit with memory deduplication."""
        result = await self._inner.aedit(file_path, old_string, new_string, replace_all)
        if self._is_memory_path(file_path) and not getattr(result, "error", None):
            current = _text_from_read(await self._inner.aread(file_path))
            if current:
                deduped = self._deduplicate_content(current)
                if deduped != current:
                    await self._inner.awrite(file_path, deduped)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other methods to the inner backend."""
        return getattr(self._inner, name)


def _base_python() -> str:
    """Resolve the base (non-venv) Python so the agent venv is independent.

    Prefers the versioned binary (e.g. python3.12) to avoid picking up the
    UBI9 system python3 → 3.9 symlink when the app runs inside a 3.12 venv.
    """
    if sys.prefix != sys.base_prefix:
        v = sys.version_info
        base_bin = Path(sys.base_prefix) / "bin"
        for name in (f"python{v.major}.{v.minor}", "python3"):
            candidate = base_bin / name
            if candidate.exists():
                return str(candidate)
    return sys.executable


def _ensure_venv(root_dir: Path, pyproject: Path) -> Path:
    """Create an isolated venv in user cache directory and install from *pyproject*.

    The venv directory is keyed by a hash of *root_dir* **and** the contents of
    *pyproject* so a changed ``pyproject.toml`` triggers a reinstall.

    Uses /app/.cache/template-agent/venvs/ (or ~/.cache/ outside containers) to
    avoid security risks with world-readable /tmp directories on shared hosts.
    """
    project_hash = hashlib.sha256(str(root_dir.resolve()).encode()).hexdigest()[:12]
    toml_hash = hashlib.sha256(pyproject.read_bytes()).hexdigest()[:8]

    # Prefer /app/.cache inside containers (always writable on OpenShift);
    # fall back to /tmp then ~/.cache for local / non-container runs.
    # OpenShift runs with arbitrary UID so Path.home() may not resolve.
    app_cache = Path("/app/.cache")
    if app_cache.parent.is_dir():
        base_cache = app_cache
    else:
        try:
            base_cache = Path.home() / ".cache"
        except (RuntimeError, KeyError):
            base_cache = Path("/tmp/.cache")  # noqa: S108 — OpenShift arbitrary UID fallback
    cache_dir = base_cache / "template-agent" / "venvs"
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)  # User-only permissions

    venv_dir = cache_dir / f"agent-venv-{project_hash}"
    stamp = venv_dir / ".toml_hash"

    needs_install = False

    if not (venv_dir / "bin" / "python").exists():
        base = _base_python()
        logger.info(f"Creating agent venv at {venv_dir} (python: {base})")
        subprocess.run(
            [base, "-m", "venv", "--clear", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        needs_install = True

    if not needs_install and stamp.exists() and stamp.read_text() == toml_hash:
        logger.info(f"Agent venv up-to-date ({venv_dir})")
        return venv_dir

    # If pyproject.toml changed, clear the venv to remove stale dependencies
    if stamp.exists() and stamp.read_text() != toml_hash:
        base = _base_python()
        logger.info(f"pyproject.toml changed — clearing venv at {venv_dir}")
        subprocess.run(
            [base, "-m", "venv", "--clear", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    pkg_dir = venv_dir / "_pkg"
    pkg_dir.mkdir(exist_ok=True)
    shutil.copy2(pyproject, pkg_dir / "pyproject.toml")

    pip = str(venv_dir / "bin" / "pip")
    logger.info(f"Installing dependencies from {pyproject.name}")
    result = subprocess.run(
        [pip, "install", "--quiet", str(pkg_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed: {result.stderr.strip()}")

    stamp.write_text(toml_hash)
    return venv_dir


def _build_env(venv_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal env: allowlisted host vars + venv activation + optional overrides."""
    env = {k: os.environ[k] for k in _PASSTHROUGH_VARS if k in os.environ}
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir}/bin:{_SYSTEM_PATH}"
    if extra:
        env.update(extra)
    return env


def create_backend(
    root_dir: Path,
    pyproject: Path,
    *,
    timeout: int = 120,
    max_output_bytes: int = 100_000,
    extra_env: dict[str, str] | None = None,
) -> LocalShellBackend:
    """Create a :class:`LocalShellBackend` backed by an isolated agent venv.

    Args:
        root_dir: Shell working directory.
        pyproject: Path to a ``pyproject.toml`` whose dependencies are installed.
        timeout: Default per-command timeout in seconds.
        max_output_bytes: Max captured output before truncation.
        extra_env: Extra env vars (highest priority).
    """
    if not pyproject.is_file():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject}")

    venv_dir = _ensure_venv(root_dir, pyproject)
    env = _build_env(venv_dir, extra_env)

    logger.info(f"Backend ready — venv={venv_dir}, pyproject={pyproject}")
    return LocalShellBackend(
        root_dir=str(root_dir),
        virtual_mode=False,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        env=env,
    )


def get_backend(
    root_dir: Path | None = None,
    pyproject: Path | None = None,
    *,
    timeout: int = 120,
    max_output_bytes: int = 100_000,
    extra_env: dict[str, str] | None = None,
) -> LocalShellBackend:
    """Return the singleton backend, creating it on the first call.

    Subsequent calls return the same instance regardless of arguments.
    When *root_dir* or *pyproject* are ``None`` the module-level defaults
    (``_REPO_ROOT`` / ``agent_config.get_pyproject_path()``) are used.
    """
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = create_backend(
            root_dir or _REPO_ROOT,
            pyproject or agent_config.get_pyproject_path(),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            extra_env=extra_env,
        )
    return _backend


def get_configured_backend() -> LocalShellBackend | Any:
    """Return the backend configured by filesystem.yaml or agent.yaml.

    Reads the backend type from config and builds the appropriate backend:
    - state: StateBackend (thread-scoped scratch, recommended for production)
    - composite: CompositeBackend (routes paths to different backends)
    - store: StoreBackend (cross-thread persistent via LangGraph Store)
    - local_shell: LocalShellBackend (local dev only — NOT for deployed agents)

    Falls back to StateBackend if config is missing or invalid.
    """
    config_path = agent_config.base_dir / "filesystem.yaml"
    if config_path.is_file():
        from deep_agent.src.agent.config.filesystem import load_filesystem_config

        fs_config = load_filesystem_config(config_path)
    else:
        fs_config = agent_config.get_filesystem_config()

    backend_type = fs_config.backend.type

    if backend_type == "state":
        return _build_state_backend()

    if backend_type == "store":
        return _build_store_backend(fs_config)

    if backend_type == "composite":
        return _build_composite_backend(fs_config)

    if backend_type == "local_shell":
        logger.warning(
            "LocalShellBackend accesses the host directly. "
            "Do NOT use in deployed agents (OpenShift, LangSmith, etc.). "
            "Set backend.type to 'state' or 'composite' for production."
        )
        return get_backend(
            timeout=fs_config.backend.local_shell.timeout,
            max_output_bytes=fs_config.backend.local_shell.max_output_bytes,
        )

    # Fallback for any backend type not explicitly handled above
    logger.warning("Unknown backend type '%s', falling back to state", backend_type)  # type: ignore[unreachable]
    return _build_state_backend()


def _build_state_backend() -> Any:
    """Build a StateBackend instance (thread-scoped scratch space).

    Recommended for production. Files persist across turns within a thread
    via checkpointer but are not shared across threads.
    """
    try:
        from deepagents.backends.state import StateBackend

        logger.info("Using StateBackend (thread-scoped scratch)")
        return StateBackend()
    except ImportError:
        logger.warning("StateBackend not available, falling back to LocalShellBackend")
        return get_backend()


def _as_runtime(ctx: Any) -> Any:
    """Return the LangGraph Runtime from a namespace-factory argument."""
    if ctx is None:
        return None
    inner = getattr(ctx, "runtime", None)
    return ctx if inner is None else inner


def _get_assistant_id_from_config(ctx: Any) -> str:
    """Extract assistant_id from runtime config metadata, falling back to 'default'.

    Mirrors the fallback logic in StoreBackend._get_namespace_legacy:
    check runtime.config → metadata → assistant_id.
    """
    runtime = _as_runtime(ctx)
    cfg = getattr(runtime, "config", None) or {}
    if isinstance(cfg, dict):
        metadata = cfg.get("metadata")
        assistant_id: Any = (
            metadata.get("assistant_id") if isinstance(metadata, dict) else None
        )
        if assistant_id:
            return str(assistant_id)
    return "default"


def _get_user_id_from_runtime(runtime: Any) -> str | None:
    """Extract the BFF user id from run config (metadata or configurable).

    The UI proxy sets ``config.metadata.user_id`` (and ``configurable.user_id``)
    to the same value as ``X-User-ID``, so local runs without Aegra auth still
    get a per-user Store namespace.
    """
    cfg = getattr(runtime, "config", None) or {}
    if not isinstance(cfg, dict):
        return None
    for bag_name in ("metadata", "configurable"):
        bag = cfg.get(bag_name)
        if not isinstance(bag, dict):
            continue
        uid = bag.get("user_id") or bag.get("x_user_id")
        if uid:
            uid_str = str(uid).strip()
            if uid_str:
                return uid_str
    return None


def _preferred_username_from_runtime(runtime: Any) -> str | None:
    """Read ``preferred_username`` from the access token on the runtime user."""
    candidates = [getattr(runtime, "user", None)]
    si = getattr(runtime, "server_info", None)
    if si is not None:
        candidates.append(getattr(si, "user", None))
    for user in candidates:
        token = getattr(user, "access_token", None) if user is not None else None
        if not isinstance(token, str) or token.count(".") < 2:
            continue
        try:
            import base64

            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload_b64))
            username = claims.get("preferred_username")
            if username:
                return str(username)
        except Exception:
            continue
    return None


def _safe_namespace_user(ctx: Any) -> tuple[str, ...]:
    """User-scoped Store namespace: (assistant_id, memory_user_id).

    Memories are keyed by the BFF user id (``X-User-ID`` /
    ``preferred_username``), not JWT ``sub``. That matches Settings REST
    (:func:`memory_user_id`) and run ``metadata.user_id``.

    Resolution for the user component:
    1. Run metadata / configurable ``user_id`` (BFF)
    2. ``preferred_username`` on the access token
    3. ``server_info.user.identity`` (JWT ``sub``) as last resort
    """
    runtime = _as_runtime(ctx)
    si: Any = getattr(runtime, "server_info", None)
    meta_user = _get_user_id_from_runtime(runtime)
    jwt_username = _preferred_username_from_runtime(runtime)
    identity: str | None = None
    if si is not None:
        user: Any = getattr(si, "user", None)
        raw_identity = getattr(user, "identity", None) if user is not None else None
        if raw_identity:
            identity = str(raw_identity)

    if si is not None and getattr(si, "assistant_id", None):
        assistant = str(si.assistant_id)
    else:
        assistant = _get_assistant_id_from_config(ctx)

    user_part = meta_user or jwt_username or identity
    if user_part:
        return (assistant, user_part)
    return (assistant,)


def _safe_namespace_assistant(ctx: Any) -> tuple[str, ...]:
    """Assistant-scoped namespace: (assistant_id,) on server, config fallback locally."""
    si: Any = getattr(_as_runtime(ctx), "server_info", None)
    if si is not None and getattr(si, "assistant_id", None):
        return (si.assistant_id,)
    return (_get_assistant_id_from_config(ctx),)


def _safe_namespace_org(ctx: Any) -> tuple[str, ...]:
    """Org-scoped namespace: (org_id,)."""
    runtime = _as_runtime(ctx)
    return (runtime.context.org_id,)


_STORE_NAMESPACE_FACTORIES: dict[str, Any] = {
    "user": _safe_namespace_user,
    "assistant": _safe_namespace_assistant,
    "org": _safe_namespace_org,
}


def _build_store_backend(fs_config: Any) -> Any:
    """Build a StoreBackend (cross-thread persistent via LangGraph Store).

    Scope determines namespace partitioning:
    - user: per-user private memory (recommended)
    - assistant: shared across all users of one assistant
    - org: shared across all users and assistants
    """
    try:
        from deepagents.backends.store import StoreBackend

        scope = getattr(fs_config.backend, "store", None)
        scope_name = scope.scope if scope else "user"

        namespace = _STORE_NAMESPACE_FACTORIES.get(scope_name)
        if namespace is None:
            logger.warning("Unknown store scope '%s', using 'user'", scope_name)
            namespace = _safe_namespace_user

        logger.info("Using StoreBackend (scope=%s)", scope_name)
        return StoreBackend(namespace=namespace)
    except ImportError:
        logger.warning("StoreBackend not available, falling back to StateBackend")
        return _build_state_backend()


def _build_composite_backend(fs_config: Any) -> Any:
    """Build a CompositeBackend instance with configured routes."""
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend

    state_backend = StateBackend()
    routes: dict[str, Any] = {}

    for path_prefix, backend_name in fs_config.backend.routes.items():
        if backend_name == "filesystem_readonly":
            dir_name = path_prefix.strip("/")
            routes[path_prefix] = _build_filesystem_readonly_backend(
                agent_config.base_dir / dir_name
            )

    if any(v == "local_shell" for v in fs_config.backend.routes.values()):
        logger.warning(
            "local_shell in composite routes — not recommended for production"
        )
        local_shell_backend = get_backend(
            timeout=fs_config.backend.local_shell.timeout,
            max_output_bytes=fs_config.backend.local_shell.max_output_bytes,
        )
        for path_prefix, backend_name in fs_config.backend.routes.items():
            if backend_name == "local_shell":
                routes[path_prefix] = local_shell_backend

    store_route_prefixes = [
        p for p, v in fs_config.backend.routes.items() if v == "store"
    ]
    if store_route_prefixes:
        try:
            from deepagents.backends.store import StoreBackend

            scope = getattr(fs_config.backend, "store", None)
            store_scope = scope.scope if scope else "user"
            ns = _STORE_NAMESPACE_FACTORIES.get(store_scope, _safe_namespace_user)
            store_backend = StoreBackend(namespace=ns)
            for prefix in store_route_prefixes:
                if prefix.rstrip("/").endswith("memories"):
                    routes[prefix] = DeduplicatingStoreBackend(
                        store_backend, memory_prefix=prefix
                    )
                else:
                    routes[prefix] = store_backend
        except ImportError:
            logger.warning(
                "StoreBackend not available — store routes will use StateBackend"
            )
            for prefix in store_route_prefixes:
                routes[prefix] = state_backend

    known_types = {"filesystem_readonly", "local_shell", "store", "state"}
    for path_prefix, backend_name in fs_config.backend.routes.items():
        if backend_name not in known_types:
            logger.warning(
                "Unknown backend '%s' in route for '%s'", backend_name, path_prefix
            )

    logger.info(
        "Built CompositeBackend: %d route(s), default=StateBackend",
        len(routes),
    )

    default_backend = routes.pop("/", state_backend)
    return CompositeBackend(default=default_backend, routes=routes)


def _build_filesystem_readonly_backend(root_dir: Path) -> ReadOnlyFilesystemBackend:
    """Build a read-only FilesystemBackend jailed to root_dir.

    Uses virtual_mode=True to jail all paths within the given directory.
    Write/edit/upload operations are explicitly blocked for defense-in-depth.

    Args:
        root_dir: Directory to use as the filesystem root. Derived from
            the route prefix in agent.yaml (e.g., "/skills/" → base_dir/skills).
    """
    if not root_dir.is_dir():
        logger.warning(
            "Directory does not exist: %s — reads will return empty results",
            root_dir,
        )

    logger.info(
        "Using ReadOnlyFilesystemBackend (root=%s, virtual_mode=True)", root_dir
    )
    return ReadOnlyFilesystemBackend(root_dir=str(root_dir), virtual_mode=True)
