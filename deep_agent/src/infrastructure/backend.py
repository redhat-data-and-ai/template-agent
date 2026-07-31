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
    """Build a StateBackend factory (thread-scoped scratch space).

    Recommended for production. Files persist across turns within a thread
    via checkpointer but are not shared across threads.

    Returns the StateBackend class as a factory — create_deep_agent calls it
    with ToolRuntime at execution time.
    """
    try:
        from deepagents.backends.state import StateBackend

        logger.info("Using StateBackend (thread-scoped scratch)")
        return StateBackend
    except ImportError:
        logger.warning("StateBackend not available, falling back to LocalShellBackend")
        return get_backend()


def _get_assistant_id_from_config(ctx: Any) -> str:
    """Extract assistant_id from runtime config metadata, falling back to 'default'.

    Mirrors the fallback logic in StoreBackend._get_namespace_legacy:
    check runtime.config → metadata → assistant_id.
    """
    cfg = getattr(ctx.runtime, "config", None) or {}
    if isinstance(cfg, dict):
        metadata = cfg.get("metadata")
        assistant_id: Any = (
            metadata.get("assistant_id") if isinstance(metadata, dict) else None
        )
        if assistant_id:
            return str(assistant_id)
    return "default"


def _safe_namespace_user(ctx: Any) -> tuple[str, ...]:
    """User-scoped namespace: (assistant_id, user_identity) on server, config fallback locally."""
    si: Any = getattr(ctx.runtime, "server_info", None)
    if si is not None and getattr(si, "assistant_id", None):
        parts: list[str] = [si.assistant_id]
        user: Any = getattr(si, "user", None)
        if getattr(user, "identity", None):
            parts.append(user.identity)
        return tuple(parts)
    return (_get_assistant_id_from_config(ctx),)


def _safe_namespace_assistant(ctx: Any) -> tuple[str, ...]:
    """Assistant-scoped namespace: (assistant_id,) on server, config fallback locally."""
    si: Any = getattr(ctx.runtime, "server_info", None)
    if si is not None and getattr(si, "assistant_id", None):
        return (si.assistant_id,)
    return (_get_assistant_id_from_config(ctx),)


def _safe_namespace_org(ctx: Any) -> tuple[str, ...]:
    """Org-scoped namespace: (org_id,)."""
    return (ctx.runtime.context.org_id,)


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
    """Return a factory that builds a CompositeBackend at request time.

    StateBackend and StoreBackend require ToolRuntime (only available per-request),
    so we return a callable. ReadOnlyFilesystemBackend and LocalShellBackend are
    built eagerly since they don't need runtime.
    """
    # --- Eager: backends that don't need runtime ---
    eager_routes: dict[str, Any] = {}

    for path_prefix, backend_name in fs_config.backend.routes.items():
        if backend_name == "filesystem_readonly":
            dir_name = path_prefix.strip("/")
            eager_routes[path_prefix] = _build_filesystem_readonly_backend(
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
                eager_routes[path_prefix] = local_shell_backend

    # --- Deferred config (captured for use inside factory) ---
    store_route_prefixes = [
        p for p, v in fs_config.backend.routes.items() if v == "store"
    ]
    store_scope: str | None = None
    if store_route_prefixes:
        scope = getattr(fs_config.backend, "store", None)
        store_scope = scope.scope if scope else "user"

    logger.info(
        "Prepared CompositeBackend factory: %d eager route(s), %d deferred route(s)",
        len(eager_routes),
        len(store_route_prefixes),
    )

    # --- Factory: called per-request with ToolRuntime ---
    def factory(runtime: Any) -> Any:
        """Build a CompositeBackend when invoked by create_deep_agent with ToolRuntime.

        We instantiate StateBackend/StoreBackend here (rather than returning
        bare classes) because CompositeBackend needs composed *instances* —
        this factory IS the protocol-compliant callable that create_deep_agent expects.
        """
        from deepagents.backends.composite import CompositeBackend
        from deepagents.backends.state import StateBackend

        state_backend = StateBackend(runtime)

        routes: dict[str, Any] = dict(eager_routes)

        if store_route_prefixes:
            try:
                from deepagents.backends.store import StoreBackend

                ns = _STORE_NAMESPACE_FACTORIES.get(
                    store_scope or "user", _safe_namespace_user
                )
                store_backend = StoreBackend(runtime, namespace=ns)
                for prefix in store_route_prefixes:
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

        default_backend = routes.pop("/", state_backend)
        return CompositeBackend(default=default_backend, routes=routes)

    return factory


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
