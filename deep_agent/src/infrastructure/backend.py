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

from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
_PASSTHROUGH_VARS = ("HOME", "USER", "LANG", "LC_ALL", "TZ", "TERM")


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_backend: LocalShellBackend | None = None


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
    """Return the backend configured by filesystem.yaml.

    Reads the backend type from config and builds the appropriate backend:
    - state: StateBackend (thread-scoped scratch, recommended for production)
    - composite: CompositeBackend (routes paths to different backends)
    - store: StoreBackend (cross-thread persistent via LangGraph Store)
    - local_shell: LocalShellBackend (local dev only — NOT for deployed agents)

    Falls back to StateBackend if config is missing or invalid.
    """
    from deep_agent.src.agent.config.filesystem import load_filesystem_config

    config_path = agent_config.base_dir / "filesystem.yaml"
    fs_config = load_filesystem_config(config_path)

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

        namespace_factories = {
            "user": lambda rt: (
                rt.server_info.assistant_id,
                rt.server_info.user.identity,
            ),
            "assistant": lambda rt: (rt.server_info.assistant_id,),
            "org": lambda rt: (rt.context.org_id,),
        }

        namespace = namespace_factories.get(scope_name)
        if namespace is None:
            logger.warning("Unknown store scope '%s', using 'user'", scope_name)
            namespace = namespace_factories["user"]

        logger.info("Using StoreBackend (scope=%s)", scope_name)
        return StoreBackend(namespace=namespace)
    except ImportError:
        logger.warning("StoreBackend not available, falling back to StateBackend")
        return _build_state_backend()


def _build_composite_backend(fs_config: Any) -> Any:
    """Build a CompositeBackend from route config.

    Production-recommended pattern: StateBackend as default (scratch)
    with StoreBackend routes for persistent paths like /memories/.
    """
    try:
        from deepagents.backends.composite import CompositeBackend
        from deepagents.backends.state import StateBackend

        state_backend = StateBackend()
        store_backend = None

        backend_map: dict[str, Any] = {
            "state": state_backend,
        }

        if any(v == "store" for v in fs_config.backend.routes.values()):
            store_backend = _build_store_backend(fs_config)
            backend_map["store"] = store_backend

        if any(v == "local_shell" for v in fs_config.backend.routes.values()):
            logger.warning(
                "local_shell in composite routes — not recommended for production"
            )
            backend_map["local_shell"] = get_backend(
                timeout=fs_config.backend.local_shell.timeout,
                max_output_bytes=fs_config.backend.local_shell.max_output_bytes,
            )

        routes: dict[str, Any] = {}
        for path_prefix, backend_name in fs_config.backend.routes.items():
            if backend_name in backend_map:
                routes[path_prefix] = backend_map[backend_name]
            else:
                logger.warning(
                    "Unknown backend '%s' in route for '%s'", backend_name, path_prefix
                )

        default_backend = routes.pop("/", state_backend)
        logger.info("Using CompositeBackend with %d route(s)", len(routes))
        return CompositeBackend(default=default_backend, routes=routes)
    except ImportError:
        logger.warning("CompositeBackend not available, falling back to StateBackend")
        return _build_state_backend()
