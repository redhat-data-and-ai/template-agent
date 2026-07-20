"""Headless agent entry point — runs as a background worker with event triggers.

Usage:
    python -m deep_agent.headless
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from deep_agent.src.triggers.config import AgentMode, HeadlessConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "agent"
    / "runtime"
    / "agent.yaml"
)
_HEADLESS_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "agent" / "HEADLESS_PROMPT.md"
)


def _load_headless_config() -> HeadlessConfig:
    """Load and validate headless configuration from agent.yaml."""
    if not _CONFIG_PATH.is_file():
        logger.error("Config not found: %s", _CONFIG_PATH)
        sys.exit(1)

    raw = yaml.safe_load(_CONFIG_PATH.read_text()) or {}

    config = HeadlessConfig(
        mode=AgentMode.HEADLESS,
        triggers=raw.get("triggers", {}),
        output_sinks=raw.get("output_sinks", []),
        drain_timeout=raw.get("drain_timeout", 30.0),
        health_check=raw.get("health_check", {}),
    )

    return config


async def _build_headless_graph() -> Any:
    """Build a dedicated graph from HEADLESS_PROMPT.md.

    Uses a simpler prompt than the orchestrator — no user interaction,
    no TODO lists, just task processing.
    """
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.agent.config.model import parse_model_config
    from deep_agent.src.agent.config.parser import (
        inject_runtime_values,
        parse_frontmatter,
    )
    from deep_agent.src.cache.model_cache import get_or_create_model_from_spec

    headless_cfg = parse_frontmatter(_HEADLESS_PROMPT_PATH)
    model_raw = headless_cfg.get("model", "gemini-2.5-pro")
    system_prompt = inject_runtime_values(headless_cfg.get("body", ""))
    tool_names = headless_cfg.get("tools", [])
    skill_names = headless_cfg.get("skills", [])
    if skill_names:
        from deep_agent.src.agent.config.resolver import resolve_skill_paths

        available_skills = agent_config._scan_available_skills()
        skill_paths = resolve_skill_paths(
            skill_names, available_skills, agent_name="headless-worker"
        )
    else:
        skill_paths = []

    orch_spec = parse_model_config(model_raw)
    model = get_or_create_model_from_spec(orch_spec)

    from deep_agent.aegra.mcp import get_mcp_tools

    mcp_tools = await get_mcp_tools(sso_token=None, server_names=None)

    from deep_agent.src.triggers.tools import get_builtin_tools

    all_tools = list(mcp_tools) + get_builtin_tools()
    tools = agent_config.resolve_tools(
        tool_names, all_tools, agent_name="headless-worker"
    )

    from deep_agent.src.infrastructure.backend import get_configured_backend
    from deep_agent.src.infrastructure.middleware import (
        build_middleware_list,
        resolve_memory_param,
    )

    middleware_overrides = headless_cfg.get("middleware")
    resolved_mw = agent_config.resolve_agent_middleware(
        orch_spec.name, middleware_overrides
    )
    backend = get_configured_backend()
    middleware = build_middleware_list(resolved_mw, model=model, backend=backend)
    memory = resolve_memory_param(resolved_mw)

    from deepagents import create_deep_agent

    compiled = create_deep_agent(
        name="headless-worker",
        model=model,
        system_prompt=system_prompt,
        skills=skill_paths or None,
        tools=tools,
        backend=backend,
        middleware=middleware,
        memory=memory,
    )

    logger.info(
        "Headless graph built: %d tool(s), prompt=%s",
        len(tools),
        _HEADLESS_PROMPT_PATH.name,
    )
    return compiled


async def main() -> None:
    """Run the headless agent worker."""
    logger.info("Starting headless agent worker")

    config = _load_headless_config()

    from deep_agent.aegra.startup import run_startup

    startup_results = await run_startup()
    logger.info("Startup: %s", startup_results)

    logger.info("Building headless agent graph")
    compiled_graph = await _build_headless_graph()

    from deep_agent.src.settings import settings
    from deep_agent.src.triggers.middleware import EventTriggerMiddleware

    middleware = EventTriggerMiddleware(
        config=config,
        graph=compiled_graph,
        redis_url=settings.REDIS_URL,
    )

    await middleware.start()

    health_server: asyncio.Server | None = None
    if config.health_check.enabled:
        from deep_agent.src.triggers.health import start_health_server

        health_server = await start_health_server(
            config.health_check.host,
            config.health_check.port,
            middleware,
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Headless agent worker running — waiting for events")
    await stop_event.wait()

    logger.info("Shutting down headless agent worker")
    if health_server is not None:
        health_server.close()
        await health_server.wait_closed()
    await middleware.stop()

    try:
        from deep_agent.aegra.shutdown import run_shutdown

        await run_shutdown()
    except Exception:
        logger.debug("Cleanup completed with warnings", exc_info=True)

    logger.info("Headless agent worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
