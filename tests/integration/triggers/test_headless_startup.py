"""Integration tests for headless process startup and configuration loading.

These tests exercise ``deep_agent.headless`` — the headless worker entry
point — by testing config loading/validation and the ``main()`` lifecycle
with external dependencies (DB, model provider) mocked out.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.triggers.config import (
    AgentMode,
    HeadlessConfig,
    HealthCheckConfig,
    OutputSinkConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Config loading tests
# ------------------------------------------------------------------


class TestLoadHeadlessConfig:
    """Test _load_headless_config() validation logic."""

    def test_headless_mode_config_loads_successfully(self, tmp_path: Path):
        """A YAML with mode=headless produces a valid HeadlessConfig."""
        import yaml

        config_data = {
            "mode": "headless",
            "triggers": {
                "webhook": {"enabled": True, "port": 9999, "path": "/trigger"},
                "queue": {"enabled": False},
                "cron": {"enabled": False},
            },
            "output_sinks": [],
        }
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("deep_agent.headless._CONFIG_PATH", config_file):
            from deep_agent.headless import _load_headless_config

            config = _load_headless_config()

        assert config.mode == AgentMode.HEADLESS
        assert config.triggers.webhook.enabled is True
        assert config.triggers.webhook.port == 9999

    def test_mode_is_always_headless_regardless_of_yaml(self, tmp_path: Path):
        """_load_headless_config always sets mode=HEADLESS, ignoring the YAML value."""
        import yaml

        config_data = {"mode": "server", "triggers": {}, "output_sinks": []}
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("deep_agent.headless._CONFIG_PATH", config_file):
            from deep_agent.headless import _load_headless_config

            config = _load_headless_config()

        assert config.mode == AgentMode.HEADLESS

    def test_missing_config_file_causes_exit(self):
        """When the config file does not exist, sys.exit(1) is called."""
        bogus = Path("/nonexistent/agent.yaml")

        with (
            patch("deep_agent.headless._CONFIG_PATH", bogus),
            pytest.raises(SystemExit) as exc_info,
        ):
            from deep_agent.headless import _load_headless_config

            _load_headless_config()

        assert exc_info.value.code == 1


# ------------------------------------------------------------------
# main() lifecycle tests
# ------------------------------------------------------------------


class TestHeadlessMainLifecycle:
    """Test main() wires up prerequisites, graph, and middleware correctly.

    Since ``main()`` uses late/local imports from ``deep_agent.aegra``
    and ``deep_agent.src``, patches target the *source* modules so the
    imports inside ``main()`` pick up the mocks.
    """

    async def test_main_starts_middleware_and_waits_for_signal(self):
        """main() starts EventTriggerMiddleware and blocks on stop_event."""
        headless_config = HeadlessConfig(
            mode=AgentMode.HEADLESS,
            triggers=TriggerConfig(
                webhook=WebhookTriggerConfig(enabled=True, port=0),
            ),
            output_sinks=[OutputSinkConfig(type="stdout")],
            drain_timeout=2.0,
            health_check=HealthCheckConfig(enabled=False),
        )

        mock_graph = MagicMock()
        mock_mw_instance = AsyncMock()

        with (
            patch(
                "deep_agent.headless._load_headless_config",
                return_value=headless_config,
            ),
            patch(
                "deep_agent.aegra.startup.run_startup",
                new_callable=AsyncMock,
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.graph.agent",
                new_callable=AsyncMock,
                return_value=mock_graph,
            ),
            patch(
                "deep_agent.src.triggers.middleware.EventTriggerMiddleware",
            ) as MockMW,
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(REDIS_URL="redis://localhost:6379/0"),
            ),
        ):
            MockMW.return_value = mock_mw_instance

            from deep_agent.headless import main

            task = asyncio.create_task(main())

            # Give main() time to reach the stop_event.wait().
            await asyncio.sleep(0.3)

            # Verify middleware.start() was called.
            mock_mw_instance.start.assert_awaited_once()

            # Simulate SIGTERM by cancelling the task (the signal handler
            # would set stop_event, but cancellation is simpler in tests).
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # EventTriggerMiddleware was constructed with the right config.
        MockMW.assert_called_once_with(
            config=headless_config,
            graph=mock_graph,
            redis_url="redis://localhost:6379/0",
        )

    async def test_main_calls_run_startup(self):
        """main() calls run_startup before building the graph."""
        headless_config = HeadlessConfig(
            mode=AgentMode.HEADLESS,
            triggers=TriggerConfig(),
            output_sinks=[],
            drain_timeout=1.0,
            health_check=HealthCheckConfig(enabled=False),
        )

        mock_startup = AsyncMock(return_value={"status": "ok"})
        mock_mw = AsyncMock()

        with (
            patch(
                "deep_agent.headless._load_headless_config",
                return_value=headless_config,
            ),
            patch(
                "deep_agent.aegra.startup.run_startup",
                mock_startup,
            ),
            patch(
                "deep_agent.aegra.graph.agent",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.triggers.middleware.EventTriggerMiddleware",
                return_value=mock_mw,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(REDIS_URL="redis://localhost:6379/0"),
            ),
        ):
            from deep_agent.headless import main

            task = asyncio.create_task(main())
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_startup.assert_awaited_once()

    async def test_main_invokes_graph_factory_with_runtime(self):
        """main() calls ``agent(runtime)`` to build the compiled graph."""
        headless_config = HeadlessConfig(
            mode=AgentMode.HEADLESS,
            triggers=TriggerConfig(),
            output_sinks=[],
            drain_timeout=1.0,
            health_check=HealthCheckConfig(enabled=False),
        )

        mock_agent_factory = AsyncMock(return_value=MagicMock())
        mock_mw = AsyncMock()

        with (
            patch(
                "deep_agent.headless._load_headless_config",
                return_value=headless_config,
            ),
            patch(
                "deep_agent.aegra.startup.run_startup",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "deep_agent.aegra.graph.agent",
                mock_agent_factory,
            ),
            patch(
                "deep_agent.src.triggers.middleware.EventTriggerMiddleware",
                return_value=mock_mw,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(REDIS_URL="redis://localhost:6379/0"),
            ),
        ):
            from deep_agent.headless import main

            task = asyncio.create_task(main())
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_agent_factory.assert_awaited_once()
        # The argument should be a HeadlessRuntime instance.
        runtime_arg = mock_agent_factory.call_args[0][0]
        assert hasattr(runtime_arg, "user")
        assert runtime_arg.user.identity == "headless-worker"
