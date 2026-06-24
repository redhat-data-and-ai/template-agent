"""OPA-backed trajectory policy middleware for deep agents.

Evaluates the agent's message trajectory against Rego policies served by
Open Policy Agent before each LLM call and tool invocation.

Supports per-user policy settings that are fetched from the database and
passed to OPA at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

POLICY_DENIAL_MESSAGE = (
    "I'm unable to complete this request because it isn't allowed by our "
    "compliance policies. Please try a different approach or contact your "
    "administrator if you need access."
)

# Global instance for cache invalidation from API
_middleware_instance: "RegoTrajectoryMiddleware | None" = None


def get_middleware_instance() -> "RegoTrajectoryMiddleware | None":
    """Get the global middleware instance for cache invalidation.

    Returns:
        The middleware instance or None if not yet created
    """
    return _middleware_instance


class RegoTrajectoryMiddleware(AgentMiddleware):
    """Enforce trajectory rules via OPA with per-user settings support."""

    name = "rego_trajectory_policy"

    def __init__(
        self,
        opa_url: str = "http://localhost:8181/v1/data/agent/authz",
        *,
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self.opa_url = opa_url
        self.timeout = timeout
        # Simple in-memory cache: user_id -> settings dict
        self._settings_cache: dict[str, dict[str, Any]] = {}

        # Register as global instance for cache invalidation
        global _middleware_instance
        _middleware_instance = self

    def _parse_trajectory(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert message history into a structural array for Rego."""
        trajectory: list[dict[str, Any]] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "ai" and getattr(msg, "tool_calls", None):
                trajectory.append(
                    {
                        "type": "agent_action",
                        "tools": [
                            {"name": tc["name"], "args": tc.get("args", {})}
                            for tc in msg.tool_calls
                        ],
                    }
                )
            elif msg_type == "tool":
                trajectory.append(
                    {
                        "type": "tool_response",
                        "name": getattr(msg, "name", None),
                        "status": "completed",
                    }
                )
        return trajectory

    async def _get_user_settings(self, user_id: str) -> dict[str, Any] | None:
        """Fetch user policy settings from database (with caching).

        Args:
            user_id: User identifier

        Returns:
            User settings dict or None if no custom settings exist
        """
        # Check cache first
        if user_id in self._settings_cache:
            logger.debug(f"Policy settings cache HIT for user {user_id}")
            return self._settings_cache[user_id]

        logger.debug(f"Policy settings cache MISS for user {user_id}, fetching from DB")

        try:
            from deep_agent.src.policy.repository import PolicySettingsRepository
            from deep_agent.src.settings import settings as app_settings

            repo = PolicySettingsRepository(app_settings.database_uri)
            user_settings = await repo.get_user_settings(user_id)

            if user_settings:
                # Cache the settings
                self._settings_cache[user_id] = user_settings.values
                logger.info(
                    f"Loaded custom policy settings for user {user_id}: {user_settings.values}"
                )
                return user_settings.values
            else:
                logger.debug(f"No custom policy settings for user {user_id}, will use OPA defaults")
                return None

        except Exception as exc:
            logger.warning(f"Failed to load policy settings for user {user_id}: {exc}")
            return None

    def _get_user_id(self, runtime: Any) -> str:
        """Extract user ID from runtime context.

        Args:
            runtime: Runtime context (Aegra ServerRuntime or similar)

        Returns:
            User identity string or "anonymous"
        """
        user = getattr(runtime, "user", None)
        if user:
            return getattr(user, "identity", "anonymous")
        return "anonymous"

    def _evaluate_policy(
        self,
        trajectory: list[dict[str, Any]],
        current_intent: dict,
        user_settings: dict[str, Any] | None = None,
    ) -> bool:
        """Query OPA with user settings in the input.

        Args:
            trajectory: Agent's message history
            current_intent: Current action being evaluated
            user_settings: Per-user policy settings (or None for defaults)

        Returns:
            True if allowed, False if denied
        """
        payload = {
            "input": {
                "trajectory": trajectory,
                "current_intent": current_intent,
                "user_settings": user_settings,  # Pass user settings to OPA
            }
        }

        logger.debug(
            f"OPA eval: intent={current_intent.get('action')}, "
            f"trajectory_len={len(trajectory)}, "
            f"has_user_settings={user_settings is not None}"
        )

        try:
            response = httpx.post(
                self.opa_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json().get("result", {})
            allowed = result.get("allow", False)

            if not allowed:
                # Log denial reasons if available
                denial_reasons = result.get("denial_reasons", [])
                logger.warning(
                    f"OPA denied: intent={current_intent.get('action')}, "
                    f"trajectory_steps={len(trajectory)}, "
                    f"reasons={denial_reasons}"
                )

            return bool(allowed)

        except Exception as exc:
            logger.error(f"OPA policy check failed (fail-closed): {exc}")
            return False

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Evaluate trajectory before the LLM is allowed to respond."""
        trajectory = self._parse_trajectory(state.get("messages", []))

        # Extract user ID from runtime context
        user_id = self._get_user_id(runtime)
        logger.debug(f"Evaluating policy for user: {user_id}")

        # Fetch user-specific settings from database
        user_settings = await self._get_user_settings(user_id)

        # Evaluate policy with user settings
        if not self._evaluate_policy(
            trajectory, {"action": "llm_request"}, user_settings
        ):
            logger.warning(f"Policy denied LLM request for user {user_id}")
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=POLICY_DENIAL_MESSAGE)],
            }

        logger.debug(f"Policy allowed LLM request for user {user_id}")
        return None

    async def awrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Evaluate trajectory before executing a tool."""
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_call = request.tool_call
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
        tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

        current_intent = {
            "action": "tool_call",
            "name": tool_name,
            "args": tool_args,
        }

        # Get user ID from runtime
        runtime = getattr(request, "runtime", None)
        user_id = self._get_user_id(runtime) if runtime else "anonymous"

        # Fetch user settings
        user_settings = await self._get_user_settings(user_id)

        # Evaluate policy
        if not self._evaluate_policy(trajectory, current_intent, user_settings):
            logger.warning(f"Policy denied tool call {tool_name} for user {user_id}")
            return Command[str](
                update={
                    "messages": [
                        ToolMessage(
                            content=POLICY_DENIAL_MESSAGE,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            status="error",
                        ),
                        AIMessage(content=POLICY_DENIAL_MESSAGE),
                    ],
                },
                goto="end",
            )

        logger.debug(f"Policy allowed tool call {tool_name} for user {user_id}")
        return await handler(request)

    def invalidate_cache(self, user_id: str | None = None) -> None:
        """Invalidate settings cache for a user or all users.

        Call this after updating user settings via API.

        Args:
            user_id: Specific user to invalidate, or None for all users
        """
        if user_id:
            self._settings_cache.pop(user_id, None)
            logger.info(f"Invalidated policy settings cache for user {user_id}")
        else:
            self._settings_cache.clear()
            logger.info("Invalidated policy settings cache for all users")
