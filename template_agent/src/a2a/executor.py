"""A2A AgentExecutor implementation bridging A2A protocol to LangGraph agent."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import ContentTypeNotSupportedError, Part, Task, TaskState, TaskStatus
from google.protobuf.timestamp_pb2 import Timestamp

from template_agent.src.core.agent import get_template_agent
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

ACCESS_TOKEN_STATE_KEY = "access_token"


class TemplateAgentExecutor(AgentExecutor):
    """Bridges incoming A2A requests to the existing LangGraph agent.

    Extracts the user message from the A2A RequestContext, runs the
    LangGraph agent, and publishes Task/status/artifact events back
    through the A2A EventQueue via TaskUpdater.

    Tracks running asyncio tasks so that cancel() can actually stop
    in-progress LLM calls rather than just emitting a status event.
    """

    def __init__(self, supported_output_modes: list[str] | None = None) -> None:
        """Initialize with optional list of supported output content modes."""
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._supported_output_modes = supported_output_modes or []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run the agent for an incoming A2A request and stream results."""
        if self._is_output_mode_incompatible(context):
            accepted = (
                list(context.configuration.accepted_output_modes)
                if context.configuration
                else []
            )
            raise ContentTypeNotSupportedError(
                message=f"None of the accepted output modes {accepted} are supported. "
                f"Supported: {self._supported_output_modes}"
            )

        task_id = context.task_id

        current = asyncio.current_task()
        if current is not None:
            self._running_tasks[task_id] = current

        try:
            await self._execute_inner(context, event_queue)
        except asyncio.CancelledError:
            logger.info(f"A2A task {task_id} was cancelled via asyncio")
            raise
        finally:
            self._running_tasks.pop(task_id, None)

    async def _execute_inner(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if context.current_task:
            task = context.current_task
        else:
            ts = Timestamp()
            ts.FromDatetime(datetime.now(timezone.utc))
            task = Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_SUBMITTED,
                    timestamp=ts,
                ),
            )
            if context.message:
                task.history.append(context.message)

        await event_queue.enqueue_event(task)

        user_input = context.get_user_input()
        if not user_input:
            msg = updater.new_agent_message([Part(text="No text content in message")])
            await updater.failed(message=msg)
            return

        access_token = context.call_context.state.get(ACCESS_TOKEN_STATE_KEY)
        correlation_id = context.call_context.state.get("correlation_id")

        logger.info(
            f"A2A execute: task_id={context.task_id}, "
            f"input_length={len(user_input)}, "
            f"has_token={'yes' if access_token else 'no'}"
        )

        msg = updater.new_agent_message([Part(text="Processing your request...")])
        await updater.start_work(message=msg)

        try:
            await self._run_agent_streaming(
                user_input, access_token, updater, context.context_id, correlation_id
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"A2A agent execution failed: {e}", exc_info=True)
            msg = updater.new_agent_message([Part(text=f"Agent execution failed: {e}")])
            await updater.failed(message=msg)
            return

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an in-progress task by cancelling its asyncio coroutine."""
        task_id = context.task_id
        running = self._running_tasks.pop(task_id, None)
        if running and not running.done():
            running.cancel()
            logger.info(f"Cancelled running asyncio task for A2A task {task_id}")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

    def _is_output_mode_incompatible(self, context: RequestContext) -> bool:
        """Check if client-accepted output modes are incompatible with this agent.

        Returns True (incompatible) only when the client explicitly requests
        modes and none of them overlap with the agent's supported set.
        An empty accepted list means "no preference" and is always compatible.
        """
        accepted = (
            list(context.configuration.accepted_output_modes)
            if context.configuration
            else []
        )
        if not accepted:
            return False
        compatible = bool(set(accepted) & set(self._supported_output_modes))
        if not compatible:
            logger.warning(
                "Unsupported output mode. Accepted: %s, Supported: %s",
                accepted,
                self._supported_output_modes,
            )
        return not compatible

    async def _run_agent_streaming(
        self,
        user_input: str,
        access_token: str | None,
        updater: TaskUpdater,
        context_id: str | None,
        correlation_id: str | None = None,
    ) -> None:
        """Run the LangGraph agent and stream token chunks via TaskUpdater."""
        from langchain_core.messages import AIMessageChunk, HumanMessage
        from langchain_core.runnables import RunnableConfig

        thread_id = context_id or str(uuid4())
        run_id = uuid4()

        async with get_template_agent(
            sso_token=access_token,
            enable_checkpointing=True,
            a2a_context_id=thread_id,
            correlation_id=correlation_id,
        ) as agent:
            config = RunnableConfig(
                configurable={"thread_id": thread_id, "run_id": str(run_id)},
                run_id=run_id,
            )

            input_messages = {"messages": [HumanMessage(content=user_input)]}
            artifact_id = str(uuid4())
            collected_text = ""
            first_chunk = True

            async for event in agent.astream_events(
                input=input_messages, config=config, version="v2"
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        token = chunk.content if isinstance(chunk.content, str) else ""
                        if token:
                            collected_text += token
                            await updater.add_artifact(
                                [Part(text=token)],
                                artifact_id=artifact_id,
                                name="response" if first_chunk else None,
                                append=not first_chunk,
                                last_chunk=False,
                            )
                            first_chunk = False

            if not first_chunk:
                await updater.add_artifact(
                    [Part(text="")],
                    artifact_id=artifact_id,
                    append=True,
                    last_chunk=True,
                )
            else:
                await updater.add_artifact(
                    [Part(text="No response generated.")],
                    artifact_id=artifact_id,
                    name="response",
                    append=False,
                    last_chunk=True,
                )
