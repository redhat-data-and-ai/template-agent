"""Agent Manager for the template agent system.

This module provides the AgentManager class that orchestrates agent operations,
handles streaming responses, and manages the conversion between LangGraph events
and simplified streaming.
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from template_agent.src.core.agent import get_template_agent
from template_agent.src.core.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from template_agent.src.core.deep_research.nodes._cache import (
    format_cached_findings_for_triage,
    format_conversation_for_prompt,
    load_conversation_history,
    save_conversation_turn,
)
from template_agent.src.core.storage import register_thread
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger
from template_agent.utils.tracing import langfuse_handler

app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _content_to_str(content: str | list[Any] | None) -> str:
    """Convert LLM response content to string (handles multi-modal list format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return convert_message_content_to_string(content)


class AgentManager:
    """Manager class for handling agent operations and streaming responses.

    This class provides a simplified interface for agent interactions while
    preserving all enterprise features like authentication, tracing, and
    error handling from the original implementation. Supports both standard
    agent mode and deep research mode.
    """

    def __init__(
        self,
        redhat_sso_token: str | None = None,
        root_tracer: Any = None,
    ):
        """Initialize the AgentManager.

        Args:
            redhat_sso_token: Optional SSO token for enterprise authentication.
            root_tracer: Optional AgentTracer for Langfuse tracing.
        """
        self.redhat_sso_token = redhat_sso_token
        self.root_tracer = root_tracer
        self._agent: Pregel | None = None
        self._current_tool_call_id: str | None = None

    def _should_use_deep_research(self, request: StreamRequest) -> bool:
        """Determine whether to route to deep research pipeline."""
        if not settings.DEEP_RESEARCH_ENABLED:
            return False
        if getattr(request, "deep_research_enabled", False):
            return True
        if getattr(request, "deep_research_resume", False):
            return True
        return False

    async def _classify_follow_up(
        self,
        query: str,
        findings_text: str,
        conversation_text: str,
        model_name: str | None = None,
    ) -> str:
        """Classify whether a follow-up can be answered from context.

        Returns ``"answer_directly"`` or ``"needs_research"``.
        Uses a lightweight model (gemini-2.5-flash) for speed.
        """
        from langchain_core.messages import HumanMessage as _HM
        from langchain_core.messages import SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        system_prompt = (
            "You are a routing classifier. Given a user's follow-up question, "
            "previously researched findings, and the conversation history, decide "
            "whether the question can be answered directly from the available "
            "context or requires new research.\n\n"
            "Respond with EXACTLY one of:\n"
            "- answer_directly\n"
            "- needs_research\n\n"
            "## Decision Rules (follow strictly)\n\n"
            "DEFAULT to 'answer_directly'. Only choose 'needs_research' when the "
            "question asks about a topic, entity, or time period that is COMPLETELY "
            "ABSENT from the findings below.\n\n"
            "Choose 'answer_directly' when:\n"
            "- The question asks about facts, dates, people, or details that "
            "appear anywhere in the findings (even if not the exact phrasing).\n"
            "- The question can be answered by combining or filtering information "
            "already present in the findings (e.g., 'who was X at time Y').\n"
            "- The question rephrases, narrows, or drills into a topic already "
            "covered by the findings.\n"
            "- The question asks for a comparison, summary, or opinion that can "
            "be derived from existing findings.\n\n"
            "Choose 'needs_research' ONLY when:\n"
            "- The question asks about a completely new topic not mentioned at "
            "all in the findings.\n"
            "- The question explicitly requests fresh, real-time, or breaking-news "
            "data that the findings could not contain.\n\n"
            "## Previous Research Findings\n"
            f"{findings_text}\n\n"
            "## Conversation History\n"
            f"{conversation_text or '(no prior conversation)'}"
        )

        classifier_model = model_name or "gemini-2.5-flash"
        llm = ChatGoogleGenerativeAI(model=classifier_model, temperature=0.0)

        try:
            response = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=system_prompt), _HM(content=query)]),
                timeout=settings.DEEP_RESEARCH_LLM_CALL_TIMEOUT_SECONDS,
            )
            decision = _content_to_str(response.content).strip().lower()
            if decision not in ("answer_directly", "needs_research"):
                app_logger.warning(
                    "Follow-up classifier returned unexpected value '%s', defaulting to needs_research",
                    decision,
                )
                return "needs_research"
            app_logger.info("Follow-up classifier decision: %s", decision)
            return decision
        except Exception as exc:
            app_logger.warning(
                "Follow-up classification failed (%s), defaulting to needs_research",
                exc,
            )
            return "needs_research"

    async def _stream_follow_up_answer(
        self,
        request: StreamRequest,
        findings_text: str,
        conversation_text: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a direct answer synthesized from cached findings and history.

        Yields standard chat events (``message`` and ``token``) so the UI
        renders this as a normal response rather than a deep-research run.
        """
        from langchain_core.messages import HumanMessage as _HM
        from langchain_core.messages import SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        thread_id = request.thread_id or str(uuid4())
        run_id = str(uuid4())
        session_id = request.session_id or thread_id

        model_name = (
            getattr(request, "deep_research_model", None)
            or settings.DEEP_RESEARCH_DEFAULT_MODEL
        )
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)

        system_prompt = (
            "You are a knowledgeable assistant. Answer the user's question "
            "using ONLY the research findings and conversation history below. "
            "Be thorough but concise. If the findings don't fully cover the "
            "question, say so clearly.\n\n"
            "## Research Findings\n"
            f"{findings_text}\n\n"
            "## Conversation History\n"
            f"{conversation_text or '(first question in thread)'}"
        )

        collected_answer = ""
        try:
            async for chunk in llm.astream(
                [SystemMessage(content=system_prompt), _HM(content=request.message)]
            ):
                token_text = _content_to_str(chunk.content)
                if token_text:
                    collected_answer += token_text
                    yield {
                        "type": "token",
                        "content": token_text,
                    }

            yield {
                "type": "message",
                "content": {
                    "type": "ai",
                    "content": collected_answer,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "session_id": session_id,
                },
            }

            save_conversation_turn(thread_id, request.message, collected_answer)

        except Exception as exc:
            app_logger.error("Follow-up direct answer failed: %s", exc, exc_info=True)
            yield {
                "type": "error",
                "content": {
                    "message": f"Failed to generate follow-up answer: {exc}",
                    "recoverable": True,
                    "error_type": "followup_error",
                },
            }

    async def _stream_deep_research(
        self, request: StreamRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream deep research by reading _pending_events from graph updates."""
        thread_id = request.thread_id or str(uuid4())
        run_id = str(uuid4())
        session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        if settings.USE_INMEMORY_SAVER:
            register_thread(effective_user_id, thread_id)

        try:
            from template_agent.src.core.deep_research.streaming import (
                create_initial_state,
                get_deep_research_agent,
            )
            from template_agent.utils.tracing import TokenUsageTracker

            model_name = (
                getattr(request, "deep_research_model", None)
                or settings.DEEP_RESEARCH_DEFAULT_MODEL
            )
            max_mode = getattr(request, "deep_research_max_mode", False)
            max_subqueries = getattr(request, "deep_research_max_subqueries", None)
            token_tracker = TokenUsageTracker(model_name=model_name or "")

            require_approval = settings.DEEP_RESEARCH_REQUIRE_PLAN_APPROVAL
            user_approved = bool(getattr(request, "deep_research_plan_approved", False))
            plan_approved = (not require_approval) or user_approved
            user_plan = getattr(request, "deep_research_plan", None)
            plan_override = user_plan if isinstance(user_plan, list) else None
            is_resume = bool(plan_override and plan_approved)

            async with get_deep_research_agent(
                model_name=model_name,
                user_id=request.user_id,
                max_subqueries_override=max_subqueries,
                max_mode=max_mode,
                token_tracker=token_tracker,
                root_tracer=self.root_tracer,
            ) as dr_agent:
                ctx = dr_agent.ctx

                cached_findings_text = ""
                cached_triage_text = ""
                cached_raw: dict = {}
                try:
                    from template_agent.src.core.deep_research.nodes._cache import (
                        format_cached_findings_for_prompt,
                        load_findings_in_memory,
                    )

                    cached_raw = load_findings_in_memory(thread_id)
                    if cached_raw:
                        cached_findings_text = format_cached_findings_for_prompt(
                            cached_raw
                        )
                        app_logger.info(
                            "Loaded %d cached findings for thread %s",
                            len(cached_raw),
                            thread_id,
                        )
                except Exception as e:
                    app_logger.warning("Failed to load cached findings: %s", e)

                if cached_raw and not is_resume:
                    from template_agent.src.core.deep_research.streaming import (
                        select_relevant_findings,
                    )

                    cached_triage_text = await select_relevant_findings(
                        ctx.base_model,
                        request.message,
                        cached_raw,
                        max_chars=30000,
                    )
                    if not cached_triage_text:
                        cached_triage_text = format_cached_findings_for_triage(
                            cached_raw
                        )

                if cached_triage_text and not is_resume:
                    conversation_history = load_conversation_history(thread_id)
                    conversation_text = format_conversation_for_prompt(
                        conversation_history
                    )

                    decision = await self._classify_follow_up(
                        request.message, cached_triage_text, conversation_text
                    )
                    if decision == "answer_directly":
                        async for event in self._stream_follow_up_answer(
                            request, cached_triage_text, conversation_text
                        ):
                            yield event
                        return

                initial = await create_initial_state(
                    query=request.message,
                    thread_id=thread_id,
                    plan_override=plan_override,
                    plan_approved=plan_approved,
                    skip_to_research=is_resume,
                    user_id=request.user_id,
                    cached_findings_text=cached_findings_text,
                )

                from template_agent.src.core.deep_research.events import (
                    emit_token_usage_update,
                )
                from template_agent.src.core.deep_research.streaming import (
                    _should_pause_for_plan_approval,
                )

                dr_final_answer: str | None = None
                run_config = {
                    "run_id": run_id,
                    "configurable": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": effective_user_id,
                    },
                }
                run_config_metadata = {"user_id": effective_user_id}

                async for event in dr_agent._run_graph_with_events(
                    initial,
                    thread_id,
                    run_config_metadata,
                    run_config=run_config,
                ):
                    if event.get("type") == "message":
                        content = event.get("content", {})
                        if isinstance(content, dict) and content.get("type") == "ai":
                            dr_final_answer = content.get("content", "")
                            content["run_id"] = run_id
                            content["thread_id"] = thread_id
                            content["session_id"] = session_id
                        yield event
                    else:
                        yield event
                        if _should_pause_for_plan_approval(
                            event, require_approval, plan_approved
                        ):
                            break

                if ctx.token_tracker:
                    total = ctx.token_tracker.get_total()
                    if total.llm_calls > 0:
                        yield emit_token_usage_update(
                            input_tokens=total.input_tokens,
                            output_tokens=total.output_tokens,
                            total_tokens=total.total_tokens,
                            llm_calls=total.llm_calls,
                            estimated_cost_usd=total.estimated_cost_usd,
                        )

            if dr_final_answer:
                save_conversation_turn(thread_id, request.message, dr_final_answer)

            app_logger.info("Deep research completed for thread %s", thread_id)

        except ImportError as e:
            app_logger.error(f"Deep research module not available: {e}")
            from template_agent.src.core.deep_research.events import (
                DeepResearchEventType as _DREventType,
            )
            from template_agent.src.core.deep_research.events import (
                emit_event as _emit_event,
            )

            yield _emit_event(
                _DREventType.ERROR,
                f"Deep research not available: {e}",
                ui_visible=True,
            )
        except Exception as e:
            app_logger.error(f"Deep research error: {e}", exc_info=True)
            try:
                from template_agent.src.core.deep_research.events import (
                    DeepResearchEventType as _DREventType,
                )
                from template_agent.src.core.deep_research.events import (
                    emit_event as _emit_event,
                )

                yield _emit_event(
                    _DREventType.ERROR,
                    f"Deep research encountered an error: {e}",
                    ui_visible=True,
                )
            except Exception:
                pass
            yield {
                "type": "message",
                "content": {
                    "type": "ai",
                    "content": "Deep research encountered an error. Please try again.",
                    "run_id": "",
                    "thread_id": "",
                    "session_id": "",
                },
            }

    async def stream_response(
        self, request: StreamRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with simplified event structure.

        Routes to deep research pipeline when enabled, otherwise uses
        standard agent mode.

        Args:
            request: The streaming request containing user input and configuration.

        Yields:
            Simplified event dictionaries with 'type' and 'content' fields.
        """
        if self._should_use_deep_research(request):
            async for event in self._stream_deep_research(request):
                yield event
            return

        # Standard agent mode
        async with get_template_agent(
            self.redhat_sso_token, enable_checkpointing=True
        ) as persistent_agent:
            try:
                # Prepare input for the persistent agent
                kwargs, run_id, thread_id = await self._handle_input(
                    request, persistent_agent
                )

                app_logger.info(
                    f"AgentManager streaming response for run_id: {run_id}, thread_id: {thread_id}"
                )

                # Reset tool call tracking for this stream
                self._current_tool_call_id = None

                # Use persistent agent for streaming - LangGraph will handle state automatically
                async for stream_event in persistent_agent.astream(
                    **kwargs, stream_mode=["updates", "messages", "custom"]
                ):
                    if not isinstance(stream_event, tuple):
                        continue

                    stream_mode, event = stream_event

                    # Update tool call tracking based on stream events
                    self._update_tool_call_tracking(stream_mode, event)

                    # Convert LangGraph events to simplified format
                    effective_session_id = request.session_id or thread_id
                    formatted_events = self._format_events(
                        stream_mode,
                        event,
                        request.stream_tokens,
                        run_id,
                        thread_id,
                        effective_session_id,
                    )

                    for formatted_event in formatted_events:
                        if formatted_event:
                            yield formatted_event

                # No manual state saving needed - LangGraph handles this automatically
                app_logger.info(
                    f"Conversation completed and auto-saved for thread {thread_id}"
                )

            except Exception as e:
                app_logger.error(f"Error in AgentManager stream_response: {e}")
                yield {
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "recoverable": False,
                        "error_type": "agent_error",
                    },
                }

    async def _handle_input(
        self, request: StreamRequest, agent: Pregel
    ) -> tuple[Dict[str, Any], str, str]:
        """Handle input preparation and configuration (preserving existing logic)."""
        run_id = uuid4()

        # Generate default thread_id if not provided
        thread_id = request.thread_id
        if thread_id is None:
            thread_id = str(uuid4())
            app_logger.info(
                f"Assigning auto-generated thread_id '{thread_id}' as thread_id is missing in user request"
            )

        # Configure tracing and session management (preserved from original)
        # If session_id is not provided, use thread_id as session_id
        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        # Register thread for user (for in-memory storage tracking)
        if settings.USE_INMEMORY_SAVER:
            register_thread(effective_user_id, thread_id)

        # Generate AI call ID
        ai_call_id = f"ai_call_{str(uuid4())}"

        configurable = {
            "thread_id": thread_id,
            "session_id": effective_session_id,
            "run_id": str(run_id),
            "user_id": effective_user_id,
            "ai_call_id": ai_call_id,
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "langfuse_observation_id": thread_id,
        }

        callbacks = []
        if langfuse_handler:
            callbacks.append(langfuse_handler)

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=callbacks if callbacks else None,
        )

        # Check for interrupts that need to be resumed (preserved from original)
        state = await agent.aget_state(config=config)
        interrupted_tasks = [
            task
            for task in state.tasks
            if hasattr(task, "interrupts") and task.interrupts
        ]

        # Prepare input based on whether we're resuming from an interrupt
        user_input_message: Command | Dict[str, Any]
        if interrupted_tasks:
            user_input_message = Command(resume=request.message)
        else:
            user_input_message = {"messages": [HumanMessage(content=request.message)]}

        kwargs = {
            "input": user_input_message,
            "config": config,
        }

        app_logger.info(
            f"AgentManager configured with run_id: {run_id}, thread_id: {thread_id}, session_id: {effective_session_id}"
        )
        return kwargs, str(run_id), thread_id

    def _format_events(
        self,
        stream_mode: str,
        event: Any,
        stream_tokens: bool,
        run_id: str,
        thread_id: str,
        session_id: str | None,
    ) -> list[Dict[str, Any]]:
        """Convert LangGraph events to simplified streaming format.

        This method implements the proposed event format while preserving
        all the business logic from the original implementation.
        """
        formatted_events = []

        if stream_mode == "updates":
            formatted_events.extend(
                self._handle_update_events(event, run_id, thread_id, session_id)
            )
        elif stream_mode == "messages" and stream_tokens:
            token_event = self._handle_token_events(event)
            if token_event:
                formatted_events.append(token_event)
        elif stream_mode == "custom":
            custom_event = self._handle_custom_events(
                event, run_id, thread_id, session_id
            )
            if custom_event:
                formatted_events.append(custom_event)

        return formatted_events

    def _handle_update_events(
        self, event: Dict[str, Any], run_id: str, thread_id: str, session_id: str | None
    ) -> list[Dict[str, Any]]:
        """Handle update events from LangGraph (preserving existing logic)."""
        formatted_events = []
        new_messages = []

        for node, updates in event.items():
            # Handle agent interrupts with structured messages (preserved)
            if node == "__interrupt__":
                interrupt: Interrupt
                for interrupt in updates:
                    new_messages.append(AIMessage(content=interrupt.value))
                continue

            updates = updates or {}
            update_messages = updates.get("messages", [])
            new_messages.extend(update_messages)

        # Process messages and convert to simplified format
        processed_messages = self._process_message_tuples(new_messages)

        for message in processed_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = run_id

                # Convert to simplified format
                formatted_event = {
                    "type": "message",
                    "content": self._convert_chat_message_to_simple_format(
                        chat_message, thread_id, session_id
                    ),
                }
                formatted_events.append(formatted_event)

            except Exception as e:
                app_logger.error(f"Error formatting message: {e}")
                formatted_events.append(
                    {
                        "type": "error",
                        "content": {
                            "message": "Message formatting error",
                            "recoverable": True,
                        },
                    }
                )

        return formatted_events

    def _handle_token_events(self, event: tuple) -> Dict[str, Any] | None:
        """Handle token streaming events with tool call ID tracking."""
        msg, metadata = event
        if "skip_stream" in metadata.get("tags", []):
            return None

        # Filter out non-LLM node messages (preserved logic)
        if not isinstance(msg, AIMessageChunk):
            return None

        content = remove_tool_calls(msg.content)
        if content:
            token_event = {
                "type": "token",
                "content": convert_message_content_to_string(content),
            }

            # Add tool call ID if this token is part of a tool call response
            tool_call_id = (
                self._extract_tool_call_id_from_message(msg)
                or self._current_tool_call_id
            )
            if tool_call_id:
                token_event["tool_call_id"] = tool_call_id

            return token_event
        return None

    def _handle_custom_events(
        self, event: Any, run_id: str, thread_id: str, session_id: str | None
    ) -> Dict[str, Any] | None:
        """Handle custom events from LangGraph."""
        try:
            chat_message = langchain_to_chat_message(event)
            chat_message.run_id = run_id

            return {
                "type": "message",
                "content": self._convert_chat_message_to_simple_format(
                    chat_message, thread_id, session_id
                ),
            }
        except Exception as e:
            app_logger.error(f"Error handling custom event: {e}")
            return None

    def _process_message_tuples(self, new_messages: list) -> list:
        """Process LangGraph streaming tuples and accumulate message parts (preserved logic)."""
        processed_messages = []
        current_message: Dict[str, Any] = {}

        for message in new_messages:
            if isinstance(message, tuple):
                key, value = message
                current_message[key] = value
            else:
                # Add complete message if we have one in progress
                if current_message:
                    processed_messages.append(self._create_ai_message(current_message))
                    current_message = {}
                processed_messages.append(message)

        # Add any remaining message parts
        if current_message:
            processed_messages.append(self._create_ai_message(current_message))

        return processed_messages

    def _create_ai_message(self, parts: Dict[str, Any]) -> AIMessage:
        """Create an AIMessage from a dictionary of parts (preserved from original)."""
        sig = inspect.signature(AIMessage)
        valid_keys = set(sig.parameters)
        filtered = {k: v for k, v in parts.items() if k in valid_keys}
        return AIMessage(**filtered)

    def _convert_chat_message_to_simple_format(
        self, chat_message, thread_id: str, session_id: str | None
    ) -> Dict[str, Any]:
        """Convert ChatMessage to simplified content format for the proposed API."""
        content = {
            "type": chat_message.type,
            "content": chat_message.content,
        }

        # Add optional fields only if present
        if chat_message.tool_calls:
            content["tool_calls"] = chat_message.tool_calls
        if chat_message.tool_call_id:
            content["tool_call_id"] = chat_message.tool_call_id
        if chat_message.run_id:
            content["run_id"] = chat_message.run_id
        if thread_id:
            content["thread_id"] = thread_id
        if session_id:
            content["session_id"] = session_id
        if chat_message.ai_call_id:
            content["ai_call_id"] = chat_message.ai_call_id
        if chat_message.response_metadata:
            content["response_metadata"] = chat_message.response_metadata
        if chat_message.custom_data:
            content["custom_data"] = chat_message.custom_data

        return content

    def _extract_tool_call_id_from_message(self, msg: AIMessageChunk) -> str | None:
        """Extract tool call ID from an AIMessageChunk if available.

        Args:
            msg: The AIMessageChunk to extract tool call ID from

        Returns:
            The tool call ID if available, None otherwise
        """
        try:
            # Check if the message has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Return the ID of the first tool call
                return msg.tool_calls[0].get("id")

            # Check if the message has tool_call_chunks (streaming tool calls)
            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                # Return the ID of the first tool call chunk
                return msg.tool_call_chunks[0].get("id")

            # Check if this is a response to a tool call (has tool_call_id)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                return msg.tool_call_id

            return None
        except (AttributeError, IndexError, KeyError) as e:
            app_logger.debug(f"Could not extract tool call ID from message: {e}")
            return None

    def _update_tool_call_tracking(self, stream_mode: str, event: Any) -> None:
        """Update the current tool call ID based on streaming events.

        Args:
            stream_mode: The type of stream event
            event: The event data
        """
        try:
            if stream_mode == "updates":
                # Look for tool calls in update events
                for node, updates in event.items():
                    if updates and "messages" in updates:
                        for message in updates["messages"]:
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                # Found a new tool call, update tracking
                                self._current_tool_call_id = message.tool_calls[0].get(
                                    "id"
                                )
                                app_logger.debug(
                                    f"Tracking tool call ID: {self._current_tool_call_id}"
                                )
                                return
                            elif (
                                hasattr(message, "tool_call_id")
                                and message.tool_call_id
                            ):
                                # This is a tool response, track its ID
                                self._current_tool_call_id = message.tool_call_id
                                app_logger.debug(
                                    f"Tracking tool response ID: {self._current_tool_call_id}"
                                )
                                return

            elif stream_mode == "messages":
                # Check message stream for tool calls
                msg, metadata = event
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    self._current_tool_call_id = msg.tool_calls[0].get("id")
                    app_logger.debug(
                        f"Tracking tool call ID from message: {self._current_tool_call_id}"
                    )
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    self._current_tool_call_id = msg.tool_call_id
                    app_logger.debug(
                        f"Tracking tool response ID from message: {self._current_tool_call_id}"
                    )

        except Exception as e:
            app_logger.debug(f"Error updating tool call tracking: {e}")
            # Don't fail streaming due to tracking issues
