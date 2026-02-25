import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import StreamRequest

logger = logging.getLogger(__name__)


class TemplateAgentExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        logger.info("Executing A2A request for query: %s", query)

        agent_manager = AgentManager()
        request = StreamRequest(message=query, stream_tokens=False)

        result_parts = []
        async for event in agent_manager.stream_response(request):
            event_type = event.get("type")
            if event_type == "message":
                content = event.get("content", {})
                if content.get("type") == "ai" and content.get("content"):
                    result_parts.append(content["content"])
            elif event_type == "error":
                error_msg = event.get("content", {}).get("message", "Unknown error")
                result_parts.append(f"Error: {error_msg}")

        result = result_parts[-1] if result_parts else "No response generated."
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise Exception("cancel not supported")
