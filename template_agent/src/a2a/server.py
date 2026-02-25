from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from template_agent.src.a2a.agent_executor import TemplateAgentExecutor
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def build_agent_card() -> AgentCard:
    skills = [
        AgentSkill(
            id=settings.A2A_SKILL_ID,
            name=settings.A2A_SKILL_NAME,
            description=settings.A2A_SKILL_DESCRIPTION,
            tags=["agent", "general"],
            examples=["Process this request", "Help me with a task"],
        ),
    ]

    return AgentCard(
        name=settings.A2A_AGENT_NAME,
        description=settings.A2A_AGENT_DESCRIPTION,
        url=settings.A2A_AGENT_URL,
        version=settings.A2A_AGENT_VERSION,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )


def create_a2a_app():
    agent_card = build_agent_card()

    request_handler = DefaultRequestHandler(
        agent_executor=TemplateAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    return server.build()
