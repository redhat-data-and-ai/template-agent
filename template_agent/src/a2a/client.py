"""A2A client for calling downstream A2A agents with token forwarding."""

from uuid import uuid4

import httpx
from a2a.client import (
    AuthInterceptor,
    Client,
    ClientCallContext,
    ClientConfig,
    CredentialService,
    create_client,
)
from a2a.client.interceptors import AfterArgs, BeforeArgs, ClientCallInterceptor
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
)

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class ForwardingCredentialService(CredentialService):
    """Returns the caller's access token for forwarding to downstream agents.

    When our agent calls a downstream A2A agent, we forward the same
    bearer token that the upstream caller sent to us. This preserves
    the identity chain across the agent network.
    """

    def __init__(self, access_token: str):
        """Store the access token for downstream forwarding."""
        self._token = access_token

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str | None:
        """Return the stored bearer token regardless of scheme or context."""
        return self._token


class CallingAgentInterceptor(ClientCallInterceptor):
    """Injects an X-Calling-Agent-ID header into every outbound A2A request.

    This lets downstream agents identify which agent is calling them,
    using the A2A spec's Service Parameters mechanism (Section 3.2.6).
    """

    def __init__(self, agent_name: str):
        """Initialize with the calling agent's identifier."""
        self._agent_name = agent_name

    async def before(self, args: BeforeArgs) -> None:
        """Inject X-Calling-Agent-ID into outbound service parameters."""
        if args.context is None:
            args.context = ClientCallContext()
        if args.context.service_parameters is None:
            args.context.service_parameters = {}
        args.context.service_parameters["X-Calling-Agent-ID"] = self._agent_name

    async def after(self, args: AfterArgs) -> None:
        """No-op post-request hook (required by interface)."""
        pass


async def create_a2a_client(agent_url: str, access_token: str | None = None) -> Client:
    """Create an A2A client for a downstream agent with token forwarding.

    Discovers the downstream agent's Agent Card, then creates a Client
    with an AuthInterceptor that attaches the forwarded bearer token.

    Args:
        agent_url: Base URL of the downstream A2A agent.
        access_token: Bearer token to forward. If None, no auth is attached.

    Returns:
        A configured A2A Client ready to send messages.
    """
    interceptors = [CallingAgentInterceptor(settings.A2A_AGENT_NAME)]
    if access_token:
        cred_service = ForwardingCredentialService(access_token)
        interceptors.append(AuthInterceptor(cred_service))

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        verify=False,  # nosec B501 — internal service mesh; TLS terminated at ingress
    )

    client = await create_client(
        agent=agent_url,
        client_config=ClientConfig(
            streaming=True,
            httpx_client=http_client,
        ),
        interceptors=interceptors,
    )

    logger.info(
        f"A2A client created for {agent_url}, "
        f"auth={'forwarded' if access_token else 'none'}"
    )
    return client


async def send_to_downstream_agent(
    agent_url: str,
    message_text: str,
    access_token: str | None = None,
    context_id: str | None = None,
) -> str:
    """Send a message to a downstream A2A agent and collect the response.

    Creates a client, sends the message, streams back events, and returns
    the final text result from completed artifacts.

    Args:
        agent_url: Base URL of the downstream A2A agent.
        message_text: The text message to send.
        access_token: Bearer token to forward to the downstream agent.
        context_id: Optional context ID for multi-turn conversations.

    Returns:
        The text content from the downstream agent's response artifacts.
    """
    client = await create_a2a_client(agent_url, access_token)

    message = Message(
        message_id=str(uuid4()),
        role=Role.ROLE_USER,
        parts=[Part(text=message_text)],
    )
    if context_id:
        message.context_id = context_id

    request = SendMessageRequest(message=message)

    terminal_states = {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }

    result_parts: list[str] = []

    event: StreamResponse
    async for event in client.send_message(request):
        if event.HasField("artifact_update"):
            for part in event.artifact_update.artifact.parts:
                if part.HasField("text") and part.text:
                    result_parts.append(part.text)

        if event.HasField("status_update"):
            state = event.status_update.status.state
            if state == TaskState.TASK_STATE_FAILED:
                status_msg = ""
                if event.status_update.status.HasField("message"):
                    for p in event.status_update.status.message.parts:
                        if p.HasField("text"):
                            status_msg += p.text
                raise RuntimeError(
                    f"Downstream agent failed: {status_msg or 'unknown error'}"
                )
            if state in terminal_states:
                break

        if event.HasField("task"):
            task: Task = event.task
            if task.status.state == TaskState.TASK_STATE_FAILED:
                raise RuntimeError("Downstream agent task failed")
            if task.status.state in terminal_states:
                break

    return (
        "\n".join(result_parts)
        if result_parts
        else "No response from downstream agent."
    )
