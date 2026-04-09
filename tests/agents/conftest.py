"""Pytest configuration and fixtures for agent-level tests."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import google.auth
import pytest
from deepagents import SubAgent, create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langgraph.checkpoint.memory import MemorySaver

from template_agent.src.core.backend import get_backend
from template_agent.utils.google_creds import initialize_google_genai

from llm_judge import LLMJudge
from mock_tools import MOCK_TOOLS


PROJECT_ROOT = Path(__file__).parent.parent.parent
AGENT_CONFIG_DIR = PROJECT_ROOT / "template_agent" / "agent_config"
AGENTS_DIR = AGENT_CONFIG_DIR / "agents"
SKILLS_DIR = AGENT_CONFIG_DIR / "skills"

MODEL_NAME = "gemini-3.1-pro-preview"
MODEL_TEMPERATURE = 0


# ============================================================================
# Session Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def skills_dir():
    """Skills directory path."""
    return SKILLS_DIR


@pytest.fixture(scope="session")
def workspace_dir():
    """Workspace directory for test outputs."""
    workspace = PROJECT_ROOT / "tests" / "workspaces"
    workspace.mkdir(exist_ok=True)
    return workspace


@pytest.fixture(scope="session")
def model():
    """Create Gemini model with credentials."""
    # Initialize Google credentials from environment
    initialize_google_genai()

    # Check if credentials are available
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("Google Cloud credentials not available - skipping agent tests")

    # Use the same pattern as agent.py
    try:
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            credentials=credentials,
            project=project,
        )
    except google.auth.exceptions.DefaultCredentialsError:
        pytest.skip("Google Cloud credentials not available - skipping agent tests")


# ============================================================================
# Function Fixtures
# ============================================================================


@pytest.fixture
def tracer():
    """Execution tracer for timing and token tracking."""
    return ExecutionTracer()


@pytest.fixture
def langfuse_client():
    """Langfuse client (optional, requires env vars)."""
    if all(
        [
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
            os.getenv("LANGFUSE_BASE_URL"),
        ]
    ):
        return Langfuse()
    return None


@pytest.fixture
def evaluator(langfuse_client):
    """LLM judge evaluator."""
    judge = LLMJudge(langfuse_client=langfuse_client)
    return AssertionEvaluator(judge)


# ============================================================================
# Agent Creation Helpers
# ============================================================================


def create_analyst_agent(model):
    """Create analyst subagent with bmi-report skill and tools."""
    skill_path = SKILLS_DIR / "bmi-report"

    agent = create_deep_agent(
        model=model,
        skills=[str(skill_path.resolve())],
        tools=[MOCK_TOOLS["calculate_bmi"], MOCK_TOOLS["search_web"]],
        backend=get_backend(),
        checkpointer=MemorySaver(),
    )
    return agent


def create_publisher_agent(model):
    """Create publisher subagent with email-formatter skill and tools."""
    skill_path = SKILLS_DIR / "email-formatter"

    agent = create_deep_agent(
        model=model,
        skills=[str(skill_path.resolve())],
        tools=[MOCK_TOOLS["send_email"]],
        backend=get_backend(),
        checkpointer=MemorySaver(),
    )
    return agent


def create_orchestrator_agent(model):
    """Create orchestrator agent with client-intake skill and subagents."""
    from subagent_loader import load_subagents

    # Load analyst and publisher subagents
    subagents = load_subagents(AGENTS_DIR, SKILLS_DIR)

    # Main agent uses client-intake skill
    main_skill_path = SKILLS_DIR / "client-intake"

    agent = create_deep_agent(
        model=model,
        skills=[str(main_skill_path.resolve())],
        subagents=subagents,
        backend=get_backend(),
        checkpointer=MemorySaver(),
    )
    return agent


# ============================================================================
# Helpers
# ============================================================================


def load_skill_evals(skill_name: str) -> Dict[str, Any]:
    """Load evals.json for a skill."""
    evals_file = SKILLS_DIR / skill_name / "evals" / "evals.json"

    if not evals_file.exists():
        raise FileNotFoundError(f"No evals.json for {skill_name}")

    with open(evals_file) as f:
        return json.load(f)


def extract_output(result: dict) -> str:
    """Extract text from agent result messages."""
    messages = result.get("messages", [])

    for msg in reversed(messages):
        if not (hasattr(msg, "content") and msg.content):
            continue
        if hasattr(msg, "type") and msg.type == "human":
            continue

        content = msg.content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    return ""


def extract_tokens(result: dict) -> int:
    """Extract total token count from messages."""
    total = 0
    for msg in result.get("messages", []):
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total += msg.usage_metadata.get("total_tokens", 0)
    return total


# ============================================================================
# Classes
# ============================================================================


class ExecutionTracer:
    """Tracks execution time and token usage."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_tokens = 0

    def start(self):
        self.start_time = time.time()

    def end(self, total_tokens: int = 0):
        self.end_time = time.time()
        self.total_tokens = total_tokens

    def duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0


class AssertionEvaluator:
    """Evaluates assertions using LLM judge."""

    def __init__(self, llm_judge: LLMJudge):
        self.llm_judge = llm_judge

    def evaluate(
        self,
        assertion: str,
        output: str,
        context: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> Dict:
        """Evaluate assertion against output."""
        result = self.llm_judge.evaluate(assertion, output, context, trace_id)
        result["method"] = "llm_judge"
        return result


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "analyst: analyst subagent tests")
    config.addinivalue_line("markers", "publisher: publisher subagent tests")
    config.addinivalue_line("markers", "orchestrator: orchestrator agent tests")
