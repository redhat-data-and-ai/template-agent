"""Pytest configuration and fixtures for skills tests with auto-discovery."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse

from llm_judge import LLMJudge

PROJECT_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / "config" / "agent" / "skills"

MODEL_NAME = "gemini-3.1-pro-preview"
MODEL_TEMPERATURE = 0


# ============================================================================
# Auto-Discovery
# ============================================================================


def pytest_generate_tests(metafunc):
    """Auto-discover all skills and their evals."""
    if "skill_eval" not in metafunc.fixturenames:
        return

    if not SKILLS_DIR.exists():
        pytest.skip(f"Skills directory not found: {SKILLS_DIR}")
        return

    test_cases = []
    ids = []

    # Discover all skills
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_name = skill_dir.name
        evals_file = skill_dir / "evals" / "evals.json"

        if not evals_file.exists():
            continue

        # Load evals for this skill
        with open(evals_file) as f:
            evals_data = json.load(f)

        # Create test case for each eval
        for eval_case in evals_data.get("evals", []):
            eval_id = eval_case["id"]
            test_cases.append(
                {
                    "skill_name": skill_name,
                    "skill_dir": str(skill_dir.resolve()),
                    "eval_id": eval_id,
                    "eval_case": eval_case,
                }
            )
            ids.append(f"{skill_name}-eval-{eval_id}")

    if not test_cases:
        pytest.skip("No skill evals found")
        return

    metafunc.parametrize(
        "skill_eval",
        test_cases,
        ids=ids,
    )


# ============================================================================
# Session Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def workspace_dir():
    """Workspace directory for test outputs."""
    workspace = PROJECT_ROOT / "tests" / "workspaces" / "skills"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def model():
    """Create Gemini model with credentials.

    Function-scoped to ensure each test gets a fresh model instance
    bound to the correct event loop.
    """
    from deep_agent.utils.google_creds import get_service_account_credentials

    # Check if credentials are available
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT"):
        pytest.skip("Google Cloud credentials not available - skipping skill tests")

    try:
        credentials, project = get_service_account_credentials()
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            credentials=credentials,
            project=project,
        )
    except RuntimeError as e:
        pytest.skip(f"Google Cloud credentials error: {e}")


# ============================================================================
# Function Fixtures
# ============================================================================


@pytest.fixture
def tracer():
    """Execution tracer for timing and token tracking."""
    return ExecutionTracer()


@pytest.fixture
def langfuse_client():
    """Langfuse client (optional, requires env vars).

    Ensures traces are flushed before test teardown.
    """
    if all(
        [
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
            os.getenv("LANGFUSE_BASE_URL"),
        ]
    ):
        client = Langfuse()
        yield client
        # Flush pending traces before test cleanup
        client.flush()
    else:
        yield None


@pytest.fixture
def evaluator(langfuse_client):
    """LLM judge evaluator."""
    judge = LLMJudge(langfuse_client=langfuse_client)
    return AssertionEvaluator(judge)


# ============================================================================
# Helper Functions
# ============================================================================


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
    config.addinivalue_line("markers", "skills: skills evaluation tests")
