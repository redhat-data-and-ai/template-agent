"""LLM-as-Judge evaluator using Gemini."""

from typing import Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse

from deep_agent.utils.google_creds import get_service_account_credentials

# Model configuration
MODEL_NAME = "gemini-3.1-pro-preview"
MODEL_TEMPERATURE = 0
OUTPUT_TRUNCATE_LENGTH = 500

# Response field markers
VERDICT_MARKER = "VERDICT:"
EVIDENCE_MARKER = "EVIDENCE:"
CONFIDENCE_MARKER = "CONFIDENCE:"
REASONING_MARKER = "REASONING:"


def create_judge_prompt(assertion: str, output: str, context: Optional[Dict]) -> str:
    """Build evaluation prompt for LLM judge."""
    sections = [
        "You are an expert evaluator. Assess whether the agent's output satisfies the assertion.",
        "",
        f"ASSERTION: {assertion}",
        "",
        f"AGENT OUTPUT:\n{output}",
    ]

    if context:
        sections.extend(["", "CONTEXT:"])
        if context.get("expected_output"):
            sections.append(f"Expected: {context['expected_output']}")
        if context.get("prompt"):
            sections.append(f"User Prompt: {context['prompt']}")
        if context.get("skill_name"):
            sections.append(f"Skill: {context['skill_name']}")

    sections.extend(
        [
            "",
            "Evaluate strictly but fairly. Provide:",
            "VERDICT: YES or NO",
            "EVIDENCE: Quote or describe specific evidence",
            "CONFIDENCE: 0.0 to 1.0",
            "REASONING: Brief explanation",
        ]
    )

    return "\n".join(sections)


def parse_judge_response(response: str) -> Dict:
    """Parse structured LLM judge response."""
    result = {
        "passed": None,
        "evidence": "",
        "confidence": 0.5,
        "reasoning": "",
    }

    for line in response.strip().split("\n"):
        line = line.strip()

        if line.startswith(VERDICT_MARKER):
            verdict = line.split(":", 1)[1].strip().upper()
            result["passed"] = verdict == "YES"
        elif line.startswith(EVIDENCE_MARKER):
            result["evidence"] = line.split(":", 1)[1].strip()
        elif line.startswith(CONFIDENCE_MARKER):
            try:
                conf = float(line.split(":", 1)[1].strip())
                result["confidence"] = max(0.0, min(1.0, conf))
            except ValueError:
                pass
        elif line.startswith(REASONING_MARKER):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result


def extract_text_content(content) -> str:
    """Extract text from Gemini response content (handles str or list)."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return str(content)


class LLMJudge:
    """LLM-as-judge evaluator with Langfuse tracing."""

    def __init__(self, langfuse_client: Optional[Langfuse] = None):
        credentials, project = get_service_account_credentials()
        self.model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            credentials=credentials,
            project=project,
        )
        self.langfuse = langfuse_client

    def evaluate(
        self,
        assertion: str,
        output: str,
        context: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> Dict:
        """Evaluate assertion using LLM judge."""
        prompt = create_judge_prompt(assertion, output, context)
        generation = self._create_generation(assertion, output, context, trace_id)

        try:
            response = self.model.invoke(prompt)
            content = extract_text_content(response.content)
            result = parse_judge_response(content)
            self._finalize_generation(generation, result, assertion)
            return result

        except Exception as e:
            self._handle_error(generation, e)
            return {
                "passed": None,
                "evidence": f"LLM judge error: {str(e)}",
                "confidence": 0.0,
                "reasoning": "",
            }

    def _create_generation(
        self,
        assertion: str,
        output: str,
        context: Optional[Dict],
        trace_id: Optional[str],
    ):
        """Create Langfuse generation span."""
        if not (self.langfuse and trace_id):
            return None

        generation_input = {
            "assertion": assertion,
            "output": output[:OUTPUT_TRUNCATE_LENGTH],
        }
        if context:
            generation_input["context"] = context

        return self.langfuse.generation(
            trace_id=trace_id,
            name="llm_judge_evaluation",
            model=MODEL_NAME,
            input=generation_input,
        )

    def _finalize_generation(self, generation, result: Dict, assertion: str):
        """Update and close Langfuse generation."""
        if not generation:
            return

        generation.update(
            output=result,
            metadata={
                "assertion": assertion,
                "passed": result["passed"],
                "confidence": result.get("confidence", 0.0),
            },
        )
        generation.end()

    def _handle_error(self, generation, error: Exception):
        """Handle and log error to Langfuse."""
        if not generation:
            return

        generation.update(
            level="ERROR",
            status_message=str(error),
        )
        generation.end()
