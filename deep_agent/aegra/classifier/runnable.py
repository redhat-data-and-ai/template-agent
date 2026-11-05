"""Classifier-aware runnable wrapper.

Intercepts human messages entering the graph, runs DiffusionGemma classification,
and prepends a <classification> block to the message content so the agent's
PROMPT.md instructions can parse and use the structured routing/intent data.

This wrapper sits INSIDE the SafetyAwareRunnable (classification before safety)
so the classifier output is also subject to content safety checks.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from deep_agent.aegra.classifier.classify import classify_query

logger = logging.getLogger(__name__)


def _extract_human_text(input_data: Any) -> str | None:
    """Pull the last human/user message text from the graph input."""
    if not isinstance(input_data, dict):
        return None
    messages = input_data.get("messages", [])
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, dict):
        role = last.get("role", "")
        if role in ("human", "user"):
            content = last.get("content", "")
            return content if isinstance(content, str) and content.strip() else None
    if hasattr(last, "type") and last.type in ("human",):
        content = getattr(last, "content", "")
        return content if isinstance(content, str) and content.strip() else None
    return None


def _inject_classification(input_data: dict, block: str) -> dict:
    """Prepend the <classification> block to the last human message."""
    messages = list(input_data.get("messages", []))
    if not messages:
        return input_data

    last = messages[-1]
    if isinstance(last, dict):
        original = last.get("content", "")
        messages[-1] = {**last, "content": f"{block}\n\n{original}"}
    elif hasattr(last, "content"):
        original = last.content or ""
        messages[-1] = HumanMessage(content=f"{block}\n\n{original}")

    return {**input_data, "messages": messages}


def _skipped_thought(reason: str) -> str:
    """Thought text when classification does not produce a route."""
    return "\n".join(
        [
            "**Intent classification**",
            "Source: skipped",
            f"Why: {reason}",
            "The agent is continuing without a new product route.",
        ]
    )


def _thought_text(result: Any) -> str:
    """Use the classifier record when it can describe itself."""
    writer = getattr(result, "to_thought", None)
    if not callable(writer):
        return _skipped_thought("classifier returned no thought record")
    text = writer()
    if isinstance(text, str) and text.strip():
        return text
    return _skipped_thought("classifier returned an empty thought")


def _thought_chunk(text: str, stream_mode: Any) -> Any:
    """Shape the thought the way Aegra turns messages into thoughts."""
    message = AIMessage(
        id=f"classification-{uuid.uuid4().hex}",
        content=[{"type": "text", "text": text, "thought": True}],
    )
    payload = (message, {"langgraph_node": "classifier"})
    if isinstance(stream_mode, str):
        return payload
    return ("messages", payload)


class ClassifierRunnable:
    """Proxy that classifies the user query before forwarding to the inner graph.

    Usage in graph.py:
        compiled = ClassifierRunnable(compiled)
    """

    def __init__(self, runnable: Any) -> None:
        """Wrap the compiled graph."""
        self._runnable = runnable

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped graph."""
        return getattr(self._runnable, name)

    def copy(self, **kwargs: Any) -> "ClassifierRunnable":
        """Copy the inner graph and keep the classifier wrapper."""
        return ClassifierRunnable(self._runnable.copy(**kwargs))

    def with_config(self, config: Any = None, **kwargs: Any) -> "ClassifierRunnable":
        """Re-wrap after with_config so the classifier is not stripped."""
        if config is not None:
            inner = self._runnable.with_config(config, **kwargs)
        else:
            inner = self._runnable.with_config(**kwargs)
        return ClassifierRunnable(inner)

    async def _classify_and_inject(self, input_data: Any) -> tuple[Any, str]:
        """Classify the user text and return the enriched input plus the thought."""
        human_text = _extract_human_text(input_data)
        if not human_text:
            thought = _skipped_thought("no user text on this turn")
            logger.info("%s", thought)
            return input_data, thought

        if "<classification>" in human_text:
            thought = _skipped_thought("classification block already present")
            logger.info("%s", thought)
            return input_data, thought

        try:
            result = await classify_query(human_text)
        except Exception as exc:
            thought = _skipped_thought(f"classification raised {exc}")
            logger.warning("%s", thought, exc_info=True)
            return input_data, thought

        thought = _thought_text(result)
        block = ""
        if hasattr(result, "to_classification_block"):
            block = result.to_classification_block()
        if block:
            return _inject_classification(input_data, block), thought
        return input_data, thought

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Classify, then invoke the inner graph."""
        enriched, _thought = await self._classify_and_inject(input)
        return await self._runnable.ainvoke(enriched, config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Classify, emit the thought, then forward each agent chunk.

        The thought is yielded before the first answer token. Classification
        has to finish first because the model needs the product and tables.
        Later chunks are yielded immediately and are not buffered.
        """
        enriched, thought = await self._classify_and_inject(input)
        yield _thought_chunk(thought, kwargs.get("stream_mode"))
        async for chunk in self._runnable.astream(enriched, config, **kwargs):
            yield chunk

    async def astream_events(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        """Same as astream, for the events API the UI stream uses."""
        enriched, thought = await self._classify_and_inject(input)
        yield _thought_chunk(thought, kwargs.get("stream_mode"))
        async for event in self._runnable.astream_events(enriched, config, **kwargs):
            yield event
