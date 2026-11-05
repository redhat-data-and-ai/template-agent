"""Classifier-aware runnable wrapper.

Intercepts human messages entering the graph, runs DiffusionGemma classification,
and prepends a <classification> block to the message content so the agent's
PROMPT.md instructions can parse and use the structured routing/intent data.

This wrapper sits INSIDE the SafetyAwareRunnable (classification before safety)
so the classifier output is also subject to content safety checks.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

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

    async def _classify_and_inject(self, input_data: Any) -> Any:
        """Run classification and inject the block into the input."""
        human_text = _extract_human_text(input_data)
        if not human_text:
            return input_data

        if "<classification>" in human_text:
            logger.debug("Classification block already present, skipping")
            return input_data

        try:
            result = await classify_query(human_text)
            block = result.to_classification_block()
            if block:
                logger.info(
                    "Classification injected: %d sub_queries, intents=%s",
                    len(result.sub_queries),
                    [i.intent_class for i in result.intents],
                )
                return _inject_classification(input_data, block)
        except Exception:
            logger.warning("Classification failed, proceeding without", exc_info=True)

        return input_data

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Classify, then invoke the inner graph."""
        enriched = await self._classify_and_inject(input)
        return await self._runnable.ainvoke(enriched, config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Classify, then forward each agent chunk as it arrives.

        Classification finishes before the first answer token because the
        model needs the product and tables. After that, chunks are yielded
        immediately and are not collected into a buffer.
        """
        enriched = await self._classify_and_inject(input)
        async for chunk in self._runnable.astream(enriched, config, **kwargs):
            yield chunk

    async def astream_events(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        """Same as astream, for the events API the UI stream uses."""
        enriched = await self._classify_and_inject(input)
        async for event in self._runnable.astream_events(enriched, config, **kwargs):
            yield event
