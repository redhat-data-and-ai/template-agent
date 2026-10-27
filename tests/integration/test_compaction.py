"""Exercise manual compaction through a real graph with a scripted local model."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deep_agent.src.infrastructure.middleware import (
    _build_summarization_middleware,
    _build_summarization_tool_middleware,
)


class RecordingChatModel(FakeMessagesListChatModel):
    """Script tool selection and record the context delivered to the model."""

    requests: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _compact_call(input_tokens: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "compact_conversation", "args": {}, "id": "compact-1"}],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 1,
            "total_tokens": input_tokens + 1,
        },
    )


def _build_agent(tmp_path, responses, threshold=0.75):
    model = RecordingChatModel(
        responses=responses, profile={"max_input_tokens": 100000}
    )
    # The library trusts reported usage only when its provider matches the model.
    for response in responses:
        response.response_metadata["model_provider"] = model._get_ls_params()[
            "ls_provider"
        ]
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    automatic = _build_summarization_middleware(
        model=model, backend=backend, threshold=threshold
    )
    manual = _build_summarization_tool_middleware(summarization=automatic)
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[automatic, manual],
        checkpointer=InMemorySaver(),
    )
    return graph, model, automatic


def _history():
    return [HumanMessage(content="history " * 1000, id=str(i)) for i in range(7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold,eligible", [(0.60, True), (0.90, False)])
async def test_manual_compaction_uses_configured_engine(tmp_path, threshold, eligible):
    responses = [_compact_call(40000)]
    if eligible:
        responses.append(AIMessage(content="Summary of older conversation"))
    responses.extend(
        [
            AIMessage(content="Compaction result received."),
            AIMessage(content="Continuing."),
        ]
    )
    graph, model, _ = _build_agent(tmp_path, responses, threshold)
    config = {"configurable": {"thread_id": "manual-compaction"}}
    history = _history()
    request = HumanMessage(content="compact this conversation", id="request")

    result = await graph.ainvoke({"messages": [*history, request]}, config)
    state = (await graph.aget_state(config)).values

    assert "When the user asks to compact the conversation" in model.requests[0][0].text
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "compact-1"
    archives = list((tmp_path / "conversation_history").glob("*.md"))
    if eligible:
        assert "Conversation compacted" in tool_results[0].content
        assert len(archives) == 1
        assert "history history" in archives[0].read_text()
        event = state["_summarization_event"]
        assert event["cutoff_index"] > 0
        assert "Summary of older conversation" in event["summary_message"].content
        # The model receives compacted context while the checkpoint retains the log.
        assert history[0] not in model.requests[-1]
        assert history[-1] in model.requests[-1]
        assert request in model.requests[-1]
        assert history[0] in result["messages"]
    else:
        assert "Nothing to compact" in tool_results[0].content
        assert archives == []
        assert not state.get("_summarization_event")

    continued = await graph.ainvoke(
        {"messages": [HumanMessage(content="Let's continue.")]}, config
    )
    assert continued["messages"][-1].content == "Continuing."
    if eligible:
        assert history[0] not in model.requests[-1]
        assert any(
            "Summary of older conversation" in m.content for m in model.requests[-1]
        )


@pytest.mark.asyncio
async def test_short_conversation_returns_nothing_to_compact(tmp_path):
    graph, _, _ = _build_agent(
        tmp_path, [_compact_call(20), AIMessage(content="Nothing to compact yet.")]
    )
    config = {"configurable": {"thread_id": "short-conversation"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="compact this conversation")]},
        config,
    )
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Nothing to compact" in tool_result.content
    assert not (await graph.aget_state(config)).values.get("_summarization_event")
    assert not (tmp_path / "conversation_history").exists()


@pytest.mark.asyncio
async def test_summary_failure_returns_tool_error_without_losing_history(
    tmp_path, monkeypatch
):
    graph, _, automatic = _build_agent(
        tmp_path, [_compact_call(40000), AIMessage(content="Compaction failed.")]
    )
    monkeypatch.setattr(
        automatic,
        "_acreate_summary",
        AsyncMock(side_effect=RuntimeError("summary unavailable")),
    )
    history = _history()
    config = {"configurable": {"thread_id": "failed-compaction"}}
    result = await graph.ainvoke(
        {"messages": [*history, HumanMessage(content="compact this conversation")]},
        config,
    )
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Compaction failed" in tool_result.content
    assert "summary unavailable" in tool_result.content
    assert not (await graph.aget_state(config)).values.get("_summarization_event")
    assert all(message in result["messages"] for message in history)
