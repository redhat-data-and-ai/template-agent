"""Unit tests for the embedded query classifier."""

import asyncio
from unittest.mock import patch

import pytest

from deep_agent.aegra.classifier.classify import classify_query
from deep_agent.aegra.classifier.config import (
    classification_dir,
    get_maas_settings,
    load_logic,
    load_product_catalog,
    load_product_tables,
    load_system_prompt,
)
from deep_agent.aegra.classifier.maas_client import consume_sse_line
from deep_agent.aegra.classifier.runnable import ClassifierRunnable


def test_maas_settings_read_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "from-env")
    monkeypatch.setenv("MAAS_VERIFY_SSL", "false")
    settings = get_maas_settings()
    assert settings.api_key == "from-env"
    assert settings.verify_ssl is False


def test_classification_yaml_loads_catalog_logic_and_tables() -> None:
    directory = classification_dir()
    assert (directory / "logic.yaml").is_file()
    assert (directory / "product_catalog.yaml").is_file()
    assert (directory / "product_tables.yaml").is_file()

    logic = load_logic()
    catalog = load_product_catalog()
    tables = load_product_tables()
    prompt = load_system_prompt()

    assert "data-query" in logic["intent_classes"]
    assert catalog["product_count"] == 74
    assert "bookingsmaster" in tables
    assert "bookingsmaster" in prompt
    assert "YOUR TASK" in prompt


def test_classify_query_without_key_uses_yaml_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    result = asyncio.run(classify_query("hello there"))
    assert result.intents[0].intent_class == "direct-response"
    assert "<classification>" in result.to_classification_block()


def test_consume_sse_line_joins_content_and_stops_on_done() -> None:
    content: list[str] = []
    reasoning: list[str] = []
    line = 'data: {"choices":[{"delta":{"content":"abc"}}]}'
    assert consume_sse_line(line, content, reasoning) is False
    assert consume_sse_line("data: [DONE]", content, reasoning) is True
    assert content == ["abc"]


@pytest.mark.asyncio
async def test_runnable_streams_answer_chunks_without_buffering() -> None:
    class _Inner:
        def __init__(self) -> None:
            self.seen = None

        async def astream(self, data, config=None, **kwargs):
            self.seen = data
            yield {"type": "token", "text": "Hel"}
            yield {"type": "token", "text": "lo"}

    class _Result:
        sub_queries = [object()]
        intents = []

        def to_classification_block(self) -> str:
            return "<classification>\n  intent: data-query\n</classification>"

    async def _fake_classify(_query: str) -> _Result:
        return _Result()

    inner = _Inner()
    wrapper = ClassifierRunnable(inner)
    chunks = []
    with patch(
        "deep_agent.aegra.classifier.runnable.classify_query",
        new=_fake_classify,
    ):
        async for chunk in wrapper.astream(
            {"messages": [{"role": "human", "content": "bookings"}]}
        ):
            chunks.append(chunk)

    assert [chunk["text"] for chunk in chunks] == ["Hel", "lo"]
    injected = inner.seen["messages"][-1]["content"]
    assert injected.startswith("<classification>")
    assert injected.endswith("bookings")
