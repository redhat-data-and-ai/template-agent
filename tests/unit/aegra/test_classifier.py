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


def test_classify_query_without_key_uses_yaml_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    result = asyncio.run(classify_query("hello there"))
    assert result.intents[0].intent_class == "direct-response"
    assert "<classification>" in result.to_classification_block()
    thought = result.to_thought()
    assert "Source: fallback" in thought
    assert "MAAS_API_KEY is not set" in thought
    assert "direct-response" in thought


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

        def to_thought(self) -> str:
            return "**Intent classification**\nSource: model\nIntents: data-query"

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
            {"messages": [{"role": "human", "content": "bookings"}]},
            stream_mode=["messages", "updates"],
        ):
            chunks.append(chunk)

    mode, payload = chunks[0]
    message, meta = payload
    assert mode == "messages"
    assert meta["langgraph_node"] == "classifier"
    assert message.content[0]["thought"] is True
    assert "Intents: data-query" in message.content[0]["text"]
    assert [chunk["text"] for chunk in chunks[1:]] == ["Hel", "lo"]
    injected = inner.seen["messages"][-1]["content"]
    assert injected.startswith("<classification>")
    assert injected.endswith("bookings")


def test_classification_dir_override_and_yaml_edges(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deep_agent.aegra.classifier import config as cfg

    monkeypatch.setenv("CLASSIFICATION_DATA_DIR", str(tmp_path))
    cfg._load_yaml.cache_clear()
    assert cfg.classification_dir() == tmp_path
    assert cfg.load_logic() == {}
    assert "data-query" in cfg.intent_classes()
    assert "person" in cfg.entity_types()

    (tmp_path / "logic.yaml").write_text("just a string\n", encoding="utf-8")
    cfg._load_yaml.cache_clear()
    assert cfg.load_logic() == {}

    (tmp_path / "logic.yaml").write_text(
        "prompts:\n  routing_guide: only-guide\n",
        encoding="utf-8",
    )
    cfg._load_yaml.cache_clear()
    assert cfg.load_system_prompt() == "only-guide"

    (tmp_path / "logic.yaml").write_text(
        "prompts:\n  task: only-task\n",
        encoding="utf-8",
    )
    cfg._load_yaml.cache_clear()
    assert cfg.load_system_prompt() == "only-task"

    (tmp_path / "product_tables.yaml").write_text(
        "bookingsmaster:\n  - not-a-dict\n  - name: T\n    schema: S\nother: 1\n",
        encoding="utf-8",
    )
    cfg._load_yaml.cache_clear()
    tables = cfg.load_product_tables()
    assert tables["bookingsmaster"] == [{"name": "T", "schema": "S"}]
    assert "other" not in tables
    cfg._load_yaml.cache_clear()


def test_classification_block_formats_parts_and_omits_empty_fields() -> None:
    from deep_agent.aegra.classifier.classify import (
        ClassifyResult,
        IntentResult,
        SubQuery,
        TableInfo,
    )

    empty = ClassifyResult()
    assert empty.to_classification_block() == ""

    plain = ClassifyResult(sub_queries=[SubQuery(query="only")])
    plain_block = plain.to_classification_block()
    assert "unknown" in plain_block
    assert "product:" not in plain_block
    assert "tables:" not in plain_block
    assert "entities:" not in plain_block

    rich = ClassifyResult(
        is_compound=True,
        sub_queries=[
            SubQuery(
                query="bookings",
                intents=[IntentResult("data-query")],
                product="bookingsmaster",
                tables_list=[TableInfo(name="T", schema="MARTS")],
                entities=["Q3"],
            )
        ],
    )
    block = rich.to_classification_block()
    assert "compound_query: true (1 parts)" in block
    assert "product: bookingsmaster" in block
    assert "tables: T (MARTS)" in block
    assert "entities: Q3" in block


def test_parse_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from deep_agent.aegra.classifier.classify import (
        _lookup_tables,
        _parse_unified_response,
        classify_query,
    )

    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    earnings = asyncio.run(classify_query("show the earnings call"))
    assert earnings.intents[0].intent_class == "unstructured"
    default = asyncio.run(classify_query("total acv please"))
    assert default.intents[0].intent_class == "data-query"
    blank = asyncio.run(classify_query("   "))
    assert blank.intents[0].intent_class == "data-query"

    assert _lookup_tables("unknown-product") == []
    assert _lookup_tables("bookings_master")[0].name == "EOQ_TRANSACTION_ACV_SNAPSHOT"
    assert _parse_unified_response("not json", "q") == ([], [])
    assert _parse_unified_response("[1, 2]", "q") == ([], [])

    entities, subs = _parse_unified_response(
        '{"entities":[{"text":"Q3","type":"time_period"}],"sub_queries":[]}',
        "original",
    )
    assert entities[0].text == "Q3"
    assert subs == []

    _entities, clamped = _parse_unified_response(
        '{"sub_queries":[{"confidence":250,"product":"bookingsmaster"}]}',
        "original",
    )
    assert clamped[0].query == "original"
    assert clamped[0].confidence == 1.0
    assert clamped[0].intents[0].intent_class == "data-query"
    assert clamped[0].tables_list

    _entities, low = _parse_unified_response(
        '{"sub_queries":[{"confidence":-20,"intents":["nope"],'
        '"product":"bookingsmaster"}]}',
        "q",
    )
    assert low[0].confidence == 0.0
    assert low[0].intents[0].intent_class == "data-query"

    _entities, catalog = _parse_unified_response(
        '{"sub_queries":[{"intents":["catalog"],"product":"bookingsmaster",'
        '"confidence":10}]}',
        "q",
    )
    assert catalog[0].tables_list == []


@pytest.mark.asyncio
async def test_classify_query_uses_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "test-key")
    raw = (
        '{"entities":['
        '{"text":"Q3","type":"time_period"},'
        '{"text":"  ","type":"metric"},'
        '{"text":"nope","type":"not-a-type"},'
        '"bad"'
        "],"
        '"sub_queries":['
        '{"query":"total bookings","intents":["data-query","not-real"],'
        '"product":"Bookings Master","confidence":90,"entities":["Q3"],'
        '"reasoning":"bookings product"},'
        '{"query":"earnings call notes","intents":["unstructured"],'
        '"confidence":40,"entities":[],"reasoning":"docs"}'
        "]}"
    )

    async def _fake(query: str, system_prompt: str, settings: object) -> str:
        assert query == "total bookings and earnings"
        assert system_prompt
        assert getattr(settings, "api_key") == "test-key"
        return raw

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_fake):
        result = await classify_query("total bookings and earnings")

    assert result.is_compound is True
    assert result.routing is not None
    assert result.routing.decision == "clear_winner"
    assert result.routing.selected_product == "Bookings Master"
    thought = result.to_thought()
    assert "Source: model" in thought
    assert "Bookings Master" in thought
    assert "EOQ_TRANSACTION_ACV_SNAPSHOT" in thought
    assert {item.intent_class for item in result.intents} == {
        "data-query",
        "unstructured",
    }
    assert result.sub_queries[0].tables_list[0].name == "EOQ_TRANSACTION_ACV_SNAPSHOT"
    assert any(entity.text == "Q3" for entity in result.entities)
    block = result.to_classification_block()
    assert "compound_query: true (2 parts)" in block
    assert "EOQ_TRANSACTION_ACV_SNAPSHOT (MARTS)" in block


@pytest.mark.asyncio
async def test_classify_query_marks_low_confidence_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "test-key")
    raw = (
        '{"sub_queries":[{"query":"bookings","intents":["data-query"],'
        '"product":"bookingsmaster","confidence":50,"reasoning":"maybe"}]}'
    )

    async def _fake(**_kwargs: object) -> str:
        return raw

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_fake):
        result = await classify_query("bookings")

    assert result.routing is not None
    assert result.routing.decision == "ambiguous"
    assert result.is_compound is False


@pytest.mark.asyncio
async def test_classify_query_omits_routing_without_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "test-key")
    raw = '{"sub_queries":[{"query":"what tables exist","intents":["catalog"]}]}'

    async def _fake(**_kwargs: object) -> str:
        return raw

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_fake):
        result = await classify_query("what tables exist")

    assert result.routing is None
    assert result.sub_queries[0].tables_list == []
    assert result.intents[0].intent_class == "catalog"


@pytest.mark.asyncio
async def test_classify_query_falls_back_when_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deep_agent.aegra.classifier.classify import _call_maas

    monkeypatch.setenv("MAAS_API_KEY", "test-key")

    async def _boom(**_kwargs: object) -> str:
        raise RuntimeError("down")

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_boom):
        failed = await classify_query("total acv please")
    assert failed.reasoning.startswith("Fallback")

    async def _empty(**_kwargs: object) -> None:
        return None

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_empty):
        empty = await classify_query("total acv please")
    assert empty.reasoning.startswith("Fallback")

    async def _no_subs(**_kwargs: object) -> str:
        return '{"entities":[{"text":"Q3","type":"time_period"}],"sub_queries":[]}'

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_no_subs):
        unparsed = await classify_query("total acv please")
    assert unparsed.reasoning.startswith("Fallback")

    async def _garbage(**_kwargs: object) -> str:
        return "not json"

    with patch("deep_agent.aegra.classifier.classify.stream_completion", new=_garbage):
        garbage = await classify_query("total acv please")
    assert garbage.reasoning.startswith("Fallback")

    with patch(
        "deep_agent.aegra.classifier.classify.load_system_prompt", return_value=""
    ):
        assert await _call_maas("bookings") is None


def test_json_repair_handles_fences_and_truncation() -> None:
    from deep_agent.aegra.classifier.json_repair import (
        fix_missing_outer_brace,
        repair_truncated_json,
        safe_parse_json,
        strip_code_fences,
    )

    fenced = '```json\n{"a": 1}\n```'
    assert strip_code_fences(fenced) == '{"a": 1}'
    assert strip_code_fences("  plain  ") == "plain"
    assert safe_parse_json(fenced) == {"a": 1}
    assert safe_parse_json('"a": 1}') == {"a": 1}
    assert safe_parse_json('{"a": {"b": 1}') == {"a": {"b": 1}}
    assert safe_parse_json("[1, 2]") == [1, 2]
    assert safe_parse_json("5") is None
    assert safe_parse_json("not json") is None
    assert safe_parse_json('{"a": "hi') is None
    assert repair_truncated_json('{"a": 1}') is None
    assert repair_truncated_json("[1, 2") == "[1, 2]"
    assert repair_truncated_json('{"a": "x\\"y"}') is None
    assert fix_missing_outer_brace('{"a": 1}') == '{"a": 1}'
    assert "a" in (repair_truncated_json('{"a": "x\\"') or "")


class _Response:
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        lines: list[str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._lines = lines or []
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return self._body

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple] = []

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def stream(self, method: str, url: str, json: dict | None = None, headers=None):
        self.calls.append((method, url, json, headers))
        return self.response


def _settings(**overrides: object):
    from deep_agent.aegra.classifier.config import MaasSettings

    values = {
        "api_key": "k",
        "base_url": "https://models.example/v1",
        "model_id": "m",
        "timeout": 5.0,
        "verify_ssl": False,
        "max_tokens": 16,
    }
    values.update(overrides)
    return MaasSettings(**values)


@pytest.mark.asyncio
async def test_stream_completion_reads_sse_and_json() -> None:
    from deep_agent.aegra.classifier.maas_client import (
        _delta_text,
        _from_json_body,
        _join_stream,
        consume_sse_line,
        stream_completion,
    )

    assert _delta_text({"delta": "nope"}) == ("", "")
    assert _delta_text({"message": {"content": "a", "reasoning": "b"}}) == ("a", "b")
    assert _delta_text({}) == ("", "")
    assert _join_stream(["a"], ["b"]) == "a"
    assert _join_stream(["  "], ["b"]) == "b"
    assert _from_json_body("not json") == "not json"
    assert _from_json_body('{"choices": []}') == ""
    assert _from_json_body('{"choices": ["x"]}') == ""
    assert _from_json_body('{"choices":[{"delta":{"reasoning":"think"}}]}') == "think"
    assert _from_json_body('{"choices":[{"message":{"content":" body "}}]}') == "body"

    content: list[str] = []
    reasoning: list[str] = []
    assert consume_sse_line("event: ping", content, reasoning) is False
    assert consume_sse_line("data: {", content, reasoning) is False
    assert consume_sse_line('data: {"choices":[]}', content, reasoning) is False
    assert consume_sse_line('data: {"choices":["x"]}', content, reasoning) is False
    assert (
        consume_sse_line(
            'data: {"choices":[{"delta":{"reasoning":"why"}}]}',
            content,
            reasoning,
        )
        is False
    )
    assert reasoning == ["why"]

    holder: dict[str, _Client] = {}

    def _factory(*_args: object, **_kwargs: object) -> _Client:
        client = _Client(holder["response"])
        holder["client"] = client
        return client

    async def _run(response: _Response) -> str | None:
        holder["response"] = response
        with patch(
            "deep_agent.aegra.classifier.maas_client.httpx.AsyncClient",
            side_effect=_factory,
        ):
            return await stream_completion(
                query="q",
                system_prompt="sys",
                settings=_settings(),
            )

    sse = await _run(
        _Response(
            200,
            {"content-type": "text/event-stream"},
            lines=[
                "",
                'data: {"choices":[{"delta":{"content":"{\\"ok\\":true}"}}]}',
                "data: [DONE]",
            ],
        )
    )
    assert sse == '{"ok":true}'
    assert holder["client"].calls[0][0] == "POST"

    thought = await _run(
        _Response(
            202,
            {"content-type": "text/event-stream"},
            lines=['data: {"choices":[{"delta":{"reasoning":"because"}}]}'],
        )
    )
    assert thought == "because"

    body = await _run(
        _Response(
            200,
            {"content-type": "application/json"},
            body=b'{"choices":[{"message":{"content":"plain"}}]}',
        )
    )
    assert body == "plain"

    empty_body = await _run(
        _Response(200, {"content-type": "application/json"}, body=b'{"choices":[]}')
    )
    assert empty_body is None

    missing = await _run(_Response(404, {"content-type": "text/html"}, body=b"missing"))
    assert missing is None

    assert (
        await stream_completion(
            query="q", system_prompt="sys", settings=_settings(api_key="")
        )
        is None
    )
    assert (
        await stream_completion(query="q", system_prompt="  ", settings=_settings())
        is None
    )


@pytest.mark.asyncio
async def test_runnable_skips_injects_and_delegates() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from deep_agent.aegra.classifier.runnable import _inject_classification

    class _Inner:
        marker = "kept"

        def __init__(self) -> None:
            self.invoke_args = None
            self.event_args = None

        def copy(self, **kwargs):
            self.copied = kwargs
            return _Inner()

        def with_config(self, config=None, **kwargs):
            self.configured = (config, kwargs)
            return _Inner()

        async def ainvoke(self, data, config=None, **kwargs):
            self.invoke_args = (data, config, kwargs)
            return "invoked"

        async def astream(self, data, config=None, **kwargs):
            yield "s"

        async def astream_events(self, data, config=None, **kwargs):
            self.event_args = data
            yield "e"

    inner = _Inner()
    wrapper = ClassifierRunnable(inner)
    copied = wrapper.copy(update=1)
    assert isinstance(copied, ClassifierRunnable)
    assert wrapper.with_config({"tags": ["t"]}, x=1)
    assert wrapper.with_config(x=2)
    assert wrapper.marker == "kept"
    assert _inject_classification({"messages": []}, "block") == {"messages": []}

    assert await wrapper.ainvoke("not-a-dict") == "invoked"
    assert await wrapper.ainvoke({"messages": []}) == "invoked"
    assert (
        await wrapper.ainvoke({"messages": [{"role": "assistant", "content": "hi"}]})
        == "invoked"
    )
    assert (
        await wrapper.ainvoke({"messages": [{"role": "user", "content": "   "}]})
        == "invoked"
    )
    assert (
        await wrapper.ainvoke({"messages": [{"role": "user", "content": ["list"]}]})
        == "invoked"
    )
    assert await wrapper.ainvoke({"messages": [AIMessage(content="ai")]}) == "invoked"

    async def _unused(_query: str) -> None:
        raise AssertionError("classification should be skipped")

    with patch("deep_agent.aegra.classifier.runnable.classify_query", new=_unused):
        chunks = [
            chunk
            async for chunk in wrapper.astream(
                {"messages": [HumanMessage(content="<classification>\n\nalready")]},
                stream_mode=["messages"],
            )
        ]
    assert chunks[1] == "s"
    assert "already present" in chunks[0][1][0].content[0]["text"]

    class _Empty:
        sub_queries: list = []
        intents: list = []

        def to_classification_block(self) -> str:
            return ""

    async def _empty(_query: str) -> _Empty:
        return _Empty()

    with patch("deep_agent.aegra.classifier.runnable.classify_query", new=_empty):
        events = [
            event
            async for event in wrapper.astream_events(
                {"messages": [{"role": "human", "content": "q"}]},
                config={"c": 1},
                stream_mode=["messages"],
            )
        ]
    assert events[1] == "e"
    assert "Intent classification" in events[0][1][0].content[0]["text"]
    assert inner.event_args["messages"][-1]["content"] == "q"

    async def _boom(_query: str) -> None:
        raise RuntimeError("x")

    with patch("deep_agent.aegra.classifier.runnable.classify_query", new=_boom):
        failed = await wrapper.ainvoke(
            {"messages": [{"role": "user", "content": "q"}]},
            config=None,
            extra=1,
        )
    assert failed == "invoked"
    assert inner.invoke_args[0]["messages"][-1]["content"] == "q"

    class _Ok:
        sub_queries = [object()]
        intents: list = []

        def to_classification_block(self) -> str:
            return "<classification>\n  intent: data-query\n</classification>"

    async def _ok(_query: str) -> _Ok:
        return _Ok()

    with patch("deep_agent.aegra.classifier.runnable.classify_query", new=_ok):
        await wrapper.ainvoke({"messages": [HumanMessage(content="bookings")]})
    injected = inner.invoke_args[0]["messages"][-1]
    assert isinstance(injected, HumanMessage)
    assert str(injected.content).startswith("<classification>")
