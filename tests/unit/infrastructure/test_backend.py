"""Unit tests for backend module."""

import inspect
import os
from unittest.mock import MagicMock, patch

from deepagents.backends.protocol import EditResult, ReadResult, WriteResult

from deep_agent.src.infrastructure.backend import (
    DeduplicatingStoreBackend,
    _STORE_NAMESPACE_FACTORIES,
    _as_runtime,
    _backend_accepts_runtime,
    _base_python,
    _build_env,
    _get_assistant_id_from_config,
    _get_user_id_from_runtime,
    _make_state_backend,
    _make_store_backend,
    _safe_namespace_assistant,
    _safe_namespace_org,
    _safe_namespace_user,
    _text_from_read,
)


class TestBasePython:
    def test_returns_string(self):
        result = _base_python()
        assert isinstance(result, str)
        assert "python" in result.lower()


class TestBuildEnv:
    def test_contains_virtual_env(self, tmp_path):
        env = _build_env(tmp_path)
        assert env["VIRTUAL_ENV"] == str(tmp_path)

    def test_contains_path(self, tmp_path):
        env = _build_env(tmp_path)
        assert str(tmp_path) in env["PATH"]

    def test_extra_env_overrides(self, tmp_path):
        env = _build_env(tmp_path, extra={"MY_VAR": "my_val"})
        assert env["MY_VAR"] == "my_val"

    def test_passthrough_vars(self, tmp_path):
        with patch.dict(os.environ, {"HOME": "/test/home", "USER": "tester"}):
            env = _build_env(tmp_path)
            assert env.get("HOME") == "/test/home"
            assert env.get("USER") == "tester"


def _make_ctx(server_info=None, config=None, context=None):
    """Build a minimal ctx mock for namespace tests."""
    ctx = MagicMock()
    ctx.runtime.server_info = server_info
    ctx.runtime.config = config
    if context is not None:
        ctx.runtime.context = context
    return ctx


def _make_server_info(assistant_id="", user_identity=""):
    """Build a server_info mock with optional assistant_id and user."""
    si = MagicMock()
    si.assistant_id = assistant_id
    if user_identity:
        si.user = MagicMock()
        si.user.identity = user_identity
    else:
        si.user = None
    return si


class TestGetAssistantIdFromConfig:
    """Tests for _get_assistant_id_from_config."""

    def test_returns_assistant_id_from_metadata(self):
        ctx = _make_ctx(config={"metadata": {"assistant_id": "agent-42"}})
        assert _get_assistant_id_from_config(ctx) == "agent-42"

    def test_returns_default_when_config_is_none(self):
        ctx = _make_ctx(config=None)
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_config_missing_metadata(self):
        ctx = _make_ctx(config={"other_key": "val"})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_has_no_assistant_id(self):
        ctx = _make_ctx(config={"metadata": {}})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_assistant_id_is_empty_string(self):
        ctx = _make_ctx(config={"metadata": {"assistant_id": ""}})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_config_is_not_dict(self):
        ctx = _make_ctx(config="not-a-dict")
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_runtime_has_no_config_attr(self):
        ctx = MagicMock()
        del ctx.runtime.config
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_is_none(self):
        ctx = _make_ctx(config={"metadata": None})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_is_not_dict(self):
        ctx = _make_ctx(config={"metadata": ["not", "a", "dict"]})
        assert _get_assistant_id_from_config(ctx) == "default"


class TestSafeNamespaceUser:
    """Tests for _safe_namespace_user."""

    def test_returns_assistant_id_and_user_identity_from_server_info(self):
        si = _make_server_info(assistant_id="asst-1", user_identity="user@example.com")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1", "user@example.com")

    def test_returns_only_assistant_id_when_user_is_none(self):
        si = _make_server_info(assistant_id="asst-1", user_identity="")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_returns_only_assistant_id_when_user_identity_empty(self):
        si = MagicMock()
        si.assistant_id = "asst-1"
        si.user = MagicMock()
        si.user.identity = ""
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_returns_only_assistant_id_when_server_info_has_no_user_attr(self):
        si = MagicMock(spec=[])
        si.assistant_id = "asst-1"
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_falls_back_to_config_when_server_info_is_none(self):
        ctx = _make_ctx(
            server_info=None,
            config={"metadata": {"assistant_id": "cfg-agent"}},
        )
        assert _safe_namespace_user(ctx) == ("cfg-agent",)

    def test_falls_back_to_config_when_assistant_id_empty(self):
        si = _make_server_info(assistant_id="", user_identity="user@x.com")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"assistant_id": "cfg-agent"}},
        )
        assert _safe_namespace_user(ctx) == ("cfg-agent", "user@x.com")

    def test_falls_back_to_default_when_no_server_info_and_no_config(self):
        ctx = _make_ctx(server_info=None, config=None)
        assert _safe_namespace_user(ctx) == ("default",)

    def test_appends_metadata_user_id_when_no_server_info(self):
        ctx = _make_ctx(
            server_info=None,
            config={"metadata": {"assistant_id": "cfg-agent", "user_id": "johnwick"}},
        )
        assert _safe_namespace_user(ctx) == ("cfg-agent", "johnwick")

    def test_appends_configurable_user_id_when_no_server_info(self):
        ctx = _make_ctx(
            server_info=None,
            config={"configurable": {"user_id": "johnwick"}},
        )
        assert _safe_namespace_user(ctx) == ("default", "johnwick")

    def test_prefers_metadata_user_id_over_server_info_identity(self):
        si = _make_server_info(assistant_id="asst-1", user_identity="jwt-sub")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"user_id": "johnwick"}},
        )
        assert _safe_namespace_user(ctx) == ("asst-1", "johnwick")

    def test_uses_preferred_username_from_access_token_when_no_metadata(self):
        import base64
        import json

        payload = (
            base64.urlsafe_b64encode(
                json.dumps({"preferred_username": "dpundir", "sub": "uuid-1"}).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        si = _make_server_info(assistant_id="asst-1", user_identity="uuid-1")
        si.user.access_token = f"h.{payload}.s"
        ctx = _make_ctx(server_info=si, config={})
        assert _safe_namespace_user(ctx) == ("asst-1", "dpundir")
        si = _make_server_info(assistant_id="asst-1", user_identity="")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"user_id": "johnwick"}},
        )
        assert _safe_namespace_user(ctx) == ("asst-1", "johnwick")

    def test_accepts_langgraph_runtime_without_nested_runtime_attr(self):
        """StoreBackend 0.7 passes Runtime, not a wrapper with ``.runtime``."""

        class Runtime:
            pass

        rt = Runtime()
        rt.server_info = _make_server_info(
            assistant_id="asst-rt", user_identity="jwt-sub"
        )
        rt.config = {}
        assert _safe_namespace_user(rt) == ("asst-rt", "jwt-sub")


class TestGetUserIdFromRuntime:
    def test_reads_metadata(self):
        runtime = MagicMock()
        runtime.config = {"metadata": {"user_id": "alice"}}
        assert _get_user_id_from_runtime(runtime) == "alice"

    def test_reads_configurable(self):
        runtime = MagicMock()
        runtime.config = {"configurable": {"user_id": "bob"}}
        assert _get_user_id_from_runtime(runtime) == "bob"

    def test_returns_none_when_missing(self):
        runtime = MagicMock()
        runtime.config = {"metadata": {"trace_id": "t1"}}
        assert _get_user_id_from_runtime(runtime) is None


class TestSafeNamespaceAssistant:
    """Tests for _safe_namespace_assistant."""

    def test_returns_assistant_id_from_server_info(self):
        si = _make_server_info(assistant_id="asst-2", user_identity="ignored")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_assistant(ctx) == ("asst-2",)

    def test_falls_back_to_config_when_server_info_is_none(self):
        ctx = _make_ctx(
            server_info=None,
            config={"metadata": {"assistant_id": "cfg-asst"}},
        )
        assert _safe_namespace_assistant(ctx) == ("cfg-asst",)

    def test_falls_back_to_config_when_assistant_id_empty(self):
        si = _make_server_info(assistant_id="")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"assistant_id": "cfg-asst"}},
        )
        assert _safe_namespace_assistant(ctx) == ("cfg-asst",)

    def test_falls_back_to_default_when_no_config(self):
        ctx = _make_ctx(server_info=None, config=None)
        assert _safe_namespace_assistant(ctx) == ("default",)


class TestSafeNamespaceOrg:
    """Tests for _safe_namespace_org."""

    def test_returns_org_id(self):
        context = MagicMock()
        context.org_id = "org-123"
        ctx = _make_ctx(context=context)
        assert _safe_namespace_org(ctx) == ("org-123",)


class TestStoreNamespaceFactories:
    """Tests for the _STORE_NAMESPACE_FACTORIES mapping."""

    def test_contains_all_expected_keys(self):
        assert set(_STORE_NAMESPACE_FACTORIES.keys()) == {"user", "assistant", "org"}

    def test_user_maps_to_safe_namespace_user(self):
        assert _STORE_NAMESPACE_FACTORIES["user"] is _safe_namespace_user

    def test_assistant_maps_to_safe_namespace_assistant(self):
        assert _STORE_NAMESPACE_FACTORIES["assistant"] is _safe_namespace_assistant

    def test_org_maps_to_safe_namespace_org(self):
        assert _STORE_NAMESPACE_FACTORIES["org"] is _safe_namespace_org


class TestBackendConstructorHelpers:
    """StateBackend/StoreBackend construction across deepagents versions."""

    def test_make_state_backend_returns_instance(self):
        assert _make_state_backend(MagicMock()) is not None

    def test_make_store_backend_returns_instance(self):
        def fake_namespace(ctx: object) -> tuple[str, ...]:
            return ("test",)

        assert _make_store_backend(MagicMock(), fake_namespace) is not None

    def test_accepts_runtime_matches_constructor(self):
        from deepagents.backends.state import StateBackend

        expected = "runtime" in inspect.signature(StateBackend).parameters
        assert _backend_accepts_runtime(StateBackend) is expected

    def test_accepts_runtime_false_without_runtime_param(self):
        class NoRuntime:
            def __init__(self) -> None:
                pass

        assert _backend_accepts_runtime(NoRuntime) is False

    def test_accepts_runtime_false_when_signature_unavailable(self):
        assert _backend_accepts_runtime(42) is False


class _FakeStoreInner:
    """Minimal backend that returns ReadResult from read/aread."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.writes: list[str] = []
        self.read_error: str | None = None

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        self.content = self.content.replace(old_string, new_string)
        return EditResult(path=file_path, occurrences=1)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        return self.edit(file_path, old_string, new_string, replace_all)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        if self.read_error:
            return ReadResult(error=self.read_error)
        return ReadResult(file_data={"content": self.content, "encoding": "utf-8"})

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        return self.read(file_path, offset, limit)

    def write(self, file_path: str, content: str):
        self.content = content
        self.writes.append(content)
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str):
        return self.write(file_path, content)


class TestTextFromRead:
    def test_returns_string_unchanged(self):
        assert _text_from_read("hello") == "hello"

    def test_extracts_file_data_content(self):
        result = ReadResult(
            file_data={
                "content": "The user's date of birth is June 12.\n",
                "encoding": "utf-8",
            }
        )
        assert _text_from_read(result) == "The user's date of birth is June 12.\n"

    def test_returns_none_on_error(self):
        assert _text_from_read(ReadResult(error="File not found")) is None

    def test_returns_none_for_none(self):
        assert _text_from_read(None) is None

    def test_returns_none_for_unknown_object(self):
        assert _text_from_read(object()) is None


class TestDeduplicatingStoreBackendEdit:
    """edit_file must not crash when the inner backend returns ReadResult."""

    def test_edit_replaces_single_fact_without_crashing(self):
        inner = _FakeStoreInner("The user's date of birth is June 12.\n")
        backend = DeduplicatingStoreBackend(inner)
        result = backend.edit(
            "user_profile.md",
            "The user's date of birth is June 12.",
            "The user's date of birth is June 30.",
        )
        assert result.error is None
        assert "June 30" in inner.content
        assert inner.writes == []

    async def test_aedit_replaces_single_fact_without_crashing(self):
        inner = _FakeStoreInner("The user's date of birth is June 12.\n")
        backend = DeduplicatingStoreBackend(inner)
        result = await backend.aedit(
            "user_profile.md",
            "The user's date of birth is June 12.",
            "The user's date of birth is June 30.",
        )
        assert result.error is None
        assert "June 30" in inner.content
        assert inner.writes == []

    async def test_aedit_returns_edit_result_when_post_read_fails(self):
        inner = _FakeStoreInner("The user's date of birth is June 12.\n")
        inner.read_error = "boom"
        backend = DeduplicatingStoreBackend(inner)
        result = await backend.aedit(
            "user_profile.md",
            "The user's date of birth is June 12.",
            "The user's date of birth is June 30.",
        )
        assert result.error is None
        assert "June 30" in inner.content


class TestAsRuntime:
    def test_returns_none_for_none(self):
        assert _as_runtime(None) is None

    def test_unwraps_nested_runtime(self):
        inner = MagicMock(name="runtime")
        ctx = MagicMock()
        ctx.runtime = inner
        assert _as_runtime(ctx) is inner

    def test_returns_ctx_when_no_nested_runtime(self):
        ctx = object()
        assert _as_runtime(ctx) is ctx


class TestDeduplicatingStoreBackendWrite:
    def test_write_keeps_single_fact(self):
        inner = _FakeStoreInner("")
        backend = DeduplicatingStoreBackend(inner)
        backend.write("user_profile.md", "- user likes python\n")
        assert inner.writes[-1] == "- user likes python\n"

    def test_write_drops_near_duplicate_facts(self):
        inner = _FakeStoreInner("")
        backend = DeduplicatingStoreBackend(inner)
        backend.write(
            "user_profile.md",
            "user weighs 70kg\nuser weight is 70 kg\nlikes python programming\n",
        )
        written = inner.writes[-1]
        assert written.count("70") == 1

    async def test_awrite_deduplicates(self):
        inner = _FakeStoreInner("")
        backend = DeduplicatingStoreBackend(inner)
        await backend.awrite(
            "user_profile.md",
            "user weighs 70kg\nuser weight is 70 kg\n",
        )
        assert inner.writes

    def test_write_keeps_birth_and_joining_dates(self):
        inner = _FakeStoreInner("")
        backend = DeduplicatingStoreBackend(inner)
        backend.write(
            "user_profile.md",
            "- The user's date of birth is June 11, 2003.\n"
            "- The user's joining date is 2 June 2020.\n",
        )
        written = inner.writes[-1]
        assert "date of birth is June 11, 2003" in written
        assert "joining date is 2 June 2020" in written

    def test_write_keeps_incomplete_joining_line_with_birth(self):
        inner = _FakeStoreInner("")
        backend = DeduplicatingStoreBackend(inner)
        backend.write(
            "user_profile.md",
            "- The user's date of birth is June 11, 2003.\n"
            "- The user's joining date is\n",
        )
        written = inner.writes[-1]
        assert "date of birth" in written
        assert "joining date" in written
