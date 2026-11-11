"""Unit tests for the Daytona sandbox backend.

The point of most of these is isolation: the backend serves every
conversation on the pod from one object, so the tests drive it as two
users at once and assert that nothing crosses over.

`langchain_daytona` is stubbed before import. It is a real dependency in
production, but these tests are about this module's session bookkeeping,
not the SDK's, and stubbing keeps the unit suite installable without a
sandbox provider.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# -- stub langchain_daytona before the module under test imports it --------

if "langchain_daytona" not in sys.modules:  # pragma: no cover - import shim
    _stub = types.ModuleType("langchain_daytona")

    class _StubDaytonaSandbox:
        def __init__(self, sandbox=None, timeout=None):
            self._sandbox = sandbox
            self._timeout = timeout

        def execute(self, command, *, timeout=None):
            """Resolve the sandbox the way the real base class does."""
            from deepagents.backends.protocol import ExecuteResponse

            self._sandbox.ran = command
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    _stub.DaytonaSandbox = _StubDaytonaSandbox
    sys.modules["langchain_daytona"] = _stub

from deep_agent.src.agent import daytona_backend as db  # noqa: E402

MODULE = "deep_agent.src.agent.daytona_backend"


class FakeSandbox:
    """Stand-in for a Daytona sandbox, recording its own deletion."""

    def __init__(self, image: str, labels: dict[str, str]):
        self.id = f"sb-{labels.get('session')}-{labels.get('runtime')}"
        self.image = image
        self.labels = labels
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True


class FakeClient:
    """Stand-in for the Daytona client; hands out FakeSandboxes."""

    def __init__(self) -> None:
        self.created: list[FakeSandbox] = []

    def create(self, params, timeout=None):  # noqa: ARG002 - mirrors the SDK
        sandbox = FakeSandbox(params["image"], params["labels"])
        self.created.append(sandbox)
        return sandbox


def _fake_params(runtime: str, key: str) -> dict:
    """Replace `_create_params`, which would import the real SDK."""
    return {
        "image": db.RUNTIMES[runtime],
        "labels": {"app": "template-agent", "runtime": runtime, "session": key},
    }


@contextmanager
def as_user(user: str, thread: str):
    """Run the block as a given user and thread, the way a run would."""
    with (
        patch(f"{MODULE}.resolve_user_id", return_value=user),
        patch(f"{MODULE}.resolve_thread_id", return_value=thread),
    ):
        yield


@pytest.fixture(autouse=True)
def _clean_registry():
    """No session state crosses between tests."""
    db._sessions.clear()
    yield
    db._sessions.clear()


@pytest.fixture
def client():
    """A fake Daytona client, so no test reaches the provider."""
    with (
        patch(f"{MODULE}._create_params", side_effect=_fake_params),
        patch(f"{MODULE}._client") as get_client,
    ):
        fake = FakeClient()
        get_client.return_value = fake
        yield fake


@pytest.fixture
def backend(client):  # noqa: ARG001 - ordering fixture
    """The backend object, as the graph would build it: one for everyone."""
    return db.DaytonaRuntimeSandbox()


# -- session key -----------------------------------------------------------


class TestSessionKey:
    def test_combines_user_and_thread(self):
        with as_user("alice", "t-1"):
            assert db.session_key() == "alice/t-1"

    def test_unauthenticated_run_still_separates_by_thread(self):
        """No user id is fine: the thread id is what isolation needs."""
        with as_user(None, "t-1"):
            assert db.session_key() == "anon/t-1"

    def test_refuses_when_no_thread_id_resolves(self):
        """The failure mode that matters: never silently share a sandbox."""
        with as_user("alice", None), pytest.raises(db.SessionUnavailable):
            db.session_key()

    def test_anonymous_session_can_be_opted_into(self):
        with as_user(None, None), patch(f"{MODULE}.ALLOW_ANONYMOUS_SESSION", True):
            assert db.session_key() == "anon/local"

    def test_different_threads_are_different_sessions(self):
        with as_user("alice", "t-1"):
            first = db.session_key()
        with as_user("alice", "t-2"):
            second = db.session_key()
        assert first != second


# -- isolation -------------------------------------------------------------


class TestSandboxIsolation:
    def test_two_users_get_two_sandboxes(self, backend, client):
        with as_user("alice", "t-1"):
            alice = backend._sandbox
        with as_user("bob", "t-2"):
            bob = backend._sandbox
        assert alice is not bob
        assert len(client.created) == 2

    def test_same_conversation_reuses_one_sandbox(self, backend, client):
        with as_user("alice", "t-1"):
            first = backend._sandbox
            second = backend._sandbox
        assert first is second
        assert len(client.created) == 1

    def test_sandbox_is_labelled_with_its_session(self, backend):
        with as_user("alice", "t-1"):
            sandbox = backend._sandbox
        assert sandbox.labels["session"] == "alice/t-1"


# -- runtime selection -----------------------------------------------------


class TestUseRuntime:
    def test_switch_does_not_move_another_conversation(self, backend):
        with as_user("alice", "t-1"):
            backend._sandbox  # noqa: B018 - create alice's python sandbox
        with as_user("bob", "t-2"):
            db.daytona_use_runtime.invoke({"name": "node"})
            assert db.active_runtime() == "node"
        with as_user("alice", "t-1"):
            assert db.active_runtime() == "python"

    def test_switch_gives_this_conversation_a_new_filesystem(self, backend):
        with as_user("alice", "t-1"):
            python_box = backend._sandbox
            db.daytona_use_runtime.invoke({"name": "node"})
            node_box = backend._sandbox
        assert python_box is not node_box
        assert node_box.labels["runtime"] == "node"

    def test_unknown_runtime_is_rejected(self):
        with as_user("alice", "t-1"):
            out = db.daytona_use_runtime.invoke({"name": "rust"})
        assert out.startswith("error: unknown runtime")
        assert db._sessions == {}

    def test_runtimes_without_python3_are_flagged(self):
        with as_user("alice", "t-1"):
            out = db.daytona_use_runtime.invoke({"name": "go"})
        assert "no python3" in out

    def test_refuses_an_unattributable_call(self):
        with as_user("alice", None):
            out = db.daytona_use_runtime.invoke({"name": "node"})
        assert "no thread id" in out
        assert db._sessions == {}


# -- reset -----------------------------------------------------------------


class TestReset:
    def test_reset_leaves_other_conversations_alone(self, backend):
        with as_user("alice", "t-1"):
            alice = backend._sandbox
        with as_user("bob", "t-2"):
            bob = backend._sandbox
            db.daytona_reset_sandbox.invoke({})
        assert bob.deleted is True
        assert alice.deleted is False
        with as_user("alice", "t-1"):
            assert backend._sandbox is alice

    def test_reset_then_next_command_builds_a_fresh_one(self, backend):
        with as_user("alice", "t-1"):
            first = backend._sandbox
            db.daytona_reset_sandbox.invoke({})
            second = backend._sandbox
        assert first is not second
        assert first.deleted is True

    def test_reset_with_nothing_running_says_so(self, backend):
        with as_user("alice", "t-1"):
            out = db.daytona_reset_sandbox.invoke({})
        assert "no python sandbox was running" in out

    def test_reset_reports_a_failed_delete(self, backend):
        with as_user("alice", "t-1"):
            sandbox = backend._sandbox
            sandbox.delete = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
            out = db.daytona_reset_sandbox.invoke({})
        assert "could not delete" in out


# -- execute ---------------------------------------------------------------


class TestExecute:
    def test_runs_in_this_conversation_sandbox(self, backend):
        with as_user("alice", "t-1"):
            result = backend.execute("echo hi")
            assert backend._sandbox.ran == "echo hi"
        assert result.exit_code == 0

    def test_unattributable_call_is_an_error_not_a_traceback(self, backend):
        with as_user("alice", None):
            result = backend.execute("echo hi")
        assert result.exit_code == 127
        assert "no thread id" in result.output

    def test_missing_api_key_is_an_error_not_a_traceback(self, backend):
        with as_user("alice", "t-1"), patch(f"{MODULE}._client") as get_client:
            get_client.side_effect = db.DaytonaNotConfigured("DAYTONA_API_KEY is unset")
            result = backend.execute("echo hi")
        assert result.exit_code == 127
        assert "DAYTONA_API_KEY" in result.output


# -- lifecycle -------------------------------------------------------------


class TestLifecycle:
    def test_idle_sessions_are_deleted(self, backend):
        with as_user("alice", "t-1"):
            stale = backend._sandbox
        db._sessions["alice/t-1"].last_used -= db.IDLE_REAP_SECONDS + 1
        with as_user("bob", "t-2"):
            backend._sandbox  # noqa: B018 - triggers the reaper
        assert stale.deleted is True
        assert "alice/t-1" not in db._sessions

    def test_oldest_session_is_evicted_over_the_cap(self, backend):
        with patch(f"{MODULE}.MAX_SANDBOXES", 2):
            boxes = []
            for i in range(3):
                with as_user(f"u{i}", f"t-{i}"):
                    boxes.append(backend._sandbox)
        assert boxes[0].deleted is True
        assert boxes[2].deleted is False
        assert "u0/t-0" not in db._sessions

    def test_shutdown_deletes_everything(self, backend):
        with as_user("alice", "t-1"):
            alice = backend._sandbox
        with as_user("bob", "t-2"):
            bob = backend._sandbox
        db._shutdown()
        assert alice.deleted is True
        assert bob.deleted is True
        assert db._sessions == {}


# -- logging ---------------------------------------------------------------


class TestLogging:
    def test_commands_are_not_logged_by_default(self):
        with patch(f"{MODULE}.logger") as log:
            db._log("daytona_run_started", session="alice/t-1", command_bytes=42)
        _, kwargs = log.info.call_args
        assert "command" not in kwargs
        assert kwargs["session"] == "alice/t-1"

    def test_logging_never_breaks_a_run(self):
        with patch(f"{MODULE}.logger") as log:
            log.info.side_effect = RuntimeError("loki is down")
            db._log("daytona_run_started")  # must not raise


# -- startup posture -------------------------------------------------------


class TestStartupWarnings:
    """The two settings that are expensive to get wrong silently."""

    def test_saas_url_warns_that_data_leaves_the_cluster(self):
        with (
            patch(f"{MODULE}.API_KEY", "dtn_x"),
            patch(f"{MODULE}.API_URL", ""),
            patch(f"{MODULE}.logger") as log,
        ):
            db._warn_on_startup_posture()
        events = [c.args[0] for c in log.warning.call_args_list]
        assert "daytona_egress_leaves_cluster" in events

    def test_self_hosted_url_does_not_warn(self):
        with (
            patch(f"{MODULE}.API_KEY", "dtn_x"),
            patch(f"{MODULE}.API_URL", "https://daytona.svc.cluster.local/api"),
            patch(f"{MODULE}.logger") as log,
        ):
            db._warn_on_startup_posture()
        assert log.warning.call_args_list == []

    def test_unconfigured_is_silent(self):
        with patch(f"{MODULE}.API_KEY", ""), patch(f"{MODULE}.logger") as log:
            db._warn_on_startup_posture()
        assert log.warning.call_args_list == []

    def test_anonymous_sessions_warn(self):
        with (
            patch(f"{MODULE}.API_KEY", "dtn_x"),
            patch(f"{MODULE}.API_URL", "https://daytona.svc.cluster.local/api"),
            patch(f"{MODULE}.ALLOW_ANONYMOUS_SESSION", True),
            patch(f"{MODULE}.logger") as log,
        ):
            db._warn_on_startup_posture()
        events = [c.args[0] for c in log.warning.call_args_list]
        assert "daytona_anonymous_sessions_enabled" in events

    def test_command_logging_warns(self):
        with (
            patch(f"{MODULE}.API_KEY", "dtn_x"),
            patch(f"{MODULE}.API_URL", "https://daytona.svc.cluster.local/api"),
            patch(f"{MODULE}.LOG_COMMANDS", True),
            patch(f"{MODULE}.logger") as log,
        ):
            db._warn_on_startup_posture()
        events = [c.args[0] for c in log.warning.call_args_list]
        assert "daytona_command_logging_enabled" in events
