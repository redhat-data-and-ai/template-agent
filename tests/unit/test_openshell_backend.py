"""Tests for the OpenShell sandbox backend.

The property under test throughout is isolation: the compiled graph, and
so this backend, is a process-wide singleton shared by every request, and
sandboxes must not be. Most of the file is therefore about which caller
gets which sandbox, and about the lifecycle that keeps pods from leaking.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest

from deep_agent.src.agent import openshell_backend as osb

# ── Fakes ─────────────────────────────────────────────────────────


class FakeExecResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeSandboxHandle:
    """Stands in for an ``openshell`` session: an id plus ``exec``."""

    def __init__(self, name: str) -> None:
        self.sandbox = type("S", (), {"name": name})()
        self.id = name
        self.calls: list[str] = []
        self.dead = False

    def exec(self, command: Any, **kwargs: Any) -> FakeExecResult:
        self.calls.append(" ".join(command))
        if self.dead:
            raise RuntimeError("sandbox not found")
        return FakeExecResult(stdout=f"{self.id}:ok")


class FakeClient:
    """Records every sandbox the backend creates and deletes."""

    def __init__(self) -> None:
        self.created: list[FakeSandboxHandle] = []
        self.deleted: list[str] = []
        self.images: list[str] = []
        self.ready_calls = 0
        self._n = 0

    def create_session(self, **kwargs: Any) -> FakeSandboxHandle:
        spec = kwargs.get("spec")
        self.images.append(getattr(getattr(spec, "template", None), "image", None))
        self._n += 1
        handle = FakeSandboxHandle(f"sbx-{self._n}")
        self.created.append(handle)
        return handle

    def wait_ready(self, name: str, **kwargs: Any) -> None:
        self.ready_calls += 1

    def delete(self, name: str, **kwargs: Any) -> None:
        self.deleted.append(name)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Install a fake gateway and reset all process-wide state."""
    fake = FakeClient()
    monkeypatch.setattr(osb, "_client", lambda: fake)
    monkeypatch.setattr(osb, "_client_singleton", fake, raising=False)
    monkeypatch.setattr(osb, "is_configured", lambda: True)
    monkeypatch.setattr(osb, "_ensure_reaper", lambda: None)
    monkeypatch.setattr(osb, "_entries", {})
    monkeypatch.setattr(osb, "_runtime_choice", {})
    monkeypatch.setattr(osb, "_sandbox_singleton", None, raising=False)
    return fake


def as_caller(user: str | None = None, thread: str | None = None) -> Any:
    """Patch the run config the backend reads its caller identity from."""
    bag: dict[str, Any] = {}
    if user:
        bag["user_id"] = user
    if thread:
        bag["thread_id"] = thread
    return patch.object(osb, "_config_bags", lambda: [bag])


# ── Isolation ─────────────────────────────────────────────────────


def test_two_conversations_get_separate_sandboxes(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()

    with as_caller("alice", "t1"):
        first = sandbox.execute("whoami")
    with as_caller("alice", "t2"):
        second = sandbox.execute("whoami")

    assert len(client.created) == 2
    assert first.output != second.output


def test_two_users_never_share_a_sandbox(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()

    # Same thread id: a guessed or replayed id must still not cross users.
    with as_caller("alice", "shared-id"):
        sandbox.execute("whoami")
    with as_caller("bob", "shared-id"):
        sandbox.execute("whoami")

    assert len(client.created) == 2


def test_same_conversation_reuses_its_sandbox(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()

    with as_caller("alice", "t1"):
        sandbox.execute("echo one")
        sandbox.execute("echo two")

    assert len(client.created) == 1, "state must persist across calls"


def test_user_scope_shares_across_one_users_threads(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(osb.settings, "OPENSHELL_SESSION_SCOPE", "user")
    sandbox = osb.get_sandbox()

    with as_caller("alice", "t1"):
        sandbox.execute("echo one")
    with as_caller("alice", "t2"):
        sandbox.execute("echo two")
    with as_caller("bob", "t3"):
        sandbox.execute("echo three")

    assert len(client.created) == 2


def test_unidentified_caller_is_refused_by_default(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()

    with as_caller():
        result = sandbox.execute("echo hi")

    assert result.exit_code != 0
    assert "conversation" in result.output
    assert client.created == [], "no sandbox may be shared by every caller"


def test_unidentified_caller_allowed_when_opted_in(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(osb.settings, "OPENSHELL_ALLOW_SHARED_SANDBOX", True)
    sandbox = osb.get_sandbox()

    with as_caller():
        assert sandbox.execute("echo hi").exit_code == 0


# ── Runtime selection ─────────────────────────────────────────────


def test_runtime_choice_does_not_leak_between_conversations(
    client: FakeClient,
) -> None:
    with as_caller("alice", "t1"):
        osb.openshell_use_runtime.invoke({"name": "go"})
        assert osb._active_runtime(osb._owner_key()) == "go"

    with as_caller("bob", "t2"):
        assert osb._active_runtime(osb._owner_key()) == "python"


def test_unknown_runtime_is_rejected(client: FakeClient) -> None:
    with as_caller("alice", "t1"):
        result = osb.openshell_use_runtime.invoke({"name": "rust"})
    assert result.startswith("error:")


def test_runtimes_without_python3_are_flagged_to_the_model(
    client: FakeClient,
) -> None:
    with as_caller("alice", "t1"):
        assert "no python3" in osb.openshell_use_runtime.invoke({"name": "node"})
        assert "no python3" not in osb.openshell_use_runtime.invoke({"name": "python"})


def test_switching_runtime_uses_a_separate_sandbox(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo py")
        osb.openshell_use_runtime.invoke({"name": "node"})
        sandbox.execute("echo node")

    assert len(client.created) == 2
    assert client.images == [
        osb.RUNTIMES["python"],
        osb.RUNTIMES["node"],
    ], "each runtime must be created from its own image"


# ── Lifecycle ─────────────────────────────────────────────────────


def test_reset_only_destroys_the_callers_own_sandbox(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo a")
    with as_caller("bob", "t2"):
        sandbox.execute("echo b")

    with as_caller("alice", "t1"):
        sandbox.reset()

    assert client.deleted == ["sbx-1"]
    assert ("u.bob.t.t2", "python") in osb._entries


def test_reset_on_a_cold_conversation_is_not_an_error(client: FakeClient) -> None:
    with as_caller("alice", "t1"):
        assert "no python sandbox was running" in osb.openshell_reset_sandbox.invoke({})


def test_a_dead_sandbox_is_replaced_and_the_command_retried(
    client: FakeClient,
) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo one")
        client.created[0].dead = True
        result = sandbox.execute("echo two")

    assert len(client.created) == 2, "the evicted pod must be replaced"
    assert result.output == "sbx-2:ok"
    assert client.deleted == ["sbx-1"]


def test_an_ordinary_command_failure_does_not_destroy_the_sandbox(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo one")
        handle = client.created[0]
        monkeypatch.setattr(
            handle,
            "exec",
            lambda *a, **k: FakeExecResult(stderr="No such file", exit_code=1),
        )
        sandbox.execute("cat /nope")

    assert len(client.created) == 1, "a failing command is not a dead sandbox"
    assert client.deleted == []


def test_idle_sandboxes_are_reaped(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo one")

    monkeypatch.setattr(osb.settings, "OPENSHELL_SESSION_IDLE_TTL", 0)
    for entry in osb._reap_idle():
        osb._delete_remote(entry)

    assert client.deleted == ["sbx-1"]
    assert osb._entries == {}
    assert osb._runtime_choice == {}


def test_session_count_is_capped(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(osb.settings, "OPENSHELL_MAX_SESSIONS", 2)
    sandbox = osb.get_sandbox()
    for n in range(4):
        with as_caller("alice", f"t{n}"):
            sandbox.execute("echo hi")

    assert len(osb._entries) <= 2
    assert client.deleted, "the LRU sandbox must be deleted, not just forgotten"


def test_shutdown_deletes_every_sandbox(client: FakeClient) -> None:
    sandbox = osb.get_sandbox()
    with as_caller("alice", "t1"):
        sandbox.execute("echo a")
    with as_caller("bob", "t2"):
        sandbox.execute("echo b")

    osb.shutdown_all()

    assert sorted(client.deleted) == ["sbx-1", "sbx-2"]
    assert osb._entries == {}


def test_a_sandbox_that_never_becomes_ready_is_deleted(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_ready(*a: Any, **k: Any) -> None:
        raise TimeoutError("timed out waiting for sandbox")

    monkeypatch.setattr(client, "wait_ready", never_ready)
    sandbox = osb.get_sandbox()

    with as_caller("alice", "t1"), pytest.raises(TimeoutError):
        sandbox._current()

    assert client.deleted == ["sbx-1"], "a ready-timeout must not leak a pod"
    assert osb._entries == {}, "the failed entry must not be cached"


def test_concurrent_calls_create_one_sandbox_per_conversation(
    client: FakeClient,
) -> None:
    sandbox = osb.get_sandbox()
    barrier = threading.Barrier(8)

    def run() -> None:
        barrier.wait()
        with as_caller("alice", "t1"):
            sandbox.execute("echo hi")

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(client.created) == 1, "a lost creation race leaks a pod"


# ── Wiring ────────────────────────────────────────────────────────


def test_wrap_backend_is_a_no_op_without_a_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured sandbox must not take the agent's filesystem with it."""
    monkeypatch.setattr(osb, "is_configured", lambda: False)
    original = object()
    assert osb.wrap_backend(original) is original  # type: ignore[arg-type]


def test_wrap_backend_keeps_artifacts_off_the_sandbox(client: FakeClient) -> None:
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend

    state = StateBackend()
    wrapped = osb.wrap_backend(CompositeBackend(default=state, routes={}))

    assert isinstance(wrapped, CompositeBackend)
    assert isinstance(wrapped.default, osb.OpenShellRuntimeSandbox)
    # Offloaded messages must survive openshell_reset_sandbox.
    assert wrapped.routes[osb.settings.SANDBOX_ARTIFACTS_ROOT] is state
    assert wrapped.artifacts_root == osb.settings.SANDBOX_ARTIFACTS_ROOT


def test_wrap_backend_refuses_to_shadow_a_configured_route(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend

    taken = CompositeBackend(
        default=StateBackend(),
        routes={osb.settings.SANDBOX_ARTIFACTS_ROOT: StateBackend()},
    )
    with pytest.raises(ValueError, match="already routed"):
        osb.wrap_backend(taken)


def test_builtin_tools_are_offered_even_unconfigured() -> None:
    names = {t.name for t in osb.get_builtin_tools()}
    assert names == {"openshell_use_runtime", "openshell_reset_sandbox"}


# ── Logging ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "auth failed: client_secret=hunter2shhh",
        'rpc error, authorization: "Bearer eyJhbGciOi.abc.def"',
        "login rejected for password = s3cret",
    ],
)
def test_credentials_are_redacted_from_logged_errors(text: str) -> None:
    redacted = osb._redact(text)
    for secret in ("hunter2shhh", "eyJhbGciOi.abc.def", "s3cret"):
        assert secret not in redacted


# ── SDK contract ──────────────────────────────────────────────────
#
# The fakes above only prove the backend is self-consistent. These check
# the calls it makes actually exist on the real `openshell` SDK, with the
# keywords used here -- the one class of bug a fake can never catch, and
# the only alternative to a live gateway.

openshell = pytest.importorskip("openshell")


def params(func: Any) -> set[str]:
    import inspect

    return set(inspect.signature(func).parameters)


def test_client_construction_matches_the_sdk() -> None:
    assert {
        "endpoint",
        "tls",
        "client_credentials",
        "cluster_name",
    } <= params(openshell.SandboxClient.__init__)
    assert {
        "client_secret",
        "client_id",
        "issuer",
        "audience",
        "scopes",
        "insecure",
    } <= params(openshell.ClientCredentialsAuth.__init__)
    assert {"ca_path", "cert_path", "key_path"} <= params(openshell.TlsConfig)


def test_the_grpc_channel_is_not_secured_by_the_oidc_insecure_flag() -> None:
    """`insecure` is httpx verify for the issuer; `tls` is the channel.

    Conflating them -- the bug this replaced -- leaves the gateway channel
    plaintext while looking configured, so pin the distinction.
    """
    assert "insecure" not in params(openshell.SandboxClient.__init__)
    assert "tls" not in params(openshell.ClientCredentialsAuth.__init__)


def test_lifecycle_calls_match_the_sdk() -> None:
    client = openshell.SandboxClient
    assert {"workspace", "spec", "labels"} <= params(client.create_session)
    assert {"name", "workspace", "timeout_seconds"} <= params(client.wait_ready)
    assert {"name", "workspace", "allow_missing"} <= params(client.delete)
    assert callable(client.close), "the gRPC channel must be closable at shutdown"


def test_a_session_exposes_what_the_backend_reads_from_it() -> None:
    # `_create_sandbox` keeps the session and later reads handle.sandbox.name
    # to delete the pod; `OpenShellSandbox` calls handle.exec.
    ref = type("Ref", (), {"workspace": "default", "id": "sbx-1", "name": "sbx-1"})()
    session = openshell.SandboxSession(client=None, sandbox=ref)  # type: ignore[arg-type]

    assert session.sandbox.name == "sbx-1"
    assert session.id == "sbx-1"
    assert callable(session.exec)
    assert {"name", "id"} <= set(openshell.SandboxRef.__dataclass_fields__)


def test_the_wrapped_sandbox_accepts_a_session() -> None:
    """`OpenShellSandbox` must keep taking the handle `_create_sandbox` builds."""
    from langchain_nvidia_openshell import OpenShellSandbox

    assert {"sandbox", "timeout"} <= params(OpenShellSandbox.__init__)
    assert issubclass(osb.OpenShellRuntimeSandbox, OpenShellSandbox)


def test_tls_config_is_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_ENABLED", True)
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_CA_PATH", "/tmp/ca.pem")
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_CERT_PATH", None)
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_KEY_PATH", None)

    tls = osb._tls_config()
    assert isinstance(tls, openshell.TlsConfig)
    assert str(tls.ca_path) == "/tmp/ca.pem"

    # No paths at all is valid: it means the OS trust store.
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_CA_PATH", None)
    assert osb._tls_config().ca_path is None

    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_ENABLED", False)
    assert osb._tls_config() is None


def test_a_vanished_sandbox_is_recognised_from_the_wrapped_error() -> None:
    """The text `OpenShellSandbox.execute` produces for a dead pod."""
    from langchain_nvidia_openshell.sandbox import _exception_to_response

    class FakeRpcError(Exception):
        def __str__(self) -> str:
            return (
                "<_InactiveRpcError of RPC that terminated with:\n"
                '\tstatus = StatusCode.NOT_FOUND\n\tdetails = "sandbox sbx-1"\n>'
            )

    assert osb._looks_dead(_exception_to_response(FakeRpcError()))
    assert not osb._looks_dead(_exception_to_response(RuntimeError("exit status 1")))


@pytest.fixture()
def gateway_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A realistic non-loopback gateway configuration."""
    for name, value in {
        "OPENSHELL_ENDPOINT": "openshell-gateway.openshell.svc:8443",
        "OPENSHELL_CLIENT_ID": "template-agent",
        "OPENSHELL_CLIENT_SECRET": "s3cret",
        "OPENSHELL_ISSUER": "https://sso.example.com/realms/agents",
        "OPENSHELL_SCOPES": None,
        "OPENSHELL_TLS_ENABLED": True,
        "OPENSHELL_TLS_CA_PATH": None,
        "OPENSHELL_TLS_CERT_PATH": None,
        "OPENSHELL_TLS_KEY_PATH": None,
        "OPENSHELL_OIDC_INSECURE": False,
    }.items():
        monkeypatch.setattr(osb.settings, name, value)
    monkeypatch.setattr(osb, "_client_singleton", None)


def test_the_client_builds_against_a_real_non_loopback_gateway(
    gateway_settings: None,
) -> None:
    """The whole point: a gRPC channel is lazy, so this needs no gateway.

    It does exercise `SandboxClient.__init__`, which rejects OAuth client
    credentials on a plaintext channel -- so before TLS was wired through,
    every real deployment raised here at the first tool call.
    """
    client = osb._client()
    try:
        assert isinstance(client, openshell.SandboxClient)
        assert osb._client() is client, "the client is built once per process"
    finally:
        osb.shutdown_all()


def test_plaintext_to_a_remote_gateway_is_refused(
    gateway_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK will not send a bearer token in the clear, and neither do we."""
    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_ENABLED", False)
    with pytest.raises(openshell.SandboxError, match="require TLS"):
        osb._client()


def test_startup_rejects_plaintext_to_a_remote_gateway(
    gateway_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caught at startup rather than at the first tool call."""
    from deep_agent.src.exceptions import AppException
    from deep_agent.src.settings import validate_config

    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_ENABLED", False)
    with pytest.raises(AppException, match="cleartext"):
        validate_config(osb.settings)


def test_startup_rejects_a_half_configured_identity(
    gateway_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deep_agent.src.exceptions import AppException
    from deep_agent.src.settings import validate_config

    monkeypatch.setattr(osb.settings, "OPENSHELL_CLIENT_SECRET", None)
    with pytest.raises(AppException, match="OPENSHELL_CLIENT_SECRET"):
        validate_config(osb.settings)


def test_startup_rejects_half_of_an_mtls_identity(
    gateway_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deep_agent.src.exceptions import AppException
    from deep_agent.src.settings import validate_config

    monkeypatch.setattr(osb.settings, "OPENSHELL_TLS_CERT_PATH", "/tmp/c.pem")
    with pytest.raises(AppException, match="set together"):
        validate_config(osb.settings)


def test_shutdown_closes_the_grpc_channel(gateway_settings: None) -> None:
    client = osb._client()
    closed: list[bool] = []
    client.close = lambda: closed.append(True)  # type: ignore[method-assign]

    osb.shutdown_all()

    assert closed == [True]
    assert osb._client_singleton is None


@pytest.mark.parametrize(
    "value",
    ["u.alice.t.abc", "...", "a" * 200, "user@example.com", "-leading", "trailing-"],
)
def test_owner_labels_are_valid_kubernetes_label_values(value: str) -> None:
    """The gateway rejects the whole create if a label value is malformed."""
    import re as _re

    label = osb._sanitize(value, limit=63)
    assert len(label) <= 63
    assert _re.fullmatch(r"[a-zA-Z0-9]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?", label), label


def test_the_image_travels_on_the_sandbox_not_a_stored_template() -> None:
    """Inline spec keeps the agent at workspace `user`.

    Creating a stored workload template needs the `admin` role in the
    workspace; creating a sandbox needs only `user`. Regressing to a
    template would re-break a least-privilege deployment.
    """
    spec = osb._runtime_spec("python")
    assert spec.template.image == osb.RUNTIMES["python"]
    assert not hasattr(osb, "_ensure_template")


def test_create_session_is_preferred_over_the_template_variant() -> None:
    assert {"workspace", "spec", "labels"} <= params(
        openshell.SandboxClient.create_session
    )


# ── Trace metadata ────────────────────────────────────────────────
#
# An execute span says a command ran; without these tags it does not say
# which backend, runtime or sandbox served it -- the first thing anyone
# reading a trace needs when more than one sandbox provider is deployed.


class TestTraceMetadata:
    def test_execute_tags_the_span(self, client: FakeClient) -> None:
        import langfuse

        sandbox = osb.get_sandbox()
        with as_caller("alice", "t1"), patch.object(langfuse, "get_client") as gc:
            sandbox.execute("echo hi")

        md = gc.return_value.update_current_span.call_args.kwargs["metadata"]
        assert md["sandbox.provider"] == "openshell"
        assert md["sandbox.runtime"] == "python"
        assert md["sandbox.image"] == osb.RUNTIMES["python"]
        assert md["sandbox.session"] == "u.alice.t.t1"
        assert md["sandbox.id"] == "sbx-1"
        assert md["status"] == "ok"
        assert md["exit_code"] == 0

    def test_a_refused_call_is_tagged_too(self, client: FakeClient) -> None:
        import langfuse

        sandbox = osb.get_sandbox()
        with as_caller(), patch.object(langfuse, "get_client") as gc:
            result = sandbox.execute("echo hi")

        md = gc.return_value.update_current_span.call_args.kwargs["metadata"]
        assert md["sandbox.provider"] == "openshell"
        assert md["status"] == "OpenShellOwnerUnknown"
        assert result.exit_code == 127, "non-zero, or the message reads as output"

    def test_a_replaced_sandbox_is_marked_recreated(self, client: FakeClient) -> None:
        """Otherwise the span credits output to a pod that never ran it."""
        import langfuse

        sandbox = osb.get_sandbox()
        with as_caller("alice", "t1"):
            sandbox.execute("echo one")
            client.created[0].dead = True
            with patch.object(langfuse, "get_client") as gc:
                sandbox.execute("echo two")

        md = gc.return_value.update_current_span.call_args.kwargs["metadata"]
        assert md["recreated"] is True
        assert md["sandbox.id"] == "sbx-2"

    def test_a_tracing_failure_never_breaks_a_run(self, client: FakeClient) -> None:
        import langfuse

        sandbox = osb.get_sandbox()
        with as_caller("alice", "t1"), patch.object(langfuse, "get_client") as gc:
            gc.side_effect = RuntimeError("langfuse is down")
            assert sandbox.execute("echo hi").exit_code == 0
