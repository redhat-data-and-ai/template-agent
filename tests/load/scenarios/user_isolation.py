"""User isolation load test scenario.

Each IsolatedUser authenticates with a unique self-signed JWT and
verifies that it can ONLY see its own data across memories, rules,
threads, and feedback. Any cross-user data leak fires an
IsolationViolationError visible in Locust's failure table.
"""

import logging
import random
import threading
import time
import uuid

from locust import HttpUser, between, events, task

from tests.load.conftest import (
    FEEDBACK_ENDPOINT,
    FEEDBACK_GET_ENDPOINT,
    MEMORIES_ENDPOINT,
    RULES_ENDPOINT,
    RUNS_STREAM_ENDPOINT,
    THREADS_ENDPOINT,
    THREADS_SEARCH_ENDPOINT,
    StreamMetrics,
    build_run_payload,
    extract_content_from_event,
    get_all_prompts,
    parse_sse_stream,
)
from tests.load.jwt_provider import create_user_token

logger = logging.getLogger(__name__)

_user_counter = 0
_counter_lock = threading.Lock()

canary_registry: dict[str, dict] = {}
_registry_lock = threading.Lock()


class IsolationViolationError(Exception):
    """Raised when a user sees another user's data."""


def _next_user_id() -> str:
    global _user_counter
    with _counter_lock:
        _user_counter += 1
        n = _user_counter
    return f"locust-{n}-{uuid.uuid4().hex[:8]}"


def _marker(user_id: str) -> str:
    return f"[locust:{user_id}]"


class IsolatedUser(HttpUser):
    """Locust user that verifies data isolation under concurrent load.

    On start, generates a unique JWT and creates canary resources.
    Each task creates, reads, and deletes data while asserting that
    only its own data is ever visible.
    """

    weight = 1
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self._user_id = _next_user_id()
        self._token = create_user_token(self._user_id)
        self._auth_headers = {"Authorization": f"Bearer {self._token}"}
        self._marker = _marker(self._user_id)
        self._prompts = get_all_prompts()
        self._thread_ids: list[str] = []
        self._canary_memory_id: str | None = None
        self._canary_rule_id: str | None = None

        logger.info("IsolatedUser started: %s", self._user_id)
        self._create_canaries()

    def on_stop(self) -> None:
        self._cleanup()

    def _req(self, method: str, path: str, **kwargs) -> object:  # noqa: ANN003
        """Make an authenticated HTTP request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_headers)
        return getattr(self.client, method)(path, headers=headers, **kwargs)

    # ── Canary setup / teardown ──────────────────────────────────

    def _create_canaries(self) -> None:
        with self._req(
            "post",
            MEMORIES_ENDPOINT,
            json={"content": f"{self._marker} canary-memory"},
            name="POST /memories [canary]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                self._canary_memory_id = resp.json().get("id")
                if self._canary_memory_id is None:
                    resp.failure("201 response for canary memory omits 'id'")
                else:
                    resp.success()
            else:
                resp.failure(f"Canary memory creation failed: {resp.status_code}")

        with self._req(
            "post",
            RULES_ENDPOINT,
            json={"content": f"{self._marker} canary-rule", "is_active": True},
            name="POST /rules [canary]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                self._canary_rule_id = resp.json().get("id")
                if self._canary_rule_id is None:
                    resp.failure("201 response for canary rule omits 'id'")
                else:
                    resp.success()
            else:
                resp.failure(f"Canary rule creation failed: {resp.status_code}")

        with _registry_lock:
            canary_registry[self._user_id] = {
                "memory_id": self._canary_memory_id,
                "rule_id": self._canary_rule_id,
            }

    def _cleanup(self) -> None:
        with _registry_lock:
            canary_registry.pop(self._user_id, None)

        self._req(
            "delete",
            MEMORIES_ENDPOINT,
            name="DELETE /memories [bulk cleanup]",
        )
        self._req(
            "delete",
            RULES_ENDPOINT,
            name="DELETE /rules [bulk cleanup]",
        )

        for tid in self._thread_ids:
            self._req(
                "delete",
                f"/threads/{tid}",
                name="DELETE /threads/{id} [cleanup]",
            )

    # ── Isolation assertion helpers ──────────────────────────────

    def _assert_all_mine(
        self, items: list[dict], field: str, resource_name: str
    ) -> None:
        for item in items:
            value = item.get(field, "")
            if self._marker not in value:
                error = IsolationViolationError(
                    f"User {self._user_id} saw foreign {resource_name}: "
                    f"{field}='{value[:100]}'"
                )
                events.request.fire(
                    request_type="ISOLATION",
                    name=f"violation:{resource_name}",
                    response_time=0,
                    response_length=0,
                    context={"user_id": self._user_id, "foreign_data": value[:100]},
                    exception=error,
                )
                raise error

    def _fire_isolation_event(
        self, name: str, detail: str, is_failure: bool = True
    ) -> None:
        exc = IsolationViolationError(detail) if is_failure else None
        events.request.fire(
            request_type="ISOLATION",
            name=name,
            response_time=0,
            response_length=0,
            context={"user_id": self._user_id},
            exception=exc,
        )

    # ── Tasks ────────────────────────────────────────────────────

    @task(3)
    def memory_lifecycle(self) -> None:
        """Create → list (verify isolation) → delete → verify gone."""
        mem_tag = uuid.uuid4().hex[:8]
        content = f"{self._marker} memory-{mem_tag}"

        # Create
        with self._req(
            "post",
            MEMORIES_ENDPOINT,
            json={"content": content},
            name="POST /memories",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"Memory create failed: {resp.status_code}")
                return
            memory_id = resp.json().get("id")
            resp.success()

        # List and verify isolation
        with self._req(
            "get",
            MEMORIES_ENDPOINT,
            name="GET /memories",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory list failed: {resp.status_code}")
                return
            memories = resp.json().get("memories", [])
            try:
                self._assert_all_mine(memories, "content", "memory")
                resp.success()
            except IsolationViolationError:
                resp.failure("ISOLATION VIOLATION: saw foreign memory")
                return

        # Delete
        with self._req(
            "delete",
            f"{MEMORIES_ENDPOINT}/{memory_id}",
            name="DELETE /memories/{id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory delete failed: {resp.status_code}")
                return
            resp.success()

        # Verify gone
        with self._req(
            "get",
            MEMORIES_ENDPOINT,
            name="GET /memories [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory list failed: {resp.status_code}")
                return
            remaining_ids = [m["id"] for m in resp.json().get("memories", [])]
            if memory_id in remaining_ids:
                resp.failure(f"Deleted memory {memory_id} still visible")
            else:
                resp.success()

    @task(3)
    def rule_lifecycle(self) -> None:
        """Create → list (verify isolation) → delete → verify gone."""
        rule_tag = uuid.uuid4().hex[:8]
        content = f"{self._marker} rule-{rule_tag}"

        # Create
        with self._req(
            "post",
            RULES_ENDPOINT,
            json={"content": content, "is_active": True},
            name="POST /rules",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"Rule create failed: {resp.status_code}")
                return
            rule_id = resp.json().get("id")
            resp.success()

        # List and verify isolation
        with self._req(
            "get",
            RULES_ENDPOINT,
            name="GET /rules",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule list failed: {resp.status_code}")
                return
            rules = resp.json().get("rules", [])
            try:
                self._assert_all_mine(rules, "content", "rule")
                resp.success()
            except IsolationViolationError:
                resp.failure("ISOLATION VIOLATION: saw foreign rule")
                return

        # Delete
        with self._req(
            "delete",
            f"{RULES_ENDPOINT}/{rule_id}",
            name="DELETE /rules/{id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule delete failed: {resp.status_code}")
                return
            resp.success()

        # Verify gone
        with self._req(
            "get",
            RULES_ENDPOINT,
            name="GET /rules [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule list failed: {resp.status_code}")
                return
            remaining_ids = [r["id"] for r in resp.json().get("rules", [])]
            if rule_id in remaining_ids:
                resp.failure(f"Deleted rule {rule_id} still visible")
            else:
                resp.success()

    @task(2)
    def chat_and_thread(self) -> None:
        """Create thread → send message → stream → list threads → verify isolation."""
        # Create thread
        with self._req(
            "post",
            THREADS_ENDPOINT,
            json={},
            name="POST /threads",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 201):
                resp.failure(f"Thread create failed: {resp.status_code}")
                return
            body = resp.json()
            thread_id = body.get("thread_id") or body.get("id")
            if not thread_id:
                resp.failure(f"No thread_id in response: {body}")
                return
            self._thread_ids.append(thread_id)
            resp.success()

        # Send message and stream response
        prompt = random.choice(self._prompts)  # noqa: S311
        endpoint = RUNS_STREAM_ENDPOINT.format(thread_id=thread_id)
        payload = build_run_payload(prompt)
        start_time = time.time()

        with self._req(
            "post",
            endpoint,
            json=payload,
            name="POST /threads/{id}/runs/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Stream failed: {resp.status_code} {resp.text[:200]}")
                return

            metrics = StreamMetrics()
            try:
                for event in parse_sse_stream(resp.iter_lines()):
                    content = extract_content_from_event(event)
                    if content is not None:
                        metrics.token_count += 1
                        if not metrics.first_token_received:
                            metrics.ttft_ms = (time.time() - start_time) * 1000
                            metrics.first_token_received = True
            except Exception as exc:
                metrics.error = exc

            metrics.total_time_ms = (time.time() - start_time) * 1000

            if metrics.error:
                resp.failure(f"Stream error: {metrics.error}")
            elif not metrics.first_token_received:
                resp.failure("No content tokens in stream")
            else:
                resp.success()

        if metrics.first_token_received:
            events.request.fire(
                request_type="SSE",
                name="TTFT (isolation)",
                response_time=metrics.ttft_ms,
                response_length=0,
                context={"user_id": self._user_id},
                exception=None,
            )

        # List threads and verify isolation
        with self._req(
            "post",
            THREADS_SEARCH_ENDPOINT,
            json={},
            name="POST /threads/search",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread search failed: {resp.status_code}")
                return

            threads = resp.json()
            if not isinstance(threads, list):
                resp.failure(
                    f"Thread search returned non-list: {type(threads).__name__}"
                )
                return

            returned_ids = set()
            for t in threads:
                tid = t.get("thread_id") or t.get("id", "")
                returned_ids.add(tid)

            foreign_ids = returned_ids - set(self._thread_ids)
            if foreign_ids:
                detail = (
                    f"User {self._user_id} saw foreign thread IDs: "
                    f"{list(foreign_ids)[:5]}"
                )
                self._fire_isolation_event("violation:threads", detail)
                resp.failure(f"ISOLATION VIOLATION: {detail}")
            else:
                resp.success()

    @task(2)
    def feedback_lifecycle(self) -> None:
        """Submit feedback on own thread → get → verify only own."""
        if not self._thread_ids:
            return

        thread_id = self._thread_ids[-1]
        message_id = uuid.uuid4().hex

        # Submit feedback
        feedback_payload = {
            "trace_id": uuid.uuid4().hex,
            "name": "thumbs-up",
            "value": 1.0,
            "thread_id": thread_id,
            "message_id": message_id,
            "kwargs": {},
        }

        with self._req(
            "post",
            FEEDBACK_ENDPOINT,
            json=feedback_payload,
            name="POST /feedback",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Feedback submit failed: {resp.status_code}")
                return
            resp.success()

        # Get feedback and verify isolation
        get_url = FEEDBACK_GET_ENDPOINT.format(thread_id=thread_id)
        with self._req(
            "get",
            get_url,
            name="GET /feedback/{thread_id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Feedback get failed: {resp.status_code}")
                return
            feedback_list = resp.json().get("feedback", [])
            if not isinstance(feedback_list, list) or len(feedback_list) == 0:
                resp.failure("Created feedback not returned in GET response")
                return
            matching = [f for f in feedback_list if f.get("message_id") == message_id]
            if not matching:
                resp.failure(
                    f"Submitted feedback for message {message_id} not found in response"
                )
            else:
                resp.success()

    @task(1)
    def cross_user_delete(self) -> None:
        """Attempt to delete another user's canary resource — must fail."""
        with _registry_lock:
            other_users = {
                uid: data
                for uid, data in canary_registry.items()
                if uid != self._user_id
            }

        if not other_users:
            return

        other_uid = random.choice(list(other_users.keys()))  # noqa: S311
        other_data = other_users[other_uid]

        # Try to delete other user's canary memory
        other_memory_id = other_data.get("memory_id")
        if other_memory_id:
            with self._req(
                "delete",
                f"{MEMORIES_ENDPOINT}/{other_memory_id}",
                name="DELETE /memories/{id} [cross-user]",
                catch_response=True,
            ) as resp:
                if resp.status_code == 404:
                    resp.success()
                elif 200 <= resp.status_code < 300:
                    detail = (
                        f"User {self._user_id} DELETED {other_uid}'s "
                        f"memory {other_memory_id} (HTTP {resp.status_code})"
                    )
                    self._fire_isolation_event("cross_delete_succeeded", detail)
                    resp.failure(f"ISOLATION VIOLATION: {detail}")
                else:
                    resp.success()

        # Try to delete other user's canary rule
        other_rule_id = other_data.get("rule_id")
        if other_rule_id:
            with self._req(
                "delete",
                f"{RULES_ENDPOINT}/{other_rule_id}",
                name="DELETE /rules/{id} [cross-user]",
                catch_response=True,
            ) as resp:
                if resp.status_code == 404:
                    resp.success()
                elif 200 <= resp.status_code < 300:
                    detail = (
                        f"User {self._user_id} DELETED {other_uid}'s "
                        f"rule {other_rule_id} (HTTP {resp.status_code})"
                    )
                    self._fire_isolation_event("cross_delete_succeeded", detail)
                    resp.failure(f"ISOLATION VIOLATION: {detail}")
                else:
                    resp.success()

    @task(1)
    def thread_cleanup(self) -> None:
        """Delete own thread via cascading cleanup, verify gone."""
        if len(self._thread_ids) < 2:
            return

        thread_id = self._thread_ids.pop(0)

        with self._req(
            "delete",
            f"/threads/{thread_id}",
            name="DELETE /threads/{id} [cleanup test]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread delete failed: {resp.status_code}")
                self._thread_ids.insert(0, thread_id)
                return
            resp.success()

        # Verify gone from thread list
        with self._req(
            "post",
            THREADS_SEARCH_ENDPOINT,
            json={},
            name="POST /threads/search [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread search failed: {resp.status_code}")
                return
            threads = resp.json()
            if isinstance(threads, list):
                returned_ids = {t.get("thread_id") or t.get("id", "") for t in threads}
                if thread_id in returned_ids:
                    resp.failure(f"Deleted thread {thread_id} still visible")
                else:
                    resp.success()
            else:
                resp.failure(
                    f"Thread search returned non-list: {type(threads).__name__}"
                )
