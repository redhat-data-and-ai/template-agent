"""User isolation tests — prove data boundaries between users.

Uses an in-memory fake DB that simulates PostgreSQL WHERE clause
filtering so repository SQL logic is actually exercised.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.src.feedback.repository import FeedbackRepository
from deep_agent.src.personalization.injector import inject_personalization
from deep_agent.src.personalization.repository import PersonalizationRepository


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tables_flags():
    import deep_agent.src.feedback.repository as f_mod
    import deep_agent.src.personalization.repository as p_mod

    p_mod._TABLES_ENSURED = True
    f_mod._TABLE_ENSURED = True
    yield
    p_mod._TABLES_ENSURED = False
    f_mod._TABLE_ENSURED = False


@pytest.fixture
def repo():
    return PersonalizationRepository("postgresql://test:test@localhost/testdb")


# ── In-Memory Fake DB ────────────────────────────────────────────


class FakeCursor:
    def __init__(self):
        self.rows = []
        self.rowcount = 0

    async def fetchall(self):
        return self.rows


class FakeDB:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {
            "user_memories": [],
            "user_rules": [],
            "message_feedback": [],
        }

    def make_connection(self):
        db = self

        class FakeConn:
            async def execute(self_, query, params=None):
                return await db._execute(query, params)

            async def commit(self_):
                pass

            async def __aenter__(self_):
                return self_

            async def __aexit__(self_, *args):
                pass

        return FakeConn()

    async def _execute(self, query: str, params=None):
        cursor = FakeCursor()
        q = query.strip().lower()

        if q.startswith("insert into user_memories"):
            self.tables["user_memories"].append(
                {
                    "id": params[0],
                    "user_id": params[1],
                    "content": params[2],
                    "score": 1.0,
                    "cluster_id": None,
                    "created_at": params[3]
                    if len(params) > 3
                    else datetime.now(timezone.utc),
                    "updated_at": params[4]
                    if len(params) > 4
                    else datetime.now(timezone.utc),
                }
            )

        elif q.startswith("insert into user_rules"):
            self.tables["user_rules"].append(
                {
                    "id": params[0],
                    "user_id": params[1],
                    "content": params[2],
                    "is_active": params[3],
                    "created_at": params[4],
                    "updated_at": params[5],
                }
            )

        elif q.startswith("insert into message_feedback"):
            existing = [
                r
                for r in self.tables["message_feedback"]
                if r["thread_id"] == params[0]
                and r["message_id"] == params[1]
                and r["user_id"] == params[2]
            ]
            if existing:
                existing[0]["feedback"] = params[3]
                existing[0]["trace_id"] = params[4]
            else:
                self.tables["message_feedback"].append(
                    {
                        "thread_id": params[0],
                        "message_id": params[1],
                        "user_id": params[2],
                        "feedback": params[3],
                        "trace_id": params[4],
                    }
                )

        elif q.startswith("select") and "user_memories" in q:
            rows = self.tables["user_memories"]
            if params and "where user_id" in q:
                rows = [r for r in rows if r["user_id"] == params[0]]
            cursor.rows = rows

        elif q.startswith("select") and "user_rules" in q:
            rows = self.tables["user_rules"]
            if params and "where user_id" in q:
                rows = [r for r in rows if r["user_id"] == params[0]]
                if "is_active = true" in q:
                    rows = [r for r in rows if r["is_active"]]
            cursor.rows = rows

        elif q.startswith("select") and "message_feedback" in q:
            rows = self.tables["message_feedback"]
            if params:
                rows = [
                    r
                    for r in rows
                    if r["thread_id"] == params[0] and r["user_id"] == params[1]
                ]
            cursor.rows = [
                {"message_id": r["message_id"], "feedback": r["feedback"]} for r in rows
            ]

        elif q.startswith("delete") and "user_memories" in q:
            before = len(self.tables["user_memories"])
            if params and len(params) == 2:
                self.tables["user_memories"] = [
                    r
                    for r in self.tables["user_memories"]
                    if not (r["id"] == params[0] and r["user_id"] == params[1])
                ]
            elif params and len(params) == 1:
                self.tables["user_memories"] = [
                    r for r in self.tables["user_memories"] if r["user_id"] != params[0]
                ]
            cursor.rowcount = before - len(self.tables["user_memories"])

        elif q.startswith("delete") and "user_rules" in q:
            before = len(self.tables["user_rules"])
            if params and len(params) == 2:
                self.tables["user_rules"] = [
                    r
                    for r in self.tables["user_rules"]
                    if not (r["id"] == params[0] and r["user_id"] == params[1])
                ]
            elif params and len(params) == 1:
                self.tables["user_rules"] = [
                    r for r in self.tables["user_rules"] if r["user_id"] != params[0]
                ]
            cursor.rowcount = before - len(self.tables["user_rules"])

        elif q.startswith("delete") and "message_feedback" in q:
            before = len(self.tables["message_feedback"])
            self.tables["message_feedback"] = [
                r
                for r in self.tables["message_feedback"]
                if not (
                    r["thread_id"] == params[0]
                    and r["message_id"] == params[1]
                    and r["user_id"] == params[2]
                )
            ]
            cursor.rowcount = before - len(self.tables["message_feedback"])

        return cursor


@pytest.fixture
def fake_db():
    return FakeDB()


@pytest.fixture
def patched_repos(fake_db):
    async def _make_conn(*a, **kw):
        return fake_db.make_connection()

    with (
        patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            side_effect=_make_conn,
        ),
        patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            side_effect=_make_conn,
        ),
        patch(
            "deep_agent.src.cache.personalization_cache.invalidate",
            new_callable=AsyncMock,
        ),
    ):
        p_repo = PersonalizationRepository("postgresql://test")
        f_repo = FeedbackRepository("postgresql://test")
        yield p_repo, f_repo, fake_db


# ── Test 1-2: Memory Isolation ───────────────────────────────────


class TestMemoryIsolation:
    @pytest.mark.asyncio
    async def test_memory_read_isolation(self, patched_repos):
        """User A cannot see User B's memories."""
        p_repo, _, _ = patched_repos
        await p_repo.create_memory("user-a", "Likes Python")
        await p_repo.create_memory("user-b", "Prefers Java")

        a_mems = await p_repo.list_memories("user-a")
        b_mems = await p_repo.list_memories("user-b")

        assert len(a_mems) == 1
        assert a_mems[0].content == "Likes Python"
        assert len(b_mems) == 1
        assert b_mems[0].content == "Prefers Java"

    @pytest.mark.asyncio
    async def test_memory_write_isolation(self, patched_repos):
        """Each user's creates land in their own namespace."""
        p_repo, _, _ = patched_repos
        await p_repo.create_memory("user-a", "Memory A1")
        await p_repo.create_memory("user-a", "Memory A2")
        await p_repo.create_memory("user-b", "Memory B1")

        a_mems = await p_repo.list_memories("user-a")
        b_mems = await p_repo.list_memories("user-b")

        assert len(a_mems) == 2
        assert len(b_mems) == 1
        a_contents = {m.content for m in a_mems}
        assert a_contents == {"Memory A1", "Memory A2"}
        assert b_mems[0].content == "Memory B1"


# ── Test 3-4: Rule Isolation ─────────────────────────────────────


class TestRuleIsolation:
    @pytest.mark.asyncio
    async def test_rule_read_isolation(self, patched_repos):
        """User A cannot see User B's rules."""
        p_repo, _, _ = patched_repos
        await p_repo.upsert_rule("user-a", "Be concise")
        await p_repo.upsert_rule("user-b", "Use emojis")

        a_rules = await p_repo.list_rules("user-a")
        b_rules = await p_repo.list_rules("user-b")

        assert len(a_rules) == 1
        assert a_rules[0].content == "Be concise"
        assert len(b_rules) == 1
        assert b_rules[0].content == "Use emojis"

    @pytest.mark.asyncio
    async def test_rule_write_isolation(self, patched_repos):
        """Each user's rules land in their own namespace."""
        p_repo, _, _ = patched_repos
        await p_repo.upsert_rule("user-a", "Rule A1")
        await p_repo.upsert_rule("user-a", "Rule A2")
        await p_repo.upsert_rule("user-b", "Rule B1")

        a_rules = await p_repo.list_rules("user-a", active_only=False)
        b_rules = await p_repo.list_rules("user-b", active_only=False)

        assert len(a_rules) == 2
        assert len(b_rules) == 1


# ── Test 5: Feedback Isolation ───────────────────────────────────


class TestFeedbackIsolation:
    @pytest.mark.asyncio
    async def test_feedback_per_user_on_same_message(self, patched_repos):
        """Same message shows different feedback per user."""
        _, f_repo, _ = patched_repos
        await f_repo.upsert_feedback("t1", "msg1", "user-a", "up", None)
        await f_repo.upsert_feedback("t1", "msg1", "user-b", "down", None)

        a_fb = await f_repo.list_feedback("t1", "user-a")
        b_fb = await f_repo.list_feedback("t1", "user-b")

        assert len(a_fb) == 1
        assert a_fb[0]["feedback"] == "up"
        assert len(b_fb) == 1
        assert b_fb[0]["feedback"] == "down"


# ── Test 6-7: Cross-User Delete Blocked ──────────────────────────


class TestCrossUserDeleteBlocked:
    @pytest.mark.asyncio
    async def test_cross_user_memory_delete_blocked(self, patched_repos):
        """User B cannot delete User A's memory."""
        p_repo, _, _ = patched_repos
        mem = await p_repo.create_memory("user-a", "Secret memory")

        result = await p_repo.delete_memory("user-b", mem.id)
        assert result is False

        a_mems = await p_repo.list_memories("user-a")
        assert len(a_mems) == 1
        assert a_mems[0].content == "Secret memory"

    @pytest.mark.asyncio
    async def test_cross_user_rule_delete_blocked(self, patched_repos):
        """User B cannot delete User A's rule."""
        p_repo, _, _ = patched_repos
        rule = await p_repo.upsert_rule("user-a", "Private rule")

        result = await p_repo.delete_rule("user-b", rule.id)
        assert result is False

        a_rules = await p_repo.list_rules("user-a")
        assert len(a_rules) == 1


# ── Test 8: Hard Delete Verification ─────────────────────────────


class TestHardDelete:
    @pytest.mark.asyncio
    async def test_hard_delete_removes_data(self, patched_repos):
        """Deleted memory is fully gone from DB, not soft-deleted."""
        p_repo, _, fake_db = patched_repos
        mem = await p_repo.create_memory("user-a", "To be deleted")
        assert len(await p_repo.list_memories("user-a")) == 1

        result = await p_repo.delete_memory("user-a", mem.id)
        assert result is True

        remaining = await p_repo.list_memories("user-a")
        assert len(remaining) == 0
        assert len(fake_db.tables["user_memories"]) == 0


# ── Test 9: Concurrent Session Safety ────────────────────────────


class TestConcurrentSessions:
    @pytest.mark.asyncio
    async def test_concurrent_writes_no_cross_contamination(self, patched_repos):
        """Parallel writes for different users don't interfere."""
        p_repo, _, _ = patched_repos

        async def write_for_user(user_id: str, count: int):
            for i in range(count):
                await p_repo.create_memory(user_id, f"{user_id}-mem-{i}")

        await asyncio.gather(
            write_for_user("user-a", 5),
            write_for_user("user-b", 5),
        )

        a_mems = await p_repo.list_memories("user-a")
        b_mems = await p_repo.list_memories("user-b")

        assert len(a_mems) == 5
        assert len(b_mems) == 5
        assert all("user-a" in m.content for m in a_mems)
        assert all("user-b" in m.content for m in b_mems)


# ── Test 10: Personalization Injection Scoping ───────────────────


class TestPersonalizationInjectionScoping:
    def test_injection_contains_only_requesting_user_data(self):
        """System prompt contains only the requesting user's data."""
        prompt_a = inject_personalization(
            "Base prompt",
            ["Likes Python", "Uses Linux"],
            ["Be concise"],
        )
        prompt_b = inject_personalization(
            "Base prompt",
            ["Prefers Java"],
            ["Use emojis"],
        )

        assert "Likes Python" in prompt_a
        assert "Uses Linux" in prompt_a
        assert "Be concise" in prompt_a
        assert "Prefers Java" not in prompt_a
        assert "Use emojis" not in prompt_a

        assert "Prefers Java" in prompt_b
        assert "Use emojis" in prompt_b
        assert "Likes Python" not in prompt_b
        assert "Be concise" not in prompt_b


# ── Test 11: Cache Invalidation on Delete ────────────────────────


# ── Test 12: Decay Scoring Per-User ──────────────────────────────


class TestDecayScoping:
    @pytest.mark.asyncio
    async def test_decay_user_memories_scopes_by_user_id(self):
        """decay_user_memories must SELECT with WHERE user_id = %s."""
        from deep_agent.src.memory.scoring import decay_user_memories

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "psycopg.AsyncConnection.connect",
            new_callable=AsyncMock,
            return_value=mock_conn,
        ):
            await decay_user_memories("postgresql://test", "user-a")

            sql_call = mock_conn.execute.call_args_list[0]
            query = sql_call[0][0]
            params = sql_call[0][1]
            assert "WHERE user_id" in query
            assert params == ("user-a",)


# ── Test 14-15: Bulk Delete Isolation ────────────────────────────


class TestBulkDeleteIsolation:
    @pytest.mark.asyncio
    async def test_bulk_delete_memories_only_affects_own_user(self, patched_repos):
        """delete_all_memories removes only the caller's memories."""
        p_repo, _, fake_db = patched_repos
        await p_repo.create_memory("user-a", "A memory 1")
        await p_repo.create_memory("user-a", "A memory 2")
        await p_repo.create_memory("user-b", "B memory 1")

        count = await p_repo.delete_all_memories("user-a")
        assert count == 2

        assert len(await p_repo.list_memories("user-a")) == 0
        b_mems = await p_repo.list_memories("user-b")
        assert len(b_mems) == 1
        assert b_mems[0].content == "B memory 1"

    @pytest.mark.asyncio
    async def test_bulk_delete_rules_only_affects_own_user(self, patched_repos):
        """delete_all_rules removes only the caller's rules."""
        p_repo, _, _ = patched_repos
        await p_repo.upsert_rule("user-a", "A rule 1")
        await p_repo.upsert_rule("user-a", "A rule 2")
        await p_repo.upsert_rule("user-b", "B rule 1")

        count = await p_repo.delete_all_rules("user-a")
        assert count == 2

        assert len(await p_repo.list_rules("user-a", active_only=False)) == 0
        b_rules = await p_repo.list_rules("user-b", active_only=False)
        assert len(b_rules) == 1
        assert b_rules[0].content == "B rule 1"


# ── Test 16: Top Memories Isolation ──────────────────────────────


class TestTopMemoriesIsolation:
    @pytest.mark.asyncio
    async def test_top_memories_scoped_by_user(self, patched_repos):
        """list_top_memories returns only the requesting user's data."""
        p_repo, _, _ = patched_repos
        await p_repo.create_memory("user-a", "A top mem")
        await p_repo.create_memory("user-b", "B top mem 1")
        await p_repo.create_memory("user-b", "B top mem 2")

        a_top = await p_repo.list_top_memories("user-a", limit=10)
        b_top = await p_repo.list_top_memories("user-b", limit=10)

        assert len(a_top) == 1
        assert a_top[0].content == "A top mem"
        assert len(b_top) == 2
        assert {m.content for m in b_top} == {"B top mem 1", "B top mem 2"}


# ── Test 17: Hard Delete for Rules ───────────────────────────────


class TestHardDeleteRules:
    @pytest.mark.asyncio
    async def test_hard_delete_rule_removes_from_db(self, patched_repos):
        """Deleted rule is fully gone, not soft-deleted."""
        p_repo, _, fake_db = patched_repos
        rule = await p_repo.upsert_rule("user-a", "Ephemeral rule")
        assert len(fake_db.tables["user_rules"]) == 1

        result = await p_repo.delete_rule("user-a", rule.id)
        assert result is True
        assert len(await p_repo.list_rules("user-a", active_only=False)) == 0
        assert len(fake_db.tables["user_rules"]) == 0


# ── Test 18: Cross-User Feedback Delete Blocked ──────────────────


class TestCrossUserFeedbackDelete:
    @pytest.mark.asyncio
    async def test_cross_user_feedback_delete_blocked(self, patched_repos):
        """User B cannot delete User A's feedback."""
        _, f_repo, _ = patched_repos
        await f_repo.upsert_feedback("t1", "msg1", "user-a", "up", None)

        result = await f_repo.delete_feedback("t1", "msg1", "user-b")
        assert result is False

        a_fb = await f_repo.list_feedback("t1", "user-a")
        assert len(a_fb) == 1
        assert a_fb[0]["feedback"] == "up"


# ── Test 19: Delete Rule Invalidates Cache ───────────────────────


# ── Test 20: Concurrent Rule Writes ──────────────────────────────


class TestConcurrentRuleSessions:
    @pytest.mark.asyncio
    async def test_concurrent_rule_writes_isolated(self, patched_repos):
        """Parallel rule writes for different users stay isolated."""
        p_repo, _, _ = patched_repos

        async def write_rules(user_id: str, count: int):
            for i in range(count):
                await p_repo.upsert_rule(user_id, f"{user_id}-rule-{i}")

        await asyncio.gather(
            write_rules("user-a", 4),
            write_rules("user-b", 4),
        )

        a_rules = await p_repo.list_rules("user-a", active_only=False)
        b_rules = await p_repo.list_rules("user-b", active_only=False)

        assert len(a_rules) == 4
        assert len(b_rules) == 4
        assert all("user-a" in r.content for r in a_rules)
        assert all("user-b" in r.content for r in b_rules)


# ── Test 21: Cache Key Namespace Isolation ───────────────────────


class TestCacheKeyNamespace:
    def test_cache_keys_are_user_scoped(self):
        """Personalization cache keys include user_id, preventing cross-user hits."""
        from deep_agent.src.cache.personalization_cache import _cache_key

        key_a = _cache_key("user-a")
        key_b = _cache_key("user-b")

        assert key_a != key_b
        assert "user-a" in key_a
        assert "user-b" in key_b
        assert "user-b" not in key_a
        assert "user-a" not in key_b


# ── Test 22: Three-User Isolation ────────────────────────────────


class TestThreeUserIsolation:
    @pytest.mark.asyncio
    async def test_three_users_fully_isolated(self, patched_repos):
        """Data for users A, B, C is completely separated."""
        p_repo, f_repo, _ = patched_repos

        for user in ["alice", "bob", "carol"]:
            await p_repo.create_memory(user, f"{user} memory")
            await p_repo.upsert_rule(user, f"{user} rule")
            await f_repo.upsert_feedback(
                "t1", "msg1", user, "up" if user != "carol" else "down", None
            )

        for user in ["alice", "bob", "carol"]:
            mems = await p_repo.list_memories(user)
            rules = await p_repo.list_rules(user)
            fb = await f_repo.list_feedback("t1", user)

            assert len(mems) == 1
            assert mems[0].content == f"{user} memory"
            assert len(rules) == 1
            assert rules[0].content == f"{user} rule"
            assert len(fb) == 1

        carol_fb = await f_repo.list_feedback("t1", "carol")
        assert carol_fb[0]["feedback"] == "down"
        alice_fb = await f_repo.list_feedback("t1", "alice")
        assert alice_fb[0]["feedback"] == "up"


# ── Test 23-26: Aegra Thread Ownership (SQL Verification) ────────


class TestAegraThreadOwnership:
    """Verify Aegra thread routes enforce user_id scoping in SQL.

    These tests read the Aegra server source to confirm every thread
    operation includes WHERE user_id = user.identity.
    """

    def _read_threads_source(self) -> str:
        try:
            import aegra_api.api.threads as _threads_mod
        except ImportError:
            pytest.skip("aegra_api not installed")
        from pathlib import Path

        threads_path = Path(_threads_mod.__file__)
        return threads_path.read_text()

    def test_thread_create_stamps_user_identity(self):
        """Thread creation sets user_id=user.identity on the ORM object."""
        src = self._read_threads_source()
        assert "user_id=user.identity" in src
        assert 'metadata["owner"] = user.identity' in src

    def test_thread_get_scoped_by_user(self):
        """GET /threads/{id} queries with AND user_id == user.identity."""
        src = self._read_threads_source()
        assert "ThreadORM.user_id == user.identity" in src

    def test_thread_list_scoped_by_user(self):
        """GET /threads queries with WHERE user_id == user.identity."""
        src = self._read_threads_source()
        lines = [
            l for l in src.split("\n") if "ThreadORM.user_id == user.identity" in l
        ]
        assert len(lines) >= 3, f"Expected 3+ user_id filters, found {len(lines)}"

    def test_thread_delete_scoped_by_user(self):
        """DELETE /threads/{id} queries with AND user_id == user.identity."""
        src = self._read_threads_source()
        delete_section = src[src.index("async def delete_thread") :]
        assert "ThreadORM.user_id == user.identity" in delete_section

    def _read_runs_source(self) -> str:
        from pathlib import Path

        runs_path = Path(
            "/Users/nsaharan/Desktop/template-agent/.venv/lib/python3.12/"
            "site-packages/aegra_api/api/runs.py"
        )
        if not runs_path.exists():
            pytest.skip("aegra_api not installed")
        return runs_path.read_text()

    def test_runs_scoped_by_user(self):
        """Run operations verify user_id matches user.identity."""
        src = self._read_runs_source()
        assert "user.identity" in src
        user_checks = [l for l in src.split("\n") if "user.identity" in l]
        assert len(user_checks) >= 5, (
            f"Expected 5+ user.identity checks in runs, found {len(user_checks)}"
        )
