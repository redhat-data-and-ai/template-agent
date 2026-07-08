"""Live integration tests for user isolation against real Postgres.

Requires a running Postgres with the template_agent database.
Run: .venv/bin/python -m pytest tests/integration/test_user_isolation_live.py -v

These tests use two real user_ids (alice, bob) against the actual
PersonalizationRepository and FeedbackRepository to prove isolation
at the database level — no mocks.
"""

import asyncio
import uuid

import pytest

USER_A = "test-alice-" + uuid.uuid4().hex[:8]
USER_B = "test-bob-" + uuid.uuid4().hex[:8]

DB_URI = "postgresql://postgres:postgres@localhost:5432/template_agent"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(scope="module", autouse=True)
def ensure_tables():
    from deep_agent.src.personalization.repository import PersonalizationRepository
    from deep_agent.src.feedback.repository import FeedbackRepository

    async def setup():
        p = PersonalizationRepository(DB_URI)
        f = FeedbackRepository(DB_URI)
        import deep_agent.src.personalization.repository as p_mod
        import deep_agent.src.feedback.repository as f_mod

        p_mod._TABLES_ENSURED = False
        f_mod._TABLE_ENSURED = False
        await p.ensure_tables()
        await f.ensure_table()

    _run(setup())
    yield

    async def cleanup():
        import psycopg

        async with await psycopg.AsyncConnection.connect(DB_URI) as conn:
            await conn.execute(
                "DELETE FROM user_memories WHERE user_id IN (%s, %s)", (USER_A, USER_B)
            )
            await conn.execute(
                "DELETE FROM user_rules WHERE user_id IN (%s, %s)", (USER_A, USER_B)
            )
            await conn.execute(
                "DELETE FROM message_feedback WHERE user_id IN (%s, %s)",
                (USER_A, USER_B),
            )
            await conn.commit()

    _run(cleanup())


@pytest.fixture
def p_repo():
    from deep_agent.src.personalization.repository import PersonalizationRepository

    return PersonalizationRepository(DB_URI)


@pytest.fixture
def f_repo():
    from deep_agent.src.feedback.repository import FeedbackRepository

    return FeedbackRepository(DB_URI)


class TestMemoryIsolationLive:
    @pytest.mark.asyncio
    async def test_alice_cannot_see_bob_memories(self, p_repo):
        await p_repo.create_memory(USER_A, "Alice likes tea")
        await p_repo.create_memory(USER_B, "Bob likes coffee")

        alice_mems = await p_repo.list_memories(USER_A)
        bob_mems = await p_repo.list_memories(USER_B)

        alice_contents = [m.content for m in alice_mems]
        bob_contents = [m.content for m in bob_mems]

        assert "Alice likes tea" in alice_contents
        assert "Bob likes coffee" not in alice_contents
        assert "Bob likes coffee" in bob_contents
        assert "Alice likes tea" not in bob_contents

    @pytest.mark.asyncio
    async def test_top_memories_scoped(self, p_repo):
        alice_top = await p_repo.list_top_memories(USER_A, limit=10)
        bob_top = await p_repo.list_top_memories(USER_B, limit=10)

        for m in alice_top:
            assert m.user_id == USER_A
        for m in bob_top:
            assert m.user_id == USER_B

    @pytest.mark.asyncio
    async def test_bob_cannot_delete_alice_memory(self, p_repo):
        alice_mems = await p_repo.list_memories(USER_A)
        assert len(alice_mems) > 0
        alice_mem_id = alice_mems[0].id

        result = await p_repo.delete_memory(USER_B, alice_mem_id)
        assert result is False

        still_there = await p_repo.list_memories(USER_A)
        assert any(m.id == alice_mem_id for m in still_there)

    @pytest.mark.asyncio
    async def test_bulk_delete_only_affects_own_user(self, p_repo):
        bob_before = await p_repo.list_memories(USER_B)
        assert len(bob_before) > 0

        count = await p_repo.delete_all_memories(USER_A)
        assert count >= 1

        alice_after = await p_repo.list_memories(USER_A)
        assert len(alice_after) == 0

        bob_after = await p_repo.list_memories(USER_B)
        assert len(bob_after) == len(bob_before)

    @pytest.mark.asyncio
    async def test_hard_delete_verified(self, p_repo):
        mem = await p_repo.create_memory(USER_A, "Ephemeral memory")
        result = await p_repo.delete_memory(USER_A, mem.id)
        assert result is True

        remaining = await p_repo.list_memories(USER_A)
        assert not any(m.id == mem.id for m in remaining)


class TestRuleIsolationLive:
    @pytest.mark.asyncio
    async def test_alice_cannot_see_bob_rules(self, p_repo):
        await p_repo.upsert_rule(USER_A, "Alice rule: be formal")
        await p_repo.upsert_rule(USER_B, "Bob rule: use emojis")

        alice_rules = await p_repo.list_rules(USER_A)
        bob_rules = await p_repo.list_rules(USER_B)

        assert any(r.content == "Alice rule: be formal" for r in alice_rules)
        assert not any(r.content == "Bob rule: use emojis" for r in alice_rules)
        assert any(r.content == "Bob rule: use emojis" for r in bob_rules)

    @pytest.mark.asyncio
    async def test_bob_cannot_delete_alice_rule(self, p_repo):
        alice_rules = await p_repo.list_rules(USER_A)
        assert len(alice_rules) > 0

        result = await p_repo.delete_rule(USER_B, alice_rules[0].id)
        assert result is False

    @pytest.mark.asyncio
    async def test_bulk_delete_rules_scoped(self, p_repo):
        count = await p_repo.delete_all_rules(USER_A)
        assert count >= 1

        bob_rules = await p_repo.list_rules(USER_B)
        assert len(bob_rules) > 0


class TestFeedbackIsolationLive:
    @pytest.mark.asyncio
    async def test_same_message_different_feedback_per_user(self, f_repo):
        thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"
        await f_repo.upsert_feedback(thread_id, "msg1", USER_A, "up", None)
        await f_repo.upsert_feedback(thread_id, "msg1", USER_B, "down", None)

        alice_fb = await f_repo.list_feedback(thread_id, USER_A)
        bob_fb = await f_repo.list_feedback(thread_id, USER_B)

        assert len(alice_fb) == 1
        assert alice_fb[0]["feedback"] == "up"
        assert len(bob_fb) == 1
        assert bob_fb[0]["feedback"] == "down"

    @pytest.mark.asyncio
    async def test_bob_cannot_delete_alice_feedback(self, f_repo):
        thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"
        await f_repo.upsert_feedback(thread_id, "msg1", USER_A, "up", None)

        result = await f_repo.delete_feedback(thread_id, "msg1", USER_B)
        assert result is False

        alice_fb = await f_repo.list_feedback(thread_id, USER_A)
        assert len(alice_fb) == 1


class TestConcurrentSessionsLive:
    @pytest.mark.asyncio
    async def test_concurrent_writes_isolated(self, p_repo):
        async def write_memories(user_id, count):
            for i in range(count):
                await p_repo.create_memory(user_id, f"{user_id}-concurrent-{i}")

        await asyncio.gather(
            write_memories(USER_A, 5),
            write_memories(USER_B, 5),
        )

        alice_mems = await p_repo.list_memories(USER_A)
        bob_mems = await p_repo.list_memories(USER_B)

        alice_concurrent = [m for m in alice_mems if "concurrent" in m.content]
        bob_concurrent = [m for m in bob_mems if "concurrent" in m.content]

        assert len(alice_concurrent) == 5
        assert len(bob_concurrent) == 5
        assert all(USER_A in m.content for m in alice_concurrent)
        assert all(USER_B in m.content for m in bob_concurrent)


class TestPersonalizationInjectionLive:
    @pytest.mark.asyncio
    async def test_injection_scoped_per_user(self, p_repo):
        from deep_agent.src.personalization.injector import inject_personalization

        alice_mems = await p_repo.list_memories(USER_A)
        bob_mems = await p_repo.list_memories(USER_B)
        alice_rules = await p_repo.list_rules(USER_A)
        bob_rules = await p_repo.list_rules(USER_B)

        prompt_a = inject_personalization(
            "Base prompt",
            [m.content for m in alice_mems],
            [r.content for r in alice_rules],
        )
        prompt_b = inject_personalization(
            "Base prompt",
            [m.content for m in bob_mems],
            [r.content for r in bob_rules],
        )

        for m in bob_mems:
            assert m.content not in prompt_a
        for m in alice_mems:
            assert m.content not in prompt_b
