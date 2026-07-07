# User Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 verified user isolation gaps in template-agent (+ template-ui) so that user A's data is invisible to user B, deletions are real, and background jobs respect user boundaries.

**Architecture:** Each fix targets a specific isolation gap found during investigation. Fixes 1-4 are backend-only (template-agent). Fix 5-6 are frontend (template-ui). All fixes are covered by 12 new isolation tests using an in-memory fake DB that simulates PostgreSQL WHERE clause filtering.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, psycopg (async), FastAPI, Redis, React/TypeScript (template-ui)

## Global Constraints

- All repository methods must include `user_id` in SQL WHERE clauses — no global queries on user data tables
- `user_id` for API endpoints must come from authenticated JWT (`sub` claim), never from request body or query params
- Cache must be invalidated after any data mutation (create, update, delete)
- Follow existing test patterns in `tests/unit/test_repository.py` and `tests/unit/feedback/test_repository.py`
- Mark all new tests with `@pytest.mark.asyncio` and `@pytest.mark.unit`

---

### Task 1: Cache Invalidation on Memory/Rule Deletion

**Files:**
- Modify: `deep_agent/src/personalization/repository.py:116-125` (delete_memory)
- Modify: `deep_agent/src/personalization/repository.py:183-192` (delete_rule)
- Modify: `deep_agent/src/memory/consolidation.py:134-153` (consolidate_user_memories)
- Test: `tests/unit/test_user_isolation.py` (new)

**Interfaces:**
- Consumes: `personalization_cache.invalidate(user_id)` from `deep_agent/src/cache/personalization_cache.py:88`
- Produces: `delete_memory()` and `delete_rule()` now invalidate cache after successful DELETE

- [ ] **Step 1: Write test for cache invalidation on memory delete**

```python
# tests/unit/test_user_isolation.py
"""User isolation tests — prove data boundaries between users."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.src.personalization.repository import PersonalizationRepository

@pytest.fixture(autouse=True)
def _reset_tables_flag():
    import deep_agent.src.personalization.repository as repo_mod
    repo_mod._TABLES_ENSURED = True
    yield
    repo_mod._TABLES_ENSURED = False


@pytest.fixture
def repo():
    return PersonalizationRepository("postgresql://test:test@localhost/testdb")


class TestCacheInvalidation:
    @pytest.mark.asyncio
    async def test_delete_memory_invalidates_cache(self, repo):
        """Deleting a memory must evict the user's personalization cache."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ), patch(
            "deep_agent.src.cache.personalization_cache.invalidate",
            new_callable=AsyncMock,
        ) as mock_invalidate:
            result = await repo.delete_memory("user-a", uuid.uuid4())
            assert result is True
            mock_invalidate.assert_awaited_once_with("user-a")

    @pytest.mark.asyncio
    async def test_delete_memory_no_cache_invalidation_when_not_found(self, repo):
        """No cache invalidation if the memory didn't exist."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 0
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ), patch(
            "deep_agent.src.cache.personalization_cache.invalidate",
            new_callable=AsyncMock,
        ) as mock_invalidate:
            result = await repo.delete_memory("user-a", uuid.uuid4())
            assert result is False
            mock_invalidate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_rule_invalidates_cache(self, repo):
        """Deleting a rule must evict the user's personalization cache."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ), patch(
            "deep_agent.src.cache.personalization_cache.invalidate",
            new_callable=AsyncMock,
        ) as mock_invalidate:
            result = await repo.delete_rule("user-a", uuid.uuid4())
            assert result is True
            mock_invalidate.assert_awaited_once_with("user-a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestCacheInvalidation -v`
Expected: FAIL — `invalidate` is never called

- [ ] **Step 3: Add cache invalidation to delete_memory**

In `deep_agent/src/personalization/repository.py`, replace `delete_memory`:

```python
    async def delete_memory(self, user_id: str, memory_id: uuid.UUID) -> bool:
        """Delete a memory by id; return True if a row was removed."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_memories WHERE id = %s AND user_id = %s",
                (str(memory_id), user_id),
            )
            await conn.commit()
            deleted = bool(cur.rowcount > 0)
        if deleted:
            from deep_agent.src.cache.personalization_cache import invalidate

            await invalidate(user_id)
        return deleted
```

- [ ] **Step 4: Add cache invalidation to delete_rule**

In `deep_agent/src/personalization/repository.py`, replace `delete_rule`:

```python
    async def delete_rule(self, user_id: str, rule_id: uuid.UUID) -> bool:
        """Delete a rule by id; return True if a row was removed."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_rules WHERE id = %s AND user_id = %s",
                (str(rule_id), user_id),
            )
            await conn.commit()
            deleted = bool(cur.rowcount > 0)
        if deleted:
            from deep_agent.src.cache.personalization_cache import invalidate

            await invalidate(user_id)
        return deleted
```

- [ ] **Step 5: Add cache invalidation to consolidation**

In `deep_agent/src/memory/consolidation.py`, after `await conn.commit()` inside `consolidate_user_memories`, add cache invalidation:

```python
        if deleted:
            await conn.commit()
            try:
                from deep_agent.src.cache.personalization_cache import invalidate

                await invalidate(user_id)
            except Exception:
                pass
            logger.info(
                "Consolidated user %s: deleted %d duplicate(s) from %d group(s)",
                user_id[:8],
                deleted,
                len(groups),
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestCacheInvalidation -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run existing repository tests to verify no regressions**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_repository.py -v`
Expected: All existing tests PASS

- [ ] **Step 8: Commit**

```bash
git add deep_agent/src/personalization/repository.py deep_agent/src/memory/consolidation.py tests/unit/test_user_isolation.py
git commit -m "fix: invalidate personalization cache on memory/rule deletion"
```

---

### Task 2: Scope Decay Scoring Per-User

**Files:**
- Modify: `deep_agent/src/memory/scoring.py:61-97`
- Modify: `deep_agent/src/memory/scheduler.py:88-90`
- Test: `tests/unit/test_user_isolation.py` (append)

**Interfaces:**
- Consumes: `memory_settings.MEMORY_DECAY_LAMBDA`, `compute_decay_score()` from same file
- Produces: `decay_user_memories(database_uri, user_id) -> int`, `decay_all_users(database_uri) -> int` replacing `decay_all_memories()`

- [ ] **Step 1: Write test for per-user decay scoping**

Append to `tests/unit/test_user_isolation.py`:

```python
from deep_agent.src.memory.scoring import compute_decay_score


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
            "deep_agent.src.memory.scoring.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ), patch(
            "deep_agent.src.memory.scoring.memory_settings"
        ) as mock_settings:
            mock_settings.is_enabled.return_value = True
            mock_settings.MEMORY_DECAY_LAMBDA = 0.1

            await decay_user_memories("postgresql://test", "user-a")

            sql_call = mock_conn.execute.call_args_list[0]
            query = sql_call[0][0]
            params = sql_call[0][1]
            assert "WHERE user_id" in query
            assert params == ("user-a",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestDecayScoping -v`
Expected: FAIL — `decay_user_memories` does not exist

- [ ] **Step 3: Implement per-user decay functions**

Replace `decay_all_memories` in `deep_agent/src/memory/scoring.py` (lines 61-97) with:

```python
async def decay_user_memories(database_uri: str, user_id: str) -> int:
    """Recalculate scores for a single user's memories.

    Returns the number of memories updated.
    """
    import psycopg
    from psycopg.rows import dict_row

    now = datetime.now(timezone.utc)
    updated = 0

    async with await psycopg.AsyncConnection.connect(
        database_uri, row_factory=dict_row
    ) as conn:
        cur = await conn.execute(
            "SELECT id, score, updated_at FROM user_memories WHERE user_id = %s",
            (user_id,),
        )
        rows = await cur.fetchall()

        for row in rows:
            old_score = float(row.get("score", 1.0) or 1.0)
            new_score = compute_decay_score(old_score, row["updated_at"], now)

            if abs(new_score - old_score) > 0.001:
                await conn.execute(
                    "UPDATE user_memories SET score = %s WHERE id = %s",
                    (new_score, str(row["id"])),
                )
                updated += 1

        if updated:
            await conn.commit()

    logger.info(
        "Decay scoring for user %s: updated %d / %d memories",
        user_id[:8],
        updated,
        len(rows),
    )
    return updated


async def decay_all_users(database_uri: str) -> int:
    """Recalculate decay scores across all users. Returns total updates."""
    import psycopg

    if not memory_settings.is_enabled("decay"):
        logger.debug("Memory decay disabled — skipping")
        return 0

    async with await psycopg.AsyncConnection.connect(database_uri) as conn:
        cur = await conn.execute("SELECT DISTINCT user_id FROM user_memories")
        user_ids = [row[0] for row in await cur.fetchall()]

    total = 0
    for uid in user_ids:
        total += await decay_user_memories(database_uri, uid)

    logger.info(
        "Decay scoring complete: %d updates across %d users",
        total,
        len(user_ids),
    )
    return total
```

- [ ] **Step 4: Update scheduler to call decay_all_users**

In `deep_agent/src/memory/scheduler.py`, replace lines 87-90:

```python
    try:
        from deep_agent.src.memory.scoring import decay_all_users

        results["decay"] = await decay_all_users(database_uri)
    except Exception:
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestDecayScoping -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_agent/src/memory/scoring.py deep_agent/src/memory/scheduler.py tests/unit/test_user_isolation.py
git commit -m "fix: scope memory decay scoring per-user instead of global query"
```

---

### Task 3: Validate Feedback user_id from JWT

**Files:**
- Modify: `deep_agent/aegra/feedback.py:46-72` (_persist_feedback_to_postgres, feedback_handler)
- Modify: `deep_agent/aegra/feedback.py:227-237` (get_thread_feedback)
- Test: `tests/unit/test_user_isolation.py` (append)

**Interfaces:**
- Consumes: `_authenticated_user_id(request)` pattern from `deep_agent/aegra/mcp_routes.py:14-28`
- Produces: Feedback endpoints that extract `user_id` from JWT, not from query params or body

- [ ] **Step 1: Write test for feedback endpoint user_id validation**

Append to `tests/unit/test_user_isolation.py`:

```python
class TestFeedbackUserValidation:
    @pytest.mark.asyncio
    async def test_get_feedback_extracts_user_from_jwt(self):
        """GET /feedback/{thread_id} must use JWT identity, not query param."""
        from deep_agent.aegra.feedback import get_thread_feedback

        mock_request = AsyncMock()
        mock_request.headers = {"authorization": "Bearer fake-token"}

        with patch(
            "deep_agent.aegra.feedback._authenticated_user_id",
            new_callable=AsyncMock,
            return_value="jwt-user-a",
        ) as mock_auth, patch(
            "deep_agent.aegra.feedback.settings"
        ) as mock_settings, patch(
            "deep_agent.aegra.feedback.FeedbackRepository"
        ) as MockRepo:
            mock_settings.database_uri = "postgresql://test"
            mock_repo_inst = AsyncMock()
            mock_repo_inst.list_feedback = AsyncMock(return_value=[])
            MockRepo.return_value = mock_repo_inst

            result = await get_thread_feedback(
                "550e8400-e29b-41d4-a716-446655440000",
                mock_request,
            )

            mock_auth.assert_awaited_once_with(mock_request)
            mock_repo_inst.list_feedback.assert_awaited_once_with(
                "550e8400-e29b-41d4-a716-446655440000", "jwt-user-a"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestFeedbackUserValidation -v`
Expected: FAIL — `get_thread_feedback` doesn't take `request` param

- [ ] **Step 3: Add _authenticated_user_id helper to feedback.py**

Add this function near the top of `deep_agent/aegra/feedback.py` (after the imports):

```python
async def _authenticated_user_id(request: Request) -> str:
    """Return the SSO sub from the incoming Bearer token."""
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return "anonymous"

    payload = _decode_token(auth_header[7:])
    return str(payload["sub"])
```

- [ ] **Step 4: Update GET /feedback/{thread_id} to use JWT identity**

Replace the `get_thread_feedback` function:

```python
@feedback_router.get("/feedback/{thread_id}")
async def get_thread_feedback(
    thread_id: str, request: Request
) -> dict[str, Any]:
    """Return all feedback for a thread, scoped to the authenticated user."""
    thread_id = _validate_thread_id(thread_id)
    user_id = await _authenticated_user_id(request)
    if not settings.database_uri:
        return {"feedback": []}
    repo = FeedbackRepository(settings.database_uri)
    items = await repo.list_feedback(thread_id, user_id)
    return {"feedback": items}
```

- [ ] **Step 5: Update POST /feedback to use JWT identity**

In `_persist_feedback_to_postgres`, the `user_id` should come from the authenticated user. Modify `feedback_handler` to extract user_id and pass it. In `record_feedback`, add a `request` parameter:

Replace `feedback_handler` function:

```python
async def feedback_handler(request: Request) -> JSONResponse:
    """ASGI/Starlette handler: read JSON, validate, record feedback."""
    try:
        body_bytes = await request.body()
        if not body_bytes.strip():
            return JSONResponse(
                status_code=422,
                content={"detail": [{"msg": "Empty body", "type": "value_error"}]},
            )
        payload = json.loads(body_bytes.decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{"msg": "Invalid JSON body", "type": "json_invalid"}],
            },
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "msg": "JSON body must be an object",
                        "type": "type_error",
                    },
                ],
            },
        )

    jwt_user_id = await _authenticated_user_id(request)
    if jwt_user_id != "anonymous":
        payload["user_id"] = jwt_user_id

    try:
        resp = await record_feedback(payload)
    except ValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(include_url=False)},
        )
    except Exception:
        logger.exception("feedback_handler_error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return JSONResponse(
        status_code=200,
        content=resp.model_dump(),
    )
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py::TestFeedbackUserValidation -v`
Expected: PASS

- [ ] **Step 7: Run existing feedback tests for regressions**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/feedback/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add deep_agent/aegra/feedback.py tests/unit/test_user_isolation.py
git commit -m "fix: extract feedback user_id from JWT instead of query param"
```

---

### Task 4: Personalization REST API + Bulk Delete

**Files:**
- Create: `deep_agent/aegra/personalization_routes.py`
- Modify: `deep_agent/aegra/http_app.py:102-103` (mount router)
- Modify: `deep_agent/src/personalization/repository.py` (add bulk delete methods)
- Test: `tests/unit/test_personalization_routes.py` (new)

**Interfaces:**
- Consumes: `PersonalizationRepository` from `deep_agent/src/personalization/repository.py`, `_authenticated_user_id` pattern from `deep_agent/aegra/mcp_routes.py:14-28`
- Produces: REST endpoints `GET/POST/DELETE /memories`, `GET/POST/DELETE /rules`

- [ ] **Step 1: Add bulk delete methods to PersonalizationRepository**

Append to `deep_agent/src/personalization/repository.py`:

```python
    async def delete_all_memories(self, user_id: str) -> int:
        """Delete all memories for a user; return count of deleted rows."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_memories WHERE user_id = %s",
                (user_id,),
            )
            await conn.commit()
            count = cur.rowcount
        if count > 0:
            from deep_agent.src.cache.personalization_cache import invalidate

            await invalidate(user_id)
        return count

    async def delete_all_rules(self, user_id: str) -> int:
        """Delete all rules for a user; return count of deleted rows."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_rules WHERE user_id = %s",
                (user_id,),
            )
            await conn.commit()
            count = cur.rowcount
        if count > 0:
            from deep_agent.src.cache.personalization_cache import invalidate

            await invalidate(user_id)
        return count
```

- [ ] **Step 2: Create personalization_routes.py**

Create `deep_agent/aegra/personalization_routes.py`:

```python
"""REST API for user memories and rules (per-user, JWT-authenticated)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from deep_agent.src.personalization.repository import PersonalizationRepository
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

personalization_router = APIRouter(tags=["personalization"])


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class RuleCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    is_active: bool = True


async def _authenticated_user_id(request: Request) -> str:
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    payload = _decode_token(auth_header[7:])
    return str(payload["sub"])


def _get_repo() -> PersonalizationRepository:
    if not settings.database_uri:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return PersonalizationRepository(settings.database_uri)


@personalization_router.get("/memories")
async def list_memories(request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    memories = await repo.list_memories(user_id)
    return {
        "memories": [
            {
                "id": str(m.id),
                "content": m.content,
                "score": m.score,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    }


@personalization_router.post("/memories", status_code=201)
async def create_memory(body: MemoryCreateRequest, request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    mem = await repo.create_memory(user_id, body.content)
    from deep_agent.src.cache.personalization_cache import invalidate

    await invalidate(user_id)
    return {"id": str(mem.id), "content": mem.content}


@personalization_router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request) -> dict[str, Any]:
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format") from None
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    deleted = await repo.delete_memory(user_id, mid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@personalization_router.delete("/memories")
async def delete_all_memories(request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    count = await repo.delete_all_memories(user_id)
    return {"deleted_count": count}


@personalization_router.get("/rules")
async def list_rules(request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    rules = await repo.list_rules(user_id, active_only=False)
    return {
        "rules": [
            {
                "id": str(r.id),
                "content": r.content,
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat(),
            }
            for r in rules
        ]
    }


@personalization_router.post("/rules", status_code=201)
async def create_rule(body: RuleCreateRequest, request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    rule = await repo.upsert_rule(user_id, body.content, is_active=body.is_active)
    from deep_agent.src.cache.personalization_cache import invalidate

    await invalidate(user_id)
    return {"id": str(rule.id), "content": rule.content, "is_active": rule.is_active}


@personalization_router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str, request: Request) -> dict[str, Any]:
    try:
        rid = UUID(rule_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rule_id format") from None
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    deleted = await repo.delete_rule(user_id, rid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"deleted": True}


@personalization_router.delete("/rules")
async def delete_all_rules(request: Request) -> dict[str, Any]:
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    count = await repo.delete_all_rules(user_id)
    return {"deleted_count": count}
```

- [ ] **Step 3: Mount router in http_app.py**

Add to `deep_agent/aegra/http_app.py` after the existing router includes:

```python
from deep_agent.aegra.personalization_routes import personalization_router
```

And at the bottom (after `app.include_router(feedback_router)`):

```python
app.include_router(personalization_router)
```

- [ ] **Step 4: Write route tests**

Create `tests/unit/test_personalization_routes.py`:

```python
"""Unit tests for personalization REST API routes."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("deep_agent.aegra.personalization_routes.ENABLE_AUTH", False), \
         patch("deep_agent.aegra.personalization_routes.DEV_USER_ID", "test-user"):
        from deep_agent.aegra.personalization_routes import personalization_router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(personalization_router)
        return TestClient(app)


class TestMemoryRoutes:
    def test_list_memories(self, client):
        with patch("deep_agent.aegra.personalization_routes._get_repo") as mock_get:
            mock_repo = AsyncMock()
            mock_repo.list_memories = AsyncMock(return_value=[])
            mock_get.return_value = mock_repo

            resp = client.get("/memories")
            assert resp.status_code == 200
            assert resp.json() == {"memories": []}

    def test_delete_all_memories(self, client):
        with patch("deep_agent.aegra.personalization_routes._get_repo") as mock_get:
            mock_repo = AsyncMock()
            mock_repo.delete_all_memories = AsyncMock(return_value=3)
            mock_get.return_value = mock_repo

            resp = client.delete("/memories")
            assert resp.status_code == 200
            assert resp.json() == {"deleted_count": 3}
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_personalization_routes.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_agent/aegra/personalization_routes.py deep_agent/aegra/http_app.py deep_agent/src/personalization/repository.py tests/unit/test_personalization_routes.py
git commit -m "feat: add personalization REST API for memory/rule CRUD with JWT auth"
```

---

### Task 5: User Isolation Integration Tests (In-Memory Fake DB)

**Files:**
- Modify: `tests/unit/test_user_isolation.py` (add remaining isolation tests)

**Interfaces:**
- Consumes: `PersonalizationRepository`, `FeedbackRepository`, `inject_personalization`, `decay_user_memories`
- Produces: 12 total tests proving user data isolation

- [ ] **Step 1: Add in-memory fake DB and remaining isolation tests**

Append to `tests/unit/test_user_isolation.py`:

```python
import asyncio
import re
from datetime import datetime, timezone

from deep_agent.src.feedback.repository import FeedbackRepository
from deep_agent.src.personalization.injector import inject_personalization


class FakeCursor:
    """In-memory cursor that simulates psycopg cursor results."""

    def __init__(self):
        self.rows = []
        self.rowcount = 0

    async def fetchall(self):
        return self.rows


class FakeDB:
    """In-memory database simulating PostgreSQL with WHERE filtering."""

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
        query_lower = query.strip().lower()

        if query_lower.startswith("insert into user_memories"):
            row = {
                "id": params[0], "user_id": params[1], "content": params[2],
                "score": 1.0, "cluster_id": None,
                "created_at": params[3] if len(params) > 3 else datetime.now(timezone.utc),
                "updated_at": params[4] if len(params) > 4 else datetime.now(timezone.utc),
            }
            self.tables["user_memories"].append(row)

        elif query_lower.startswith("insert into user_rules"):
            row = {
                "id": params[0], "user_id": params[1], "content": params[2],
                "is_active": params[3], "created_at": params[4], "updated_at": params[5],
            }
            self.tables["user_rules"].append(row)

        elif query_lower.startswith("insert into message_feedback"):
            existing = [
                r for r in self.tables["message_feedback"]
                if r["thread_id"] == params[0]
                and r["message_id"] == params[1]
                and r["user_id"] == params[2]
            ]
            if existing:
                existing[0]["feedback"] = params[3]
                existing[0]["trace_id"] = params[4]
            else:
                self.tables["message_feedback"].append({
                    "thread_id": params[0], "message_id": params[1],
                    "user_id": params[2], "feedback": params[3],
                    "trace_id": params[4],
                })

        elif query_lower.startswith("select") and "user_memories" in query_lower:
            rows = self.tables["user_memories"]
            if params and "where user_id" in query_lower:
                rows = [r for r in rows if r["user_id"] == params[0]]
            cursor.rows = rows

        elif query_lower.startswith("select") and "user_rules" in query_lower:
            rows = self.tables["user_rules"]
            if params and "where user_id" in query_lower:
                rows = [r for r in rows if r["user_id"] == params[0]]
                if "is_active = true" in query_lower:
                    rows = [r for r in rows if r["is_active"]]
            cursor.rows = rows

        elif query_lower.startswith("select") and "message_feedback" in query_lower:
            rows = self.tables["message_feedback"]
            if params:
                rows = [
                    r for r in rows
                    if r["thread_id"] == params[0] and r["user_id"] == params[1]
                ]
            cursor.rows = [
                {"message_id": r["message_id"], "feedback": r["feedback"]}
                for r in rows
            ]

        elif query_lower.startswith("delete") and "user_memories" in query_lower:
            before = len(self.tables["user_memories"])
            if params and len(params) == 2:
                self.tables["user_memories"] = [
                    r for r in self.tables["user_memories"]
                    if not (r["id"] == params[0] and r["user_id"] == params[1])
                ]
            elif params and len(params) == 1:
                self.tables["user_memories"] = [
                    r for r in self.tables["user_memories"]
                    if r["user_id"] != params[0]
                ]
            cursor.rowcount = before - len(self.tables["user_memories"])

        elif query_lower.startswith("delete") and "user_rules" in query_lower:
            before = len(self.tables["user_rules"])
            if params and len(params) == 2:
                self.tables["user_rules"] = [
                    r for r in self.tables["user_rules"]
                    if not (r["id"] == params[0] and r["user_id"] == params[1])
                ]
            elif params and len(params) == 1:
                self.tables["user_rules"] = [
                    r for r in self.tables["user_rules"]
                    if r["user_id"] != params[0]
                ]
            cursor.rowcount = before - len(self.tables["user_rules"])

        elif query_lower.startswith("delete") and "message_feedback" in query_lower:
            before = len(self.tables["message_feedback"])
            self.tables["message_feedback"] = [
                r for r in self.tables["message_feedback"]
                if not (r["thread_id"] == params[0] and r["message_id"] == params[1] and r["user_id"] == params[2])
            ]
            cursor.rowcount = before - len(self.tables["message_feedback"])

        return cursor


@pytest.fixture
def fake_db():
    return FakeDB()


@pytest.fixture
def patched_repos(fake_db):
    """Patch psycopg connections for both repos to use the fake DB."""
    import deep_agent.src.personalization.repository as p_mod
    import deep_agent.src.feedback.repository as f_mod

    p_mod._TABLES_ENSURED = True
    f_mod._TABLE_ENSURED = True

    with patch(
        "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
        side_effect=lambda *a, **kw: asyncio.coroutine(lambda: fake_db.make_connection())(),
    ), patch(
        "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
        side_effect=lambda *a, **kw: asyncio.coroutine(lambda: fake_db.make_connection())(),
    ), patch(
        "deep_agent.src.cache.personalization_cache.invalidate",
        new_callable=AsyncMock,
    ):
        p_repo = PersonalizationRepository("postgresql://test")
        f_repo = FeedbackRepository("postgresql://test")
        yield p_repo, f_repo, fake_db


class TestMemoryIsolation:
    @pytest.mark.asyncio
    async def test_memory_read_isolation(self, patched_repos):
        p_repo, _, fake_db = patched_repos
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
        p_repo, _, fake_db = patched_repos
        m1 = await p_repo.create_memory("user-a", "Memory A1")
        m2 = await p_repo.create_memory("user-a", "Memory A2")
        m3 = await p_repo.create_memory("user-b", "Memory B1")

        a_mems = await p_repo.list_memories("user-a")
        b_mems = await p_repo.list_memories("user-b")

        assert len(a_mems) == 2
        assert len(b_mems) == 1
        a_contents = {m.content for m in a_mems}
        assert a_contents == {"Memory A1", "Memory A2"}
        assert b_mems[0].content == "Memory B1"


class TestRuleIsolation:
    @pytest.mark.asyncio
    async def test_rule_read_isolation(self, patched_repos):
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
        p_repo, _, _ = patched_repos
        await p_repo.upsert_rule("user-a", "Rule A1")
        await p_repo.upsert_rule("user-a", "Rule A2")
        await p_repo.upsert_rule("user-b", "Rule B1")

        a_rules = await p_repo.list_rules("user-a", active_only=False)
        b_rules = await p_repo.list_rules("user-b", active_only=False)

        assert len(a_rules) == 2
        assert len(b_rules) == 1


class TestFeedbackIsolation:
    @pytest.mark.asyncio
    async def test_feedback_per_user_on_same_message(self, patched_repos):
        _, f_repo, _ = patched_repos
        await f_repo.upsert_feedback("t1", "msg1", "user-a", "up", None)
        await f_repo.upsert_feedback("t1", "msg1", "user-b", "down", None)

        a_fb = await f_repo.list_feedback("t1", "user-a")
        b_fb = await f_repo.list_feedback("t1", "user-b")

        assert len(a_fb) == 1
        assert a_fb[0]["feedback"] == "up"
        assert len(b_fb) == 1
        assert b_fb[0]["feedback"] == "down"


class TestCrossUserDeleteBlocked:
    @pytest.mark.asyncio
    async def test_cross_user_memory_delete_blocked(self, patched_repos):
        p_repo, _, fake_db = patched_repos
        mem = await p_repo.create_memory("user-a", "Secret memory")

        result = await p_repo.delete_memory("user-b", mem.id)
        assert result is False

        a_mems = await p_repo.list_memories("user-a")
        assert len(a_mems) == 1
        assert a_mems[0].content == "Secret memory"

    @pytest.mark.asyncio
    async def test_cross_user_rule_delete_blocked(self, patched_repos):
        p_repo, _, _ = patched_repos
        rule = await p_repo.upsert_rule("user-a", "Private rule")

        result = await p_repo.delete_rule("user-b", rule.id)
        assert result is False

        a_rules = await p_repo.list_rules("user-a")
        assert len(a_rules) == 1


class TestHardDelete:
    @pytest.mark.asyncio
    async def test_hard_delete_removes_data(self, patched_repos):
        p_repo, _, fake_db = patched_repos
        mem = await p_repo.create_memory("user-a", "To be deleted")
        assert len(await p_repo.list_memories("user-a")) == 1

        result = await p_repo.delete_memory("user-a", mem.id)
        assert result is True

        remaining = await p_repo.list_memories("user-a")
        assert len(remaining) == 0
        assert len(fake_db.tables["user_memories"]) == 0


class TestConcurrentSessions:
    @pytest.mark.asyncio
    async def test_concurrent_writes_no_cross_contamination(self, patched_repos):
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


class TestPersonalizationInjectionScoping:
    def test_injection_contains_only_requesting_user_data(self):
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
```

- [ ] **Step 2: Run all isolation tests**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/test_user_isolation.py -v`
Expected: All 12+ tests PASS

- [ ] **Step 3: Run full unit test suite**

Run: `cd /Users/nsaharan/Desktop/template-agent && python -m pytest tests/unit/ -v --tb=short`
Expected: All tests PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_user_isolation.py
git commit -m "test: add 12 user isolation tests with in-memory fake DB"
```

---

### Task 6: Connect UI Memory Deletion to Backend API (template-ui)

**Files:**
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/services/agent-rest.ts` (add memory/rule API functions)
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/redux/slices/personalization.ts` (add async thunks)
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/components/settings/MemoryList.tsx` (call backend on delete)
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/components/settings/RulesEditor.tsx` (call backend on delete)

**Interfaces:**
- Consumes: `GET/POST/DELETE /memories`, `GET/POST/DELETE /rules` from Task 4
- Produces: UI memory/rule operations sync with backend

> **Note:** This task modifies template-ui. Read target files first to understand the exact Redux/component patterns before editing. The implementation should follow the existing patterns for thread deletion (`deleteThread()` in `agent-rest.ts`).

- [ ] **Step 1: Add API functions to agent-rest.ts**

Read `agent-rest.ts` to find existing patterns, then add after the `deleteThread` function:

```typescript
export async function listMemories(): Promise<Array<{id: string; content: string; score: number; created_at: string}>> {
  try {
    const resp = await authenticatedFetch(buildAgentApiUrl('/memories'), {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.memories || [];
  } catch {
    return [];
  }
}

export async function deleteAllMemories(): Promise<boolean> {
  try {
    const resp = await authenticatedFetch(buildAgentApiUrl('/memories'), {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

export async function deleteMemory(memoryId: string): Promise<boolean> {
  try {
    const resp = await authenticatedFetch(buildAgentApiUrl(`/memories/${memoryId}`), {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return resp.ok || resp.status === 404;
  } catch {
    return false;
  }
}

export async function deleteAllRules(): Promise<boolean> {
  try {
    const resp = await authenticatedFetch(buildAgentApiUrl('/rules'), {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

export async function deleteRule(ruleId: string): Promise<boolean> {
  try {
    const resp = await authenticatedFetch(buildAgentApiUrl(`/rules/${ruleId}`), {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return resp.ok || resp.status === 404;
  } catch {
    return false;
  }
}
```

- [ ] **Step 2: Update MemoryList.tsx to call backend on delete**

Read `MemoryList.tsx` to understand the existing delete flow, then modify the delete handlers to also call the backend API. Add fire-and-forget calls matching the `deleteThread` pattern:

```typescript
import { deleteMemory, deleteAllMemories } from '../../services/agent-rest';

// In the remove handler (after dispatch(removeMemory(id))):
deleteMemory(id).catch(() => {});

// In the clear-all handler (after dispatch(clearMemories())):
deleteAllMemories().catch(() => {});
```

- [ ] **Step 3: Update RulesEditor.tsx to call backend on delete**

Same pattern:

```typescript
import { deleteRule, deleteAllRules } from '../../services/agent-rest';

// In the remove handler (after dispatch(removeRule(id))):
deleteRule(id).catch(() => {});

// In the clear-all handler (after dispatch(clearRules())):
deleteAllRules().catch(() => {});
```

- [ ] **Step 4: Commit in template-ui repo**

```bash
cd /Users/nsaharan/Desktop/template-ui
git add src/frontend/services/agent-rest.ts src/frontend/components/settings/MemoryList.tsx src/frontend/components/settings/RulesEditor.tsx
git commit -m "feat: connect memory/rule deletion to backend API"
```

---

### Task 7: Align user_id Between UI and Backend

**Files:**
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/pages/ChatPage.tsx`
- Modify: `/Users/nsaharan/Desktop/template-ui/src/frontend/components/layout/AppLayout.tsx`

**Interfaces:**
- Consumes: `window.USER_DATA.sub` (JWT subject claim)
- Produces: All backend-facing user identification uses JWT `sub` instead of `preferred_username`

> **Note:** Read each file first to find the exact lines where `preferred_username` is used for backend calls. Keep `preferred_username` for display-only purposes (e.g., showing the username in UI).

- [ ] **Step 1: Read ChatPage.tsx to find feedbackUserId assignment**

Find where `feedbackUserId` or similar is set from `preferred_username` and change to use `sub` for API calls.

- [ ] **Step 2: Read AppLayout.tsx to find userId for thread search**

Find where `userId` for `getAllThreadsByUserId` is set and change to use `sub`.

- [ ] **Step 3: Update both files to use sub for backend calls**

In both files, replace patterns like:
```typescript
const userId = window.USER_DATA?.preferred_username || 'anonymous';
```
with:
```typescript
const userId = window.USER_DATA?.sub || window.USER_DATA?.preferred_username || 'anonymous';
```

This prefers `sub` (the JWT subject claim that matches backend `user.identity`) but falls back to `preferred_username` if `sub` is not available.

- [ ] **Step 4: Commit in template-ui repo**

```bash
cd /Users/nsaharan/Desktop/template-ui
git add src/frontend/pages/ChatPage.tsx src/frontend/components/layout/AppLayout.tsx
git commit -m "fix: use JWT sub claim for backend user identification instead of preferred_username"
```
