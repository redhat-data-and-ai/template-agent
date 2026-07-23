# User Isolation Locust Load Tests — Design Spec

**Date**: 2026-07-23
**Status**: Draft
**Relates to**: [User Isolation Design](2026-07-07-user-isolation-design.md)

## Purpose

Verify that user isolation holds under concurrent multi-user load. Multiple simulated users hit the template-agent API simultaneously — each creating, reading, and deleting memories, rules, threads, chat messages, feedback, and checkpoints. The tests assert that NO user ever sees another user's data.

This is correctness verification under concurrency, not performance benchmarking.

## Architecture

### Auth: Self-signed JWTs

Each Locust user authenticates with a unique self-signed JWT. No dev-mode bypass — the full auth path is exercised.

**Components:**

1. **RSA key pair** — generated once at module load (RSA-2048)
2. **JWKS server** — lightweight HTTP server on a random port in a daemon thread, serves `GET /jwks` with the public key in JWK format
3. **Token factory** — `create_user_token(user_id: str) → str` signs a JWT with:
   - `sub`: the unique user_id
   - `iss`: `locust-isolation-test`
   - `exp`: now + 1 hour
   - `iat`: now
   - `name`: `Locust {user_id}`

**App configuration** (set by the locustfile entry point):

```
ENABLE_AUTH=true
SSO_JWKS_URI=http://localhost:{JWKS_PORT}/jwks
SSO_JWT_AUDIENCE=           # empty — skip audience check
SSO_ISSUER_URL=locust-isolation-test
```

The app's real `_decode_token()` validates every request through the full JWKS → RSA verification path.

### Data Markers

Each user tags all created content with `[locust:{user_id}]` as a prefix. When listing resources, every returned item's `content` field is checked — if any item does NOT contain `[locust:{self._user_id}]`, an isolation violation is raised.

### Canary Resources

On `on_start`, each user creates one persistent "canary" memory and one "canary" rule. These serve as:
- Targets for cross-user delete attempts (another user tries to delete them — expects 404)
- Persistent fixtures that should remain visible ONLY to their owner throughout the test

A thread-safe `canary_registry` dict maps `user_id → {memory_id, rule_id}` so users can pick random canary IDs from other users for cross-delete attempts. This is the only shared mutable state.

## File Structure

```
tests/load/
├── conftest.py                    # existing — shared helpers
├── locustfile.py                  # existing — performance tests (unchanged)
├── isolation_locustfile.py        # NEW — entry point for isolation tests
├── jwt_provider.py                # NEW — RSA keygen + JWKS server + token factory
└── scenarios/
    ├── single_turn.py             # existing (unchanged)
    ├── multi_turn.py              # existing (unchanged)
    └── user_isolation.py          # NEW — IsolatedUser class with all tasks
```

## Test Scenarios

One `IsolatedUser` Locust class runs the following tasks:

| Task | Weight | Description | Isolation Check |
|------|--------|-------------|-----------------|
| `memory_lifecycle` | 3 | Create memory with marker → list → verify only own → delete → verify gone | List must contain ONLY items with own user marker |
| `rule_lifecycle` | 3 | Create rule → list → verify only own → delete → verify gone | Same marker check |
| `chat_and_thread` | 2 | Create thread → send message → stream response → list threads → verify only own threads visible | Thread list must contain ONLY own thread IDs |
| `feedback_lifecycle` | 2 | Submit feedback on own thread → get feedback → verify only own | Feedback list scoped to own user |
| `cross_user_delete` | 1 | Pick a random canary resource from another user → attempt DELETE → expect 404 | Must NOT return 200 |
| `thread_cleanup` | 1 | Delete own thread via cascading DELETE → verify thread gone from list | Own data removed, others' canaries untouched |

### Task Details

**memory_lifecycle:**
1. `POST /memories` with body `{"content": "[locust:{user_id}] memory {uuid}"}`
2. `GET /memories` → parse response, assert every item's `content` starts with `[locust:{user_id}]`
3. `DELETE /memories/{id}` for the created memory
4. `GET /memories` → assert the deleted memory is gone

**rule_lifecycle:**
1. `POST /rules` with body `{"content": "[locust:{user_id}] rule {uuid}"}`
2. `GET /rules` → assert every item's `content` starts with `[locust:{user_id}]`
3. `DELETE /rules/{id}` for the created rule
4. `GET /rules` → assert deleted rule is gone

**chat_and_thread:**
1. `POST /threads` → capture `thread_id`
2. `POST /threads/{id}/runs/stream` with a short prompt → consume SSE stream, verify 200
3. `GET /threads` → assert every returned thread belongs to this user (thread IDs tracked locally)
4. Store `thread_id` for use by `feedback_lifecycle` and `thread_cleanup`

**feedback_lifecycle:**
1. Requires a `thread_id` from a prior `chat_and_thread` run
2. `POST /feedback` with `thread_id`, `message_id`, thumbs up/down
3. `GET /feedback/{thread_id}` → assert only own feedback returned

**cross_user_delete:**
1. Pick a random `(other_user_id, memory_id)` from `canary_registry` where `other_user_id != self._user_id`
2. `DELETE /memories/{memory_id}` with own JWT → expect 404
3. Optionally try `DELETE /rules/{rule_id}` from another user → expect 404
4. If either returns 200, fire an isolation violation

**thread_cleanup:**
1. Requires a `thread_id` from a prior `chat_and_thread` run
2. `DELETE /threads/{thread_id}` → expect 200
3. `GET /threads` → assert the deleted thread is gone
4. Canary resources from other users remain unaffected (verified indirectly via `cross_user_delete`)

## IsolatedUser Lifecycle

```
on_start:
  1. Generate unique user_id: f"locust-{n}-{hex(4)}"
  2. Sign JWT with user_id as sub claim
  3. Set Authorization header for all requests
  4. Create canary memory and canary rule
  5. Register canary IDs in shared canary_registry

tasks (weighted random):
  - memory_lifecycle (3)
  - rule_lifecycle (3)
  - chat_and_thread (2)
  - feedback_lifecycle (2)
  - cross_user_delete (1)
  - thread_cleanup (1)

on_stop:
  1. DELETE /memories (bulk) — clean up own memories
  2. DELETE /rules (bulk) — clean up own rules
  3. Remove self from canary_registry
```

## Failure Reporting

- **Isolation violations** fire as `events.request.fire(request_type="ISOLATION", name="violation:{resource}", exception=IsolationViolationError(...))` — visible in Locust's failure table with details on which user saw what foreign data
- **Cross-user delete success** fires as `events.request.fire(request_type="ISOLATION", name="cross_delete_succeeded", exception=...)`
- Standard HTTP errors (401, 500) reported through Locust's `catch_response` mechanism
- At `test_stop`, if any isolation violation occurred, log a critical summary line

## Load Profiles

| Profile | Users | Spawn Rate | Duration | Use Case |
|---------|-------|-----------|----------|----------|
| smoke | 5 | 1/s | 2 min | Quick CI validation |
| thorough | 10 | 1/s | 5 min | Pre-merge confidence |

Configurable via `LOAD_PROFILE` env var (defaults to `smoke`).

## Running

```bash
# Smoke test (headless)
LOAD_PROFILE=smoke locust -f tests/load/isolation_locustfile.py \
    --headless --host http://localhost:8123

# With Locust UI
locust -f tests/load/isolation_locustfile.py --host http://localhost:8123

# Thorough test
LOAD_PROFILE=thorough locust -f tests/load/isolation_locustfile.py \
    --headless --host http://localhost:8123
```

The JWKS server starts automatically when the module loads. The app must be configured to point at it:

```bash
export ENABLE_AUTH=true
export SSO_JWKS_URI=http://localhost:{JWKS_PORT}/jwks
export SSO_ISSUER_URL=locust-isolation-test
```

The `isolation_locustfile.py` prints the required env vars on startup.

## Dependencies

- `locust` (already in venv)
- `PyJWT` (already used by the app)
- `cryptography` (for RSA key generation — already a transitive dependency of PyJWT)

No new dependencies required.

## Success Criteria

- Zero isolation violations across all runs
- Every list endpoint returns ONLY data tagged with the requesting user's marker
- Every cross-user delete attempt returns 404
- Chat streaming works per-user with thread isolation
- All canary resources survive the full test duration (not deleted by other users)
