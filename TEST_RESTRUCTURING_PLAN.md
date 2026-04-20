# Test Restructuring Plan (Clean Slate)

## Philosophy

**Unit Tests**: Fast, isolated, prefer fake data over mocks
- **Fake data first**: Use real objects with fake/temp inputs where possible
  - Real Python objects (MessageDeduplicator, ToolCallTracker, LangChain messages)
  - Temp directories with fake YAML/config files
  - In-memory databases (SQLite instead of PostgreSQL)
- **Mock only external services**: LLM APIs, MCP servers, external HTTP calls
- **Test individual module logic in isolation**
- **One clear goal per test**: Each test should verify one specific behavior

**Skills Tests**: Minimal agent with auto-discovery
- Test each skill with minimal LLM (no MCP tools needed)
- Auto-discovers all skills from agent_config/skills/
- Runs evals.json assertions for each skill
- Add new skill → automatically tested

## Test Quality Principles

**Each test should have a single, clear purpose**:
- ✅ `test_deduplicator_marks_message_as_seen()` - clear goal
- ✅ `test_deduplicator_filters_duplicate_ids()` - clear goal
- ❌ `test_deduplicator()` - unclear, probably tests too much

**Test names should describe the expected behavior**:
- ✅ `test_manager_initializes_with_deduplicator_and_tracker()`
- ✅ `test_config_loads_skills_from_yaml_files()`
- ❌ `test_manager()` - too vague
- ❌ `test_config_works()` - not descriptive

**Keep tests focused and minimal**:
- Only test one behavior per test function
- Minimal setup - only what's needed for that specific test
- Single assertion or related group of assertions
- No complex logic in tests - tests should be obvious to read

**Avoid bloated tests**:
- ❌ Don't test multiple unrelated behaviors in one test
- ❌ Don't copy-paste large setup blocks (use fixtures instead)
- ❌ Don't test implementation details (test public API only)
- ❌ Don't write tests that just mock everything and assert mock calls

**Good example** (focused, clear):
```python
def test_deduplicator_marks_first_message_as_unseen():
    """First time seeing a message, it should be marked as unseen."""
    dedup = MessageDeduplicator()
    msg = AIMessage(content="Hello", id="msg_1")

    assert not dedup.is_seen(msg)
```

**Bad example** (bloated, unclear):
```python
def test_deduplicator():
    """Test deduplicator."""  # ❌ Vague docstring
    # ❌ Tests 5 different behaviors in one test
    dedup = MessageDeduplicator()
    msg1 = AIMessage(content="Hello", id="msg_1")
    msg2 = AIMessage(content="World", id="msg_2")

    assert not dedup.is_seen(msg1)
    dedup.mark_seen(msg1)
    assert dedup.is_seen(msg1)
    assert not dedup.is_seen(msg2)
    dedup.reset()
    assert not dedup.is_seen(msg1)
    # ❌ Should be 5 separate tests!
```

## Target Structure

```
tests/
├── unit/                           # Fast, isolated, fake data over mocks
│   ├── agent/
│   │   ├── test_llm.py            # Mock: Google/Anthropic credentials
│   │   ├── test_manager.py        # Real: MessageDeduplicator/ToolCallTracker instances
│   │   └── config/
│   │       └── test_config.py     # Real: AgentConfig with temp dirs + fake YAML files
│   ├── infrastructure/
│   │   ├── test_backend.py        # Real: Backend with temp directories
│   │   ├── test_checkpointer.py   # Real: SQLite in-memory DB (instead of PostgreSQL mocks)
│   │   ├── test_mcp.py           # Mock: External MCP server connections
│   │   └── test_subagents.py     # Real: Load from temp dirs + fake YAML files
│   ├── adapters/
│   │   └── test_langchain.py     # Real: LangChain message objects (AIMessage, ToolMessage)
│   ├── streaming/
│   │   └── test_streaming.py     # Real: All streaming objects + fake LangChain messages
│   ├── api/
│   │   ├── routes/
│   │   │   └── test_routes.py    # Mock: Langfuse client, checkpointer; Real: FastAPI app
│   │   └── test_app.py           # Real: App creation; Mock: External services
│   └── test_exceptions.py        # Real: Exception classes with fake error scenarios
│
├── skills/                         # Skills with minimal agent (auto-discovery)
│   ├── test_skills.py            # Real: Agent execution with fake evals.json scenarios
│   ├── conftest.py               # Real: Minimal agent fixture + skill auto-discovery
│   └── llm_judge.py              # Real: LLM judge for assertion evaluation
│
└── conftest.py                    # Shared pytest configuration
```

## Skills Testing Strategy

**Auto-discovery pattern**: Test discovers all skills and their evals automatically.

**Key insight**: Skills are self-contained utilities with NO tool dependencies:
- `client-intake` → uses local `scripts/convert_units.py`
- `bmi-report` → reads markdown references
- `email-formatter` → reads HTML templates
- No MCP tools needed for skill testing!

```python
# tests/skills/conftest.py
def pytest_generate_tests(metafunc):
    """Discover all skills and their evals dynamically."""
    if "skill_name" in metafunc.fixturenames:
        skills_dir = Path("agent_config/skills")
        skill_names = [s.name for s in skills_dir.iterdir() if s.is_dir()]
        metafunc.parametrize("skill_name", skill_names)

@pytest.fixture
def minimal_model():
    """Minimal LLM for skill testing."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

# tests/skills/test_skills.py
def test_skill_with_evals(skill_name, minimal_model, llm_judge):
    """Generic test that runs all evals for a skill."""
    evals_file = Path(f"agent_config/skills/{skill_name}/evals/evals.json")

    if not evals_file.exists():
        pytest.skip(f"No evals.json for {skill_name}")

    evals = json.loads(evals_file.read_text())

    for eval_case in evals["evals"]:
        # Create minimal agent with ONLY this skill (no tools needed!)
        agent = create_deep_agent(
            model=minimal_model,
            skills=[f"agent_config/skills/{skill_name}"],
            tools=[],  # ← Skills don't need tools!
            checkpointer=MemorySaver(),
        )

        # Run agent with eval prompt
        result = agent.invoke({
            "messages": [{"role": "user", "content": eval_case["prompt"]}]
        })

        # Verify assertions using LLM judge
        for assertion in eval_case["assertions"]:
            passed = llm_judge.evaluate(assertion, result, eval_case)
            assert passed, f"Failed: {assertion}"
```

**Benefits**:
- Add new skill → tests automatically pick it up
- Add new eval to existing skill → tests automatically run it
- No mock tools needed (skills are tool-free)
- Fast execution (minimal agent, no MCP overhead)
- Single test file maintains all skill testing logic

## Fake Data vs Mocks

**Use Real Objects with Fake Data** (preferred):
- ✅ Python objects: `MessageDeduplicator()`, `ToolCallTracker()`, `AgentConfig()`
- ✅ LangChain messages: `AIMessage(content="test", id="123")`
- ✅ Temp directories: `tmp_path / "fake_config.yaml"`
- ✅ In-memory databases: `sqlite3.connect(":memory:")`
- ✅ FastAPI app: `TestClient(app)` with real app instance
- **Why**: Tests actual behavior, no mocking brittleness, simpler code

**Use Mocks Only for External Services**:
- ❌ Google/Anthropic API credentials and clients
- ❌ MCP server connections (external HTTP)
- ❌ Langfuse observability service
- ❌ External file I/O that can't use temp dirs
- **Why**: Expensive, slow, require network, or have side effects

**Example - Bad (over-mocking)**:
```python
def test_manager_initialization():
    mock_dedup = MagicMock(spec=MessageDeduplicator)  # ❌ Unnecessary mock
    mock_tracker = MagicMock(spec=ToolCallTracker)    # ❌ Unnecessary mock
    manager = AgentManager(dedup=mock_dedup, tracker=mock_tracker)
```

**Example - Good (real objects)**:
```python
def test_manager_initialization():
    manager = AgentManager()  # ✅ Real objects inside
    assert isinstance(manager.deduplicator, MessageDeduplicator)
    assert isinstance(manager.tracker, ToolCallTracker)
```

## Migration Plan

### Current Test Files → New Location

**Unit Tests** (✅ migration complete - all 167 tests passing):

| Original File | New Location | Testing Approach | Status |
|-------------|--------------|------------------|--------|
| test_llm.py | unit/agent/test_llm.py | Mock: Google/Anthropic credentials | ✅ Done (8 tests) |
| test_manager.py | unit/agent/test_manager.py | Real: MessageDeduplicator/ToolCallTracker | ✅ Done (7 tests) |
| test_agent_config_skills.py | unit/agent/config/test_config.py | Real: AgentConfig + temp dirs | ✅ Done (3 tests) |
| test_checkpointer.py | unit/infrastructure/test_checkpointer.py | Mock: PostgreSQL checkpointer | ✅ Done (8 tests) |
| test_mcp.py | unit/infrastructure/test_mcp.py | Mock: External MCP servers | ✅ Rewritten (18 tests) |
| test_subagents.py | unit/infrastructure/test_subagents.py | Mock: agent_config, SubAgent | ✅ Rewritten (8 tests) |
| test_messages.py | unit/adapters/test_langchain.py | Real: LangChain messages | ✅ Done (24 tests) |
| test_streaming.py | unit/streaming/test_streaming.py | Real: Streaming objects | ✅ Done (38 tests) |
| test_exceptions.py | unit/test_exceptions.py | Real: Exception classes | ✅ Done (14 tests) |
| test_google_creds.py | unit/utils/test_google_creds.py | Mock: Google credentials | ✅ Done (8 tests) |
| test_feedback.py | unit/api/routes/test_feedback.py | Mock: Langfuse client | ✅ Done (6 tests) |
| test_threads.py | unit/api/routes/test_threads.py | Mock: Checkpointer | ✅ Done (4 tests) |
| test_history.py | unit/api/routes/test_history.py | Mock: Checkpointer | ✅ Done (21 tests) |

**Total**: 167 unit tests migrated and passing

**Skills Tests** (NEW - create auto-discovery pattern):
- Create `tests/skills/test_skills.py` - Generic test that discovers all skills
- Create `tests/skills/conftest.py` - pytest_generate_tests + minimal agent fixture
- Move `tests/agents/llm_judge.py` → `tests/skills/llm_judge.py` (for assertion evaluation)
- Skills need NO tools (client-intake, bmi-report, email-formatter are self-contained)

**Delete Old Agent Tests**:
- Remove entire `tests/agents/` directory after migration:
  - `test_orchestrator.py`, `test_analyst.py`, `test_publisher.py` (replaced by skills auto-discovery)
  - `mock_tools.py` (not needed for skills)
  - `conftest.py` (not needed for skills)
  - `subagent_loader.py` (not needed for skills)

**New Files to Create**:
- `tests/unit/api/test_app.py` - Test app factory, middleware, lifecycle (currently not tested)

## Import Updates Required

All imports in test files need to be updated from old `core/` and `routes/` paths:

```python
# Old imports
from template_agent.src.core.llm import create_model
from template_agent.src.core.manager import AgentManager
from template_agent.src.core.agent_config import AgentConfig
from template_agent.src.core.streaming import MessageDeduplicator
from template_agent.src.core.backend import get_backend
from template_agent.src.core.checkpointer import initialize_checkpointer
from template_agent.src.core.messages import convert_message
from template_agent.src.routes.feedback import feedback

# New imports
from template_agent.src.agent.llm import create_model
from template_agent.src.agent.manager import AgentManager
from template_agent.src.agent.config import AgentConfig
from template_agent.src.streaming import MessageDeduplicator
from template_agent.src.infrastructure.backend import get_backend
from template_agent.src.infrastructure.checkpointer import initialize_checkpointer
from template_agent.src.adapters.langchain import convert_message
from template_agent.src.api.routes.agent.feedback import feedback
```

## Implementation Steps

### Phase 1: Unit Tests ✅ COMPLETE

**Status**: All 167 unit tests passing after migration and rewrites.

1. ✅ **Created directory structure**
   ```bash
   mkdir -p tests/unit/{agent/config,infrastructure,adapters,streaming,api/routes,utils}
   ```

2. ✅ **Migrated and enhanced files**:
   - Updated all imports from `core/` → new semantic paths
   - Used fake data over mocks where appropriate
   - Ensured each test has a clear, single purpose
   - All tests verified and passing

3. ✅ **Migration completed** (in order):
   - test_exceptions.py → unit/test_exceptions.py (14 tests) ✅
   - test_streaming.py → unit/streaming/test_streaming.py (38 tests) ✅
   - test_messages.py → unit/adapters/test_langchain.py (24 tests) ✅
   - test_agent_config_skills.py → unit/agent/config/test_config.py (3 tests) ✅
   - test_manager.py → unit/agent/test_manager.py (7 tests, real objects) ✅
   - test_checkpointer.py → unit/infrastructure/test_checkpointer.py (8 tests) ✅
   - test_llm.py → unit/agent/test_llm.py (8 tests, mocks for APIs) ✅
   - test_google_creds.py → unit/utils/test_google_creds.py (8 tests) ✅
   - test_feedback.py → unit/api/routes/test_feedback.py (6 tests) ✅
   - test_threads.py → unit/api/routes/test_threads.py (4 tests) ✅
   - test_history.py → unit/api/routes/test_history.py (21 tests) ✅

4. ✅ **Rewrote skipped tests** to match refactored implementations:
   - test_subagents.py → unit/infrastructure/test_subagents.py (8 tests, completely rewritten) ✅
   - test_mcp.py → unit/infrastructure/test_mcp.py (18 tests, completely rewritten) ✅

5. ✅ **Deleted old unit test files** after migration complete

### Phase 2: Skills Tests (New Pattern - No Tools Needed!)

1. **Create directory structure**
   ```bash
   mkdir -p tests/skills
   ```

2. **Create generic skill test**:
   - `tests/skills/test_skills.py` - Auto-discovery pattern (discovers all skills from agent_config/skills/)
   - `tests/skills/conftest.py` - pytest_generate_tests + minimal agent fixture
   - `tests/skills/llm_judge.py` - Move from tests/agents/ (for assertion evaluation)
   - **NO mock_tools.py needed** - Skills are self-contained!

3. **Skills tested** (auto-discovered):
   - client-intake (uses local scripts/convert_units.py)
   - bmi-report (reads markdown references)
   - email-formatter (reads HTML templates)
   - Any future skills you add!

### Phase 3: Cleanup & Quality Check

1. **Remove old agent tests directory**:
   ```bash
   rm -rf tests/agents/
   ```

2. **Verify all tests pass**:
   ```bash
   pytest tests/unit/ -v
   pytest tests/skills/ -v
   ```

3. **Quality review checklist** (for each test file):
   - [ ] Each test has a descriptive name (not just `test_function_name`)
   - [ ] Each test has a single, clear purpose
   - [ ] No bloated tests testing 5+ behaviors
   - [ ] Using fake data instead of mocks where possible
   - [ ] No unnecessary setup/mocking
   - [ ] Test docstrings explain what's being tested
   - [ ] Tests are easy to read and understand

4. **Update CI/CD** if needed to run both test directories

## Benefits

**Two-tier testing approach**:
1. **Unit tests**: Fast, isolated tests - prefer real objects with fake data over mocks
2. **Skills tests**: Minimal agent with auto-discovery - NO tools needed (self-contained utilities)

**Specific advantages**:
- **Fake data over mocks**: Use real Python objects (MessageDeduplicator, LangChain messages) with fake inputs
- **Less brittle tests**: Real objects test actual behavior, not mock expectations
- **Focused tests**: One clear goal per test - easy to understand, debug, and maintain
- **Self-documenting**: Test names describe behavior (e.g., `test_deduplicator_marks_message_as_seen`)
- **Auto-discovery**: Add new skill → tests automatically run, no test code changes needed
- **No tool mocking for skills**: Skills are pure utilities (client-intake, bmi-report, email-formatter)
- **Clear structure**: Mirror source organization for easy navigation
- **Fast feedback**: Unit tests run in milliseconds (real objects, temp dirs, in-memory DBs)
- **Maintainability**: Change `src/streaming/tracker.py` → check `tests/unit/streaming/test_streaming.py`
- **Scalability**: Adding 10 new skills doesn't require writing 10 new test files
- **Simplicity**: No integration tests needed - unit + skills coverage is sufficient
