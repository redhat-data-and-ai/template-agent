"""Settings configuration for the template agent.

All operational defaults live HERE. No env vars needed for basic operation.
Override via environment variables only when deploying to a different context.

Hierarchy (highest wins):
  1. Environment variables (set by orchestrator, compose, or shell)
  2. .env file (secrets only — keys, passwords, credentials)
  3. Defaults below (tuned for containerized demo stack)
"""

from typing import Literal, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from deep_agent.src.exceptions import AppException, ErrorCodes
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_DEV_PUBLIC_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

SafetyThreshold = Literal[
    "BLOCK_NONE",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
]

try:
    load_dotenv()
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")


class Settings(BaseSettings):
    """All agent settings with production-ready defaults.

    Grouped by concern. Every field has a sensible default so the agent
    starts with zero configuration beyond secrets in .env.
    """

    # ── Server ────────────────────────────────────────────────────────
    AGENT_HOST: str = Field(default="0.0.0.0")
    AGENT_PORT: int = Field(default=5002)
    # Host the process reaches itself on, for in-cluster self-calls
    # (attachments, shares, forks, channel dispatch). Never rendered
    # to a user — see AGENT_PUBLIC_BASE_URL for the address that is.
    AGENT_LOOPBACK_HOST: str = Field(default="127.0.0.1")
    # Host used to build a public base URL when AGENT_PUBLIC_BASE_URL
    # is unset. Only correct on a developer machine; every deployed
    # environment sets AGENT_PUBLIC_BASE_URL explicitly.
    AGENT_FALLBACK_PUBLIC_HOST: str = Field(default="localhost")
    SSL_KEYFILE: Optional[str] = Field(default=None)
    SSL_CERTFILE: Optional[str] = Field(default=None)

    @property
    def get_ssl_keyfile_path(self) -> Optional[str]:
        """Return SSL key file path if configured, else None."""
        return None if not self.SSL_KEYFILE else self.SSL_KEYFILE

    @property
    def get_ssl_certfile_path(self) -> Optional[str]:
        """Return SSL cert file path if configured, else None."""
        return None if not self.SSL_CERTFILE else self.SSL_CERTFILE

    # ── Logging ───────────────────────────────────────────────────────
    PYTHON_LOG_LEVEL: str = Field(default="INFO")
    REQUEST_LOGGING_ENABLED: bool = Field(default=True)
    REQUEST_LOG_HEADERS: bool = Field(default=True)
    REQUEST_LOG_BODY: bool = Field(default=True)
    REQUEST_LOG_BODY_MAX_SIZE: int = Field(default=10240)

    # ── Security ──────────────────────────────────────────────────────
    REQUEST_BODY_MAX_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request body size in bytes (DoS protection)",
    )

    # ── Upload (POST /upload → agent data directory) ──────────────────
    UPLOAD_ENABLED: bool = Field(default=False)
    DATA_PATH: str = Field(
        default="/app/data",
        description="Writable directory holding the live data tree and staging dirs",
    )
    UPLOAD_MAX_ARCHIVE_BYTES: int = Field(default=10 * 1024 * 1024)  # 10MB
    UPLOAD_MAX_EXTRACTED_BYTES: int = Field(default=30 * 1024 * 1024)  # 30MB
    UPLOAD_MAX_FILE_BYTES: int = Field(default=5 * 1024 * 1024)  # 5MB
    UPLOAD_MAX_FILE_COUNT: int = Field(default=500)
    UPLOAD_LOCK_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=1,
        description=(
            "Age at which a held upload lock is treated as stale and broken, "
            "so a process killed mid-upload cannot wedge ingestion forever."
        ),
    )
    UPLOAD_ALLOWED_EXTENSIONS: str = Field(
        default=(
            ".md,.markdown,.rst,.txt,.csv,.json,.yaml,.yml,.toml,"
            ".ini,.cfg,.xml,.html,.py,.js,.ts,.go,.java,.sh,.sql,.lock"
        ),
        description=(
            "Comma-separated, case-insensitive. Applies to individually "
            "uploaded files only -- archives are taken whole. Non-matching "
            "files are skipped, not fatal."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────
    MAX_OUTPUT_TOKENS: int = Field(default=8192)
    CLAUDE_LOCATION: str = Field(default="us-east5")

    # ── Safety Settings (Gemini / Vertex AI) ─────────────────────────
    SAFETY_DANGEROUS_CONTENT: SafetyThreshold = Field(default="BLOCK_ONLY_HIGH")
    SAFETY_HATE_SPEECH: SafetyThreshold = Field(default="BLOCK_ONLY_HIGH")
    SAFETY_HARASSMENT: SafetyThreshold = Field(default="BLOCK_ONLY_HIGH")
    SAFETY_SEXUALLY_EXPLICIT: SafetyThreshold = Field(default="BLOCK_ONLY_HIGH")

    # ── Database (PostgreSQL) ─────────────────────────────────────────
    POSTGRES_HOST: str = Field(default="pgvector")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="template_agent")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")

    # ── Connection pool ───────────────────────────────────────────────
    # One pool serves everything in deep_agent that is not LangGraph.
    # Size it deliberately: the per-pod total multiplies by the HPA's
    # maxReplicas against the server's max_connections, and the ceiling
    # is shared with the aegra engine and LangGraph pools. See
    # docs/scalability.md for the arithmetic.
    DB_POOL_MIN_SIZE: int = Field(default=1, ge=0)
    DB_POOL_MAX_SIZE: int = Field(default=8, ge=1)
    # Seconds to wait for a free connection before failing the caller.
    # Shorter than the default 30: a request queued half a minute for a
    # connection has already lost, and holding it there occupies a
    # worker that could be shedding load instead.
    DB_POOL_TIMEOUT: float = Field(default=10.0, gt=0)
    # Caps a hung query. Long agent *runs* are expected and fine — they
    # are many short statements, not one long one — so this bounds a
    # genuinely stuck statement without touching run duration. Zero
    # disables it.
    DB_STATEMENT_TIMEOUT_MS: int = Field(default=30_000, ge=0)
    # Recycle connections well inside any proxy or firewall idle
    # timeout, so the pool retires them before the network does.
    DB_POOL_MAX_IDLE: float = Field(default=300.0, gt=0)
    DB_POOL_MAX_LIFETIME: float = Field(default=1800.0, gt=0)

    # ── Admission control ─────────────────────────────────────────────
    # How many runs one pod will start concurrently before shedding
    # with 503. Reads and health are never gated. Set against the
    # connection budget above, not against CPU: a run is mostly idle
    # but holds a checkpointer connection throughout. Zero disables
    # the gate.
    MAX_CONCURRENT_RUNS: int = Field(default=40, ge=0)

    # ── Prompt cache stability ────────────────────────────────────────
    # The current-time block in the system prompt is rounded down to
    # this many seconds. The system prompt is the head of the prefix a
    # provider caches and the tool schemas sit behind it, so anything
    # that varies here makes every call a cache miss on the whole
    # prefix — which is most of the input tokens. One hour keeps the
    # prefix identical across users and turns. Set to 1 to restore
    # per-second precision and give up prefix caching.
    DATETIME_PROMPT_GRANULARITY_SECONDS: int = Field(default=3600, ge=1)
    # Log the size of each model request, split into system prompt,
    # conversation and tool schemas. Cheap, and the only way to
    # apportion an input-token bill between the three — the first
    # attempt at that from the outside blamed tool schemas for a cost
    # that turned out to be unaffected by removing them.
    LOG_PROMPT_SIZE: bool = Field(default=True)

    # ── /runs/wait responsiveness ─────────────────────────────────────
    # Aegra polls a Redis done-key on a flat two-second interval, so a
    # run finishing in 300ms is reported to the caller at 2.0s. Poll
    # fast at first and back off instead: most runs finish inside the
    # first second, and a long one settles onto the old cadence.
    RUN_WAIT_FAST_POLL: bool = Field(default=True)
    RUN_WAIT_MIN_POLL_SECONDS: float = Field(default=0.05, gt=0)
    RUN_WAIT_MAX_POLL_SECONDS: float = Field(default=2.0, gt=0)
    # How often the Postgres fallback runs while waiting. Rate-limited
    # by time rather than by poll count: the done-key check is cheap
    # and the query is not, and at a 50ms poll a per-second-poll ratio
    # would put twenty queries a second behind every waiting request.
    RUN_WAIT_DB_CHECK_SECONDS: float = Field(default=4.0, gt=0)

    # ── MongoDB ───────────────────────────────────────────────────────
    MONGODB_URI: Optional[str] = Field(default=None, repr=False)
    MONGODB_DB: str = Field(default="tokenusage")

    # ── Redis ─────────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    REDIS_BROKER_ENABLED: bool = Field(default=True)

    # ── Auth / SSO ────────────────────────────────────────────────────
    ENABLE_AUTH: bool = Field(default=True)
    SSO_ISSUER_URL: Optional[str] = Field(default=None)
    SSO_CLIENT_ID: Optional[str] = Field(default=None)
    SSO_CLIENT_SECRET: Optional[str] = Field(default=None)
    SSO_DEV_USERNAME: str = Field(default="John Doe")
    SSO_DEV_USER_ID: str = Field(default="dev-user")
    # Identity handed to the agent when auth is disabled. It reaches
    # the model in the delegation preamble, so keep it a value an
    # operator can set rather than a literal in the auth module.
    SSO_DEV_EMAIL: str = Field(default="dev@example.invalid")

    # Where a user actually finds the MCP connection screen.
    #
    # Three places told the model this path and two of them were
    # wrong ("Settings → MCP"), so it confidently sent people to a
    # screen that does not exist. It is a fact about the interface,
    # it changes when the interface changes, and it belongs in one
    # place rather than in prose scattered across middleware.
    MCP_SETTINGS_PATH: str = Field(default="Settings → Components → MCP")
    # Cap on the provider-side account string shown to the model.
    MCP_ACCOUNT_MAX_CHARS: int = Field(default=160, ge=1)
    MCP_USERINFO_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)
    # Pause the run for authorization even when the tool was called
    # from a subagent, rather than reporting upward as a tool error.
    # Set false to fall back to the reported-message behaviour.
    MCP_AUTH_INTERRUPT_IN_SUBAGENTS: bool = Field(default=True)
    # How long one unanswered authorization prompt stands for.
    #
    # Pausing the turn to ask is right the first time. Repeating it
    # in every new chat, for someone who has already decided not to
    # connect that integration, is nagging — and it blocks each turn
    # to do it. Within this window the agent says the integration is
    # unavailable and shows the Connect button instead, without
    # pausing. Connecting clears it immediately.
    MCP_AUTH_PROMPT_COOLDOWN_SECONDS: int = Field(default=3600, ge=0)
    ENABLE_USER_ID_ENCRYPTION: bool = Field(default=False)

    # ── Environment ───────────────────────────────────────────────────
    ENVIRONMENT: str = Field(
        default="development",
        description="Runtime environment: development, production, staging. "
        "Production mode enforces auth, SSL verification, and PII scrubbing.",
    )

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    # ── Observability (Langfuse) ──────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None)
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None)
    LANGFUSE_BASE_URL: Optional[str] = Field(default=None)
    LANGFUSE_TRACING_ENVIRONMENT: str = Field(default="development")
    # Trace name, as a filter contract rather than an identity.
    #
    # Langfuse guidance is verb-first and action-oriented
    # (`generate-response`), low cardinality, and never the agent or
    # model name — those are already their own attributes, and a
    # name that changes silently breaks every saved filter and
    # evaluator built on it.
    LANGFUSE_TRACE_NAME: str = Field(default="handle-chat-turn")

    # ── OpenTelemetry ─────────────────────────────────────────────────
    ENABLE_OTEL_METRICS: bool = Field(default=False)
    ENABLE_OTEL_TRACES: bool = Field(default=False)
    OTEL_SERVICE_NAME: str = Field(default="template-agent")
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OTLP gRPC metrics endpoint (OpenShift: otel-gateway:4327)",
    )
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OTLP gRPC traces endpoint (local/dev-loop: Jaeger :4317)",
    )
    OTEL_AUTH_TOKEN: Optional[str] = Field(default=None, repr=False)
    OTEL_METRIC_EXPORT_INTERVAL_MILLIS: int = Field(default=10000)

    # ── Models ────────────────────────────────────────────────────────
    # No model name appears in code. Which model a component uses comes
    # from the agent config (PROMPT.md frontmatter, subagent
    # frontmatter); these are the values used when that config names
    # none, and they are the only place a name is written down.
    #
    # They were previously four literals scattered across
    # attachment_routes, graph, warming and subagents, naming three
    # different models — none of them the one actually deployed. Stale
    # defaults are harmless right up to the moment config fails to
    # load, at which point they silently pick a different, pricier
    # model and nothing says so.
    DEFAULT_MODEL: str = Field(default="gemini-3.8-flash")
    # Attachments are small, high-volume and latency-sensitive, so this
    # is deliberately separable from the orchestrator's model.
    ATTACHMENT_MODEL: str = Field(default="gemini-3.8-flash")
    # Used when the primary model fails, not as a general default.
    FALLBACK_MODEL: str = Field(default="gemini-3.8-flash")

    # ── Model routing ─────────────────────────────────────────────────
    # Prefixes that decide which client builds a model. Comma-separated
    # so a ConfigMap value stays readable; see the `*_prefixes`
    # properties below for the parsed form.
    VERTEX_MODEL_PREFIXES: str = Field(default="gemini-,claude-")
    CLAUDE_MODEL_PREFIXES: str = Field(default="claude-")
    OPENAI_MODEL_PREFIXES: str = Field(default="gpt-,o1-,o3-,o4-")
    # Models that reason before answering. Their thinking tokens come
    # out of max_output_tokens, so an uncapped budget can consume the
    # whole allowance and return an empty message.
    #
    # Note "gemini-3" matches every Gemini 3 model including Flash.
    # That is deliberate and worth stating, because the old code said
    # the opposite in a comment while doing this: Gemini 3 Flash does
    # reason, unlike 2.5 Flash, so it needs the cap.
    THINKING_MODEL_PREFIXES: str = Field(default="gemini-2.5-pro,gemini-3")
    THINKING_BUDGET_SHARE: float = Field(default=0.25, gt=0, lt=1)
    MIN_THINKING_BUDGET: int = Field(default=512, ge=0)

    # ── Provider endpoints and scopes ─────────────────────────────────
    GOOGLE_CLOUD_SCOPE: str = Field(
        default="https://www.googleapis.com/auth/cloud-platform"
    )
    SLACK_POST_MESSAGE_URL: str = Field(
        default="https://slack.com/api/chat.postMessage"
    )

    # ── Paths ─────────────────────────────────────────────────────────
    # The other container paths already read an env var with a literal
    # fallback; this one did not, so there was no way to move it.
    APP_CACHE_PATH: str = Field(default="/app/.cache")

    # ── Roles ─────────────────────────────────────────────────────────
    # Only the development identity's permissions. Real role names are
    # not here: they come from the LDAP config via `check_ldap_role`,
    # and "denied" is a sentinel rather than a configurable name — a
    # settings field for either would be dead config that reads as
    # though it does something.
    DEV_PERMISSIONS: str = Field(default="read,write,admin")

    # ── Operational limits ────────────────────────────────────────────
    # Knobs an operator would plausibly need to turn. Left in code:
    # genuine invariants that are not safe to change from outside.
    GRAPH_CACHE_MAX_SIZE: int = Field(default=16, ge=1)
    INFLIGHT_METRIC_INTERVAL_SECONDS: int = Field(default=10, ge=1)
    MCP_TOOLS_LIST_MAX_PAGES: int = Field(default=20, ge=1)

    # ── Tunable limits ────────────────────────────────────────────────
    # Every module-level numeric constant in deep_agent lives here.
    # Nothing in code carries a literal bound: a value an operator
    # might need to change should be changeable without a rebuild,
    # and a value nobody should change is still easier to find when
    # there is one place to look.

    # ── Evaluations ───────────────────────────────────────────────────────
    # TTLs and limits for the /evals surface.
    # 5-min token minus a 30s MCP buffer.
    EVAL_ACCESS_TTL_SECONDS: int = Field(default=270, ge=0)
    # Safety net; explicit cleanup via /evals/internal/cleanup.
    EVAL_KEY_TTL_SECONDS: int = Field(default=3600, ge=0)
    EVAL_EXPORT_ROW_LIMIT: int = Field(default=50_000, ge=0)
    EVAL_MAX_HITL_APPROVALS: int = Field(default=50, ge=0)
    EVAL_REFRESH_TTL_SECONDS: int = Field(default=3600, ge=0)

    # ── MCP ───────────────────────────────────────────────────────────────
    # Token lifetimes, health checks and resource limits for MCP servers.
    # 7 days.
    MCP_AUTH_SET_TTL_SECONDS: int = Field(default=604800, ge=0)
    # Calibrated against recorded input tokens, not a character count.
    MCP_APPROX_TOKENS_PER_TOOL: int = Field(default=362, ge=0)
    # Rough ratio for budgeting, not a billing figure.
    CHARS_PER_TOKEN_ESTIMATE: int = Field(default=4, ge=0)
    MCP_HEALTH_CACHE_TTL_SECONDS: float = Field(default=30.0, gt=0)
    MCP_HEALTH_CHECK_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    MCP_MAX_ICON_BYTES: int = Field(default=256 * 1024, ge=0)
    MCP_OAUTH_STATE_TTL_SECONDS: int = Field(default=300, ge=0)
    MCP_RESOLVED_TOKEN_TTL_SECONDS: float = Field(default=30.0, gt=0)
    MCP_RESOURCE_READ_LIMIT: int = Field(default=100, ge=0)
    MCP_RESOURCE_READ_OFFSET: int = Field(default=0, ge=0)
    MCP_RESOURCE_TOKEN_LIMIT: int = Field(default=100_000, ge=0)
    MCP_SSO_REFRESH_BUFFER_SECONDS: float = Field(default=60.0, gt=0)
    MCP_TOKEN_EXPIRY_BUFFER_SECONDS: float = Field(default=30.0, gt=0)
    # Margin used at tool discovery, where renewing early is cheap.
    # Wider than the call-time buffer on purpose: a turn that lasts
    # two minutes should not have its token expire halfway through.
    MCP_TOKEN_RENEW_BUFFER_SECONDS: float = Field(default=600.0, gt=0)
    MCP_TOKEN_REFRESH_LOCK_TTL_SECONDS: int = Field(default=30, ge=0)
    MCP_TOKEN_REFRESH_LOCK_WAIT_SECONDS: float = Field(default=10.0, gt=0)
    MCP_TOKEN_REFRESH_WAIT_ATTEMPTS: int = Field(default=50, ge=0)
    MCP_TOKEN_REFRESH_WAIT_POLL_SECONDS: float = Field(default=0.1, gt=0)

    # ── Caches ────────────────────────────────────────────────────────────
    # In-process cache lifetimes.
    # The pool and MCP tool caches are configured separately above.
    ASSISTANT_ID_CACHE_TTL_SECONDS: float = Field(default=60.0, gt=0)
    LDAP_MEMORY_CACHE_MAX_SIZE: int = Field(default=1024, ge=0)
    MCP_STATUS_CACHE_SECONDS: int = Field(default=60, ge=0)
    PERSONALIZATION_FALLBACK_EXPIRE_SECONDS: int = Field(default=5, ge=0)
    SELF_KNOWLEDGE_CACHE_SECONDS: int = Field(default=60, ge=0)

    # ── Attachments ───────────────────────────────────────────────────────
    ATTACHMENT_MAX_LISTED: int = Field(default=20, ge=0)
    ATTACHMENT_MAX_NAME_LENGTH: int = Field(default=128, ge=0)
    ATTACHMENT_RUN_POLL_SECONDS: float = Field(default=1.0, gt=0)
    ATTACHMENT_RUN_WAIT_SECONDS: int = Field(default=90, ge=0)
    ATTACHMENT_SUSPECT_CHARS: int = Field(default=200, ge=0)

    # ── Documents ─────────────────────────────────────────────────────────
    DOCUMENTS_CAPACITY_BYTES: int = Field(default=500 * 1024 * 1024, ge=0)
    DOCUMENTS_CAPACITY_WARN_FRACTION: float = Field(default=0.80, gt=0)
    DOCUMENTS_MAX_EDITABLE_BYTES: int = Field(default=1024 * 1024, ge=0)

    # ── Sharing and forking ───────────────────────────────────────────────
    FORK_REQUEST_TIMEOUT_SECONDS: int = Field(default=30, ge=0)
    SHARE_DEFAULT_EXPIRY_DAYS: int = Field(default=30, ge=0)
    SHARE_MAX_EXPIRY_DAYS: int = Field(default=365, ge=0)
    SHARE_MAX_MESSAGE_CHARS: int = Field(default=20_000, ge=0)
    SHARE_MAX_SNAPSHOT_MESSAGES: int = Field(default=500, ge=0)
    SHARE_REQUEST_TIMEOUT_SECONDS: int = Field(default=30, ge=0)
    SHARE_TOKEN_BYTES: int = Field(default=16, ge=0)

    # ── Skills ────────────────────────────────────────────────────────
    SKILL_MAX_FILE_BYTES: int = Field(default=256_000, ge=0)

    # ── Threads and tags ──────────────────────────────────────────────────
    MAX_TAGS_PER_THREAD: int = Field(default=20, ge=0)
    MAX_TAG_LENGTH: int = Field(default=32, ge=0)

    # ── Search indexing ───────────────────────────────────────────────────
    SEARCH_MAX_BODY_CHARS: int = Field(default=100_000, ge=0)
    SEARCH_MAX_THREADS_PER_REFRESH: int = Field(default=200, ge=0)

    # ── OpenTelemetry bounds ──────────────────────────────────────────────
    # Clamps on the export interval, and a cap on tracked HITL roots.
    OTEL_MAX_EXPORT_INTERVAL_MS: int = Field(default=60000, ge=0)
    OTEL_MIN_EXPORT_INTERVAL_MS: int = Field(default=1000, ge=0)
    TELEMETRY_MAX_OPEN_HITL_ROOTS: int = Field(default=1024, ge=0)

    # ── Retries and resilience ────────────────────────────────────────────
    CHANNEL_SEEN_LIMIT: int = Field(default=2048, ge=0)
    MAX_NODE_RETRIES: int = Field(default=2, ge=0)
    MIN_JWT_SECRET_BYTES: int = Field(default=32, ge=0)
    NODE_RETRY_DELAY_SECONDS: float = Field(default=1.0, gt=0)

    # ── Analytics and token accounting ────────────────────────────────────
    # Above this a provider figure is treated as corrupt.
    MAX_REASONABLE_TOKENS: int = Field(default=1_000_000, ge=0)
    ANALYTICS_MAX_WINDOW_DAYS: int = Field(default=365, ge=0)
    ANALYTICS_STALE_RUN_SECONDS: int = Field(default=3600, ge=0)
    TOKEN_BUDGET_FAILURE_ALERT_THRESHOLD: int = Field(default=5, ge=0)
    USAGE_FAILURE_ALERT_THRESHOLD: int = Field(default=5, ge=0)

    # ── Memory clustering ─────────────────────────────────────────────────
    MEMORY_CLUSTER_THRESHOLD: float = Field(default=0.4, gt=0)
    MEMORY_NEAR_DUPLICATE_JACCARD: float = Field(default=0.5, gt=0)

    # ── Knowledge ingestion ───────────────────────────────────────────────
    # Fetch, transcription, media upload and skill generation bounds.
    DATASET_CATEGORICAL_MAX_DISTINCT: int = Field(default=25, ge=0)
    DATASET_SAMPLE_ROWS: int = Field(default=50_000, ge=0)
    KNOWLEDGE_FETCH_MAX_BYTES: int = Field(default=5 * 1024 * 1024, ge=0)
    KNOWLEDGE_FETCH_MAX_REDIRECTS: int = Field(default=3, ge=0)
    KNOWLEDGE_FETCH_TIMEOUT_SECONDS: int = Field(default=20, ge=0)
    MEDIA_MAX_FILE_API_BYTES: int = Field(default=2 * 1024 * 1024 * 1024, ge=0)
    MEDIA_POLL_SECONDS: float = Field(default=2.0, gt=0)
    MEDIA_PROCESSING_TIMEOUT_SECONDS: int = Field(default=300, ge=0)
    SKILL_MAX_CHARS: int = Field(default=6000, ge=0)
    SKILL_MAX_REFERENCE_CHARS: int = Field(default=12000, ge=0)
    TRANSCRIBE_MAX_INLINE_BYTES: int = Field(default=14 * 1024 * 1024, ge=0)
    TRANSCRIBE_TIMEOUT_SECONDS: int = Field(default=300, ge=0)

    # ── Evaluations ───────────────────────────────────────────────────────
    # Where the eval runner reaches this agent.
    EVAL_AGENT_BASE_URL: str = Field(default="http://localhost:5002")
    EVAL_ASSISTANT_ID: str = Field(default="agent")

    # ── OpenTelemetry identity ────────────────────────────────────────────
    OTEL_DEFAULT_SERVICE_NAME: str = Field(default="template-agent")
    OTEL_DEFAULT_SERVICE_VERSION: str = Field(default="dev")

    # ── Namespaces and internal identifiers ───────────────────────────────
    MCP_SHARED_BREAKER_KEY: str = Field(default="mcp-servers")
    MCP_TOKEN_KEY_PREFIX: str = Field(default="mcp_oauth_token:")
    # How many connected users to pre-build the graph for at boot.
    # Each costs the MCP discovery it saves that user later, so this
    # trades startup time for first-request time.
    GRAPH_WARM_MAX_USERS: int = Field(default=5, ge=0)
    CIRCUIT_BREAKER_KEY_PREFIX: str = Field(default="aegra:circuit:")
    PERSONALIZATION_KEY_PREFIX: str = Field(default="personalization:")
    STOCK_MEMORY_NAMESPACE: str = Field(default="memories")
    MEDIA_UPLOAD_PREFIX: str = Field(default="agent-media")
    SUBAGENT_TOOL_NAME: str = Field(default="task")
    ORCHESTRATOR_AGENT_NAME: str = Field(default="orchestrator")

    # ── Paths ─────────────────────────────────────────────────────────────
    USER_MEMORY_FILE: str = Field(default="/memories/user_profile.md")
    USER_MEMORY_STORE_KEY: str = Field(default="user_profile.md")
    MEMORY_PATH_PREFIX: str = Field(default="/memories/")
    SKILLS_PATH_PREFIX: str = Field(default="/skills/")
    TEMPLATE_MARKERS_FILE: str = Field(default="runtime/template-markers.yaml")
    # PATH for sandboxed subprocesses.
    SYSTEM_PATH: str = Field(default="/usr/local/bin:/usr/bin:/bin")

    # Route prefixes. Changing one changes the public API surface, so
    # these are config rather than literals for consistency, not
    # because they are expected to move.
    EVAL_DATASET_ROUTE_PREFIX: str = Field(default="/dataset")
    DOCUMENTS_ROUTE_PREFIX: str = Field(default="/documents")
    TMP_CACHE_PATH: str = Field(default="/tmp/.cache")

    # ── Operator-facing policy messages ───────────────────────────────────
    OPA_BLOCKED_TOOL_MESSAGE: str = Field(default="Tool call blocked by OPA policy.")
    OPA_BLOCKED_TRAJECTORY_MESSAGE: str = Field(
        default="This thread has been terminated due to policy violation. Please start a new chat thread."
    )

    # ── MCP protocol identifiers ──────────────────────────────────────────
    # Set by the MCP spec. Here so a spec revision is a config change, not a rebuild — not because it is expected to differ per deployment.
    MCP_APPS_EXTENSION_ID: str = Field(default="io.modelcontextprotocol/ui")
    MCP_APPS_MIME_TYPE: str = Field(default="text/html;profile=mcp-app")
    MCP_LIST_RESOURCES_TOOL: str = Field(default="mcp_list_resources")
    MCP_LIST_TEMPLATES_TOOL: str = Field(default="mcp_list_resource_templates")
    MCP_READ_RESOURCE_TOOL: str = Field(default="mcp_read_resource")
    # How long a server's resource catalogue is reused. The catalogue
    # is what a server offers, not what it holds — it changes when the
    # server is redeployed, not between two questions. 0 disables.
    MCP_RESOURCE_CATALOG_TTL_SECONDS: int = Field(default=300, ge=0)

    # ── Internal state and metadata keys ──────────────────────────────────
    TOKEN_BUDGET_THREAD_ID_KEY: str = Field(default="token_budget_thread_id")
    TOKEN_BUDGET_USER_ID_KEY: str = Field(default="token_budget_user_id")
    TOKEN_BUDGET_TRACE_ID_KEY: str = Field(default="token_budget_trace_id")
    ROUTING_STATE_KEY: str = Field(default="_routed_tier")

    # ── User-facing refusal text ──────────────────────────────────────
    # Operators customise wording to match their own policy voice, so
    # these are the clearest case for config rather than code.
    INPUT_SAFETY_REFUSAL: str = Field(
        default="I can't help with that request due to content safety policy."
    )
    OPA_BLOCKED_MODEL_MESSAGE: str = Field(
        default="I'm unable to respond to that request. Please try rephrasing."
    )

    # ── Error messages ────────────────────────────────────────────────────
    ERR_INVALID_CREDENTIALS_JSON: str = Field(default="Invalid JSON in credentials")
    ERR_MISSING_PROJECT_ID: str = Field(
        default="Service account JSON does not contain 'project_id' field"
    )

    # ── PII Middleware ────────────────────────────────────────────────────
    PII_HASH_KEY: Optional[str] = Field(default=None, repr=False)
    PII_TOKEN_MAP_TTL_DAYS: int = Field(default=7, ge=1, le=365)

    @field_validator(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_AUTH_TOKEN",
        "PII_HASH_KEY",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @staticmethod
    def _csv(raw: str) -> tuple[str, ...]:
        """Split a comma-separated setting, dropping blanks."""
        return tuple(part.strip() for part in raw.split(",") if part.strip())

    @property
    def vertex_model_prefixes(self) -> tuple[str, ...]:
        """Model-name prefixes routed to Vertex AI."""
        return self._csv(self.VERTEX_MODEL_PREFIXES)

    @property
    def claude_model_prefixes(self) -> tuple[str, ...]:
        """Model-name prefixes served by Anthropic on Vertex."""
        return self._csv(self.CLAUDE_MODEL_PREFIXES)

    @property
    def openai_model_prefixes(self) -> tuple[str, ...]:
        """Model-name prefixes routed to an OpenAI-compatible endpoint."""
        return self._csv(self.OPENAI_MODEL_PREFIXES)

    @property
    def thinking_model_prefixes(self) -> tuple[str, ...]:
        """Model-name prefixes whose models reason before answering."""
        return self._csv(self.THINKING_MODEL_PREFIXES)

    @property
    def dev_permissions(self) -> list[str]:
        """Permissions granted to the development identity when auth is off."""
        return list(self._csv(self.DEV_PERMISSIONS))

    def resolved_otel_traces_endpoint(self) -> str | None:
        """Return the configured OTLP traces exporter endpoint."""
        return self.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

    def otel_traces_active(self) -> bool:
        """Return True when trace export is enabled and an endpoint is configured."""
        return bool(self.ENABLE_OTEL_TRACES and self.resolved_otel_traces_endpoint())

    # ── Google Cloud ──────────────────────────────────────────────────
    GOOGLE_APPLICATION_CREDENTIALS_CONTENT: Optional[str] = Field(default=None)
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None)

    # ── vLLM / OpenAI-compatible ─────────────────────────────────────
    VLLM_BASE_URL: Optional[str] = Field(default=None)
    VLLM_API_KEY: str = Field(default="EMPTY")

    # Replaces every model name in config — orchestrator, subagents, fallbacks
    # — with this one. Exists so a cluster with no provider credentials can
    # still exercise the agent end to end against a stub served over
    # VLLM_BASE_URL. Leave unset everywhere except test environments; the
    # whole point of naming models in config is lost if this is on.
    MODEL_OVERRIDE: Optional[str] = Field(default=None)

    # ── Granite Guardian guardrails ───────────────────────────────────
    # Guardrails are active when GUARDIAN_API_BASE is set.
    # Model and behavior config live in config/agent/runtime/guardrails.yaml.
    GUARDIAN_API_BASE: Optional[str] = Field(default=None)
    GUARDIAN_API_KEY: str = Field(default="EMPTY")
    GUARDIAN_SSL_VERIFY: bool = Field(default=True)

    # ── Cache ─────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = Field(default=True)

    # ── Personalization ──────────────────────────────────────────────
    MEMORY_ENABLED: bool = Field(default=True)
    # ── Middleware ────────────────────────────────────────────────────
    MIDDLEWARE_ENABLED: bool = Field(default=True)

    # ── CLI ───────────────────────────────────────────────────────────
    ENABLE_CLI: bool = Field(default=True)

    # ── Lifecycle Persistence ─────────────────────────────────────────
    LIFECYCLE_PERSISTENCE_ENABLED: bool = Field(default=True)
    LIFECYCLE_LEASE_SECONDS: int = Field(default=300)
    LIFECYCLE_MAX_RESUME_BATCH: int = Field(default=10)
    LIFECYCLE_RESUME_ON_STARTUP: bool = Field(default=True)
    # How often to re-check for interrupted runs after startup, and how
    # old one must be before the sweep touches it. Startup alone is not
    # enough: a pod shutting down marks its runs interrupted, and when
    # that lands after the replacement's startup scan nothing looks
    # again until the next restart — the thread stays "busy" in the
    # sidebar for ever. Zero disables the sweep.
    LIFECYCLE_SWEEP_INTERVAL_SECONDS: int = Field(default=60, ge=0)
    LIFECYCLE_SWEEP_MIN_AGE_SECONDS: int = Field(default=45, ge=0)

    # ── OPA (Authorization) ───────────────────────────────────────────
    # None = not set; agent.yaml opa: section is the source of truth.
    # Set env var to override YAML for that field only.
    OPA_ENABLED: Optional[bool] = Field(default=None)
    OPA_URL: Optional[str] = Field(default=None)
    OPA_TIMEOUT: Optional[float] = Field(default=None, gt=0)
    OPA_MAX_RETRIES: Optional[int] = Field(default=None, ge=0)

    # ── Platform ──────────────────────────────────────────────────────
    DEPLOYED_AGENT_NAME: str = Field(default="")
    DEPLOYED_AGENT_ORG: str = Field(default="")
    PLATFORM_AUDIT_ENABLED: bool = Field(default=True)
    PLATFORM_AUDIT_BUFFER_MAX: int = Field(default=1000, ge=1, le=100_000)

    # ── FLAG TO SWITCH TO RELOAD FROM DISK ────────────────────────────
    CONFIG_AUTO_RELOAD: bool = Field(default=True)

    # ── MCP OAuth ─────────────────────────────────────────────────────
    MCP_TOKEN_ENCRYPTION_KEY: Optional[str] = Field(default=None)
    MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS: Optional[str] = Field(default=None)
    MCP_DCR_ENABLED: bool = Field(default=True)
    ENABLE_DCR_REREGISTER_API: bool = Field(default=False)
    AGENT_PUBLIC_BASE_URL: Optional[str] = Field(default=None)
    UI_ORIGIN: Optional[str] = Field(
        default=None,
        description="Origin of the frontend UI (e.g. http://localhost:5173). "
        "Used as postMessage target in OAuth callback. "
        "Falls back to AGENT_PUBLIC_BASE_URL if not set.",
    )

    # ── Derived ───────────────────────────────────────────────────────

    @property
    def agent_deployment_id(self) -> str:
        """Unique identity for this agent deployment, used as DCR/token key.

        Combines org + agent name when deployed via agent-engine.
        Falls back to the generic config name for local dev.
        """
        if self.DEPLOYED_AGENT_ORG and self.DEPLOYED_AGENT_NAME:
            return f"{self.DEPLOYED_AGENT_ORG}/{self.DEPLOYED_AGENT_NAME}"
        if self.DEPLOYED_AGENT_NAME:
            return self.DEPLOYED_AGENT_NAME
        from deep_agent.src.agent.config import agent_config

        return agent_config.get_name()

    @property
    def loopback_base_url(self) -> str:
        """Base URL for this process calling its own HTTP API."""
        return f"http://{self.AGENT_LOOPBACK_HOST}:{self.AGENT_PORT}"

    @property
    def agent_public_base_url(self) -> str:
        """Public base URL for MCP OAuth connect/callback endpoints."""
        if self.AGENT_PUBLIC_BASE_URL:
            return self.AGENT_PUBLIC_BASE_URL.rstrip("/")
        return f"http://{self.AGENT_FALLBACK_PUBLIC_HOST}:{self.AGENT_PORT}"

    @property
    def is_dev_public_url(self) -> bool:
        """True when the public base URL is an allowed local HTTP dev endpoint."""
        parsed = urlparse(self.agent_public_base_url)
        hostname = parsed.hostname or ""
        return parsed.scheme == "http" and (
            hostname in _DEV_PUBLIC_HOSTS or hostname.endswith(".localhost")
        )

    @property
    def ui_origin(self) -> str | None:
        """Origin for postMessage in OAuth callback HTML.

        Returns the configured UI_ORIGIN, or falls back to
        AGENT_PUBLIC_BASE_URL. Returns None only when neither is set
        (callback HTML will use '*' as last resort).
        """
        if self.UI_ORIGIN:
            return self.UI_ORIGIN.rstrip("/")
        if self.AGENT_PUBLIC_BASE_URL:
            parsed = urlparse(self.AGENT_PUBLIC_BASE_URL)
            return f"{parsed.scheme}://{parsed.netloc}"
        return None

    @property
    def oauth_callback_url(self) -> str:
        """Canonical OAuth redirect URI derived from AGENT_PUBLIC_BASE_URL."""
        return f"{self.agent_public_base_url}/mcp/oauth/callback"

    @property
    def database_uri(self) -> str:
        """Build PostgreSQL connection URI from component settings."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


def validate_config(settings: Settings) -> None:
    """Validate port range, log level, and production constraints."""
    if not (1024 <= settings.AGENT_PORT <= 65535):
        raise AppException(
            f"AGENT_PORT must be between 1024 and 65535, got {settings.AGENT_PORT}",
            ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
        )

    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.PYTHON_LOG_LEVEL.upper() not in valid_log_levels:
        raise AppException(
            f"PYTHON_LOG_LEVEL must be one of {valid_log_levels}, got {settings.PYTHON_LOG_LEVEL}",
            ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
        )

    if settings.AGENT_PUBLIC_BASE_URL and not settings.is_dev_public_url:
        parsed = urlparse(settings.AGENT_PUBLIC_BASE_URL)
        if parsed.scheme != "https":
            raise AppException(
                "AGENT_PUBLIC_BASE_URL must use https:// in production "
                "(http:// is permitted only for localhost, *.localhost, 127.0.0.1, or ::1)",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )

    # Production-specific validations
    if settings.is_production:
        # Enforce auth in production
        if not settings.ENABLE_AUTH:
            raise AppException(
                "ENABLE_AUTH must be true in production. "
                "Configure SSO_ISSUER_URL, SSO_CLIENT_ID, and SSO_CLIENT_SECRET.",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )

        # Enforce HTTPS for public URL in production
        if settings.AGENT_PUBLIC_BASE_URL and settings.is_dev_public_url:
            raise AppException(
                "AGENT_PUBLIC_BASE_URL cannot use http://localhost in production. "
                "Configure a valid https:// URL.",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )


settings = Settings()
