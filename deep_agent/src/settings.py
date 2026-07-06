"""Settings configuration for the template agent.

All operational defaults live HERE. No env vars needed for basic operation.
Override via environment variables only when deploying to a different context.

Hierarchy (highest wins):
  1. Environment variables (set by orchestrator, compose, or shell)
  2. .env file (secrets only — keys, passwords, credentials)
  3. Defaults below (tuned for containerized demo stack)
"""

from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

from deep_agent.src.exceptions import AppException, ErrorCodes
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_DEV_PUBLIC_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

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

    # ── Model ─────────────────────────────────────────────────────────
    MAX_OUTPUT_TOKENS: int = Field(default=8192)

    # ── Database (PostgreSQL) ─────────────────────────────────────────
    POSTGRES_HOST: str = Field(default="pgvector")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="template_agent")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")

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
    ENABLE_USER_ID_ENCRYPTION: bool = Field(default=False)

    # ── Observability (Langfuse) ──────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None)
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None)
    LANGFUSE_BASE_URL: Optional[str] = Field(default=None)
    LANGFUSE_TRACING_ENVIRONMENT: str = Field(default="development")

    # ── OpenTelemetry ─────────────────────────────────────────────────
    ENABLE_OTEL_METRICS: bool = Field(default=False)
    ENABLE_OTEL_TRACES: bool = Field(default=False)
    OTEL_SERVICE_NAME: str = Field(default="template-agent")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="",
        description="OTLP gRPC metrics endpoint (OpenShift: otel-gateway:4327)",
    )
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: str = Field(
        default="",
        description="OTLP gRPC traces endpoint (local/dev-loop: Jaeger :4317)",
    )
    OTEL_AUTH_TOKEN: str = Field(default="", repr=False)
    OTEL_METRIC_EXPORT_INTERVAL_MILLIS: int = Field(default=10000)

    def resolved_otel_traces_endpoint(self) -> str:
        """Return the configured OTLP traces exporter endpoint."""
        return self.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

    def otel_traces_active(self) -> bool:
        """Return True when trace export is enabled and an endpoint is configured."""
        return bool(self.ENABLE_OTEL_TRACES and self.resolved_otel_traces_endpoint())

    # ── Google Cloud ──────────────────────────────────────────────────
    GOOGLE_APPLICATION_CREDENTIALS_CONTENT: Optional[str] = Field(default=None)

    # ── vLLM / OpenAI-compatible ─────────────────────────────────────
    VLLM_BASE_URL: Optional[str] = Field(default=None)
    VLLM_API_KEY: str = Field(default="EMPTY")

    # ── Cache ─────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = Field(default=True)

    # ── Memory Processing ─────────────────────────────────────────────
    MEMORY_CONSOLIDATION_ENABLED: bool = Field(default=True)
    MEMORY_DECAY_ENABLED: bool = Field(default=True)
    MEMORY_CLUSTERING_ENABLED: bool = Field(default=True)
    MEMORY_RELATIONSHIPS_ENABLED: bool = Field(default=True)

    # ── Middleware ────────────────────────────────────────────────────
    MIDDLEWARE_ENABLED: bool = Field(default=True)

    # ── CLI ───────────────────────────────────────────────────────────
    ENABLE_CLI: bool = Field(default=True)

    # ── FLAG TO SWITCH TO RELOAD FROM DISK ────────────────────────────
    CONFIG_AUTO_RELOAD: bool = Field(default=False)

    # ── MCP OAuth ─────────────────────────────────────────────────────
    MCP_TOKEN_ENCRYPTION_KEY: Optional[str] = Field(default=None)
    MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS: Optional[str] = Field(default=None)
    AGENT_PUBLIC_BASE_URL: Optional[str] = Field(default=None)

    # ── Derived ───────────────────────────────────────────────────────

    @property
    def agent_public_base_url(self) -> str:
        """Public base URL for MCP OAuth connect/callback endpoints."""
        if self.AGENT_PUBLIC_BASE_URL:
            return self.AGENT_PUBLIC_BASE_URL.rstrip("/")
        return f"http://localhost:{self.AGENT_PORT}"

    @property
    def is_dev_public_url(self) -> bool:
        """True when the public base URL is an allowed local HTTP dev endpoint."""
        parsed = urlparse(self.agent_public_base_url)
        return parsed.scheme == "http" and parsed.hostname in _DEV_PUBLIC_HOSTS

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
    """Validate port range and log level."""
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
                "(http:// is permitted only for localhost, 127.0.0.1, or ::1)",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )


settings = Settings()
