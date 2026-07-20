"""Webhook output sink — POSTs results to a URL."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import httpx

from deep_agent.src.triggers.sinks.protocol import OutputSink, TriggerResult
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "content"):
        return {"type": type(obj).__name__, "content": obj.content}
    return str(obj)


class WebhookSink(OutputSink):
    """POSTs TriggerResult to a configured URL with retry."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the webhook sink with URL and retry settings."""
        self._url = url
        self._headers = headers or {}
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def emit(self, result: TriggerResult) -> None:
        """POST the trigger result to the configured webhook URL."""
        client = self._ensure_client()
        data = asdict(result)
        payload = json.dumps(data, default=_default_serializer)

        backoff = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(
                    self._url,
                    content=payload,
                    headers={**self._headers, "Content-Type": "application/json"},
                )
                if resp.status_code < 500:
                    if resp.status_code >= 400:
                        logger.warning(
                            "Webhook sink got %d from %s", resp.status_code, self._url
                        )
                    return
                logger.warning(
                    "Webhook sink got %d (attempt %d/%d)",
                    resp.status_code,
                    attempt + 1,
                    self._max_retries + 1,
                )
            except httpx.HTTPError:
                logger.exception(
                    "Webhook sink error (attempt %d/%d)",
                    attempt + 1,
                    self._max_retries + 1,
                )

            if attempt < self._max_retries:
                import asyncio

                await asyncio.sleep(backoff)
                backoff *= 2

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
