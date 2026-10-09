"""Host-side LangChain tools that call MCP ``resources/*`` (not server tools).

The model cannot speak JSON-RPC. These tools are the host adapter: they open the
same request-scoped MCP session as the Apps HTTP proxy. List/templates return
the catalog JSON unchanged. Read extracts ``contents[].text``, applies
line-based ``offset``/``limit`` pagination (defaults match ``read_file``:
offset 0, limit 100), then char-truncates to the eviction token budget.
Binary blobs are omitted. No catalog cache — safe for multi-pod.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field as PydanticField

from deep_agent.aegra.mcp import (
    _current_access_token,
    _filter_by_names,
    _get_server_configs,
    _resolve_mcp_user_id,
)
from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_host import (
    list_resource_templates,
    list_resources,
    read_resource,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

LIST_TOOL = "mcp_list_resources"
TEMPLATES_TOOL = "mcp_list_resource_templates"
READ_TOOL = "mcp_read_resource"

DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100
_CHARS_PER_TOKEN = 4
_DEFAULT_TOKEN_LIMIT = 100_000


def _get_max_chars() -> int:
    """Return the char budget for resource reads.

    Independent of FilesystemMiddleware's eviction threshold — resource
    reads have their own budget sized to fit large guidance documents.
    """
    return _CHARS_PER_TOKEN * _DEFAULT_TOKEN_LIMIT


def _extract_text(payload: dict[str, Any]) -> str:
    """Join ``contents[].text`` from an MCP ``resources/read`` result.

    Blob-only items are replaced with a stub so the model knows binary
    content was present but omitted.
    """
    contents = payload.get("contents")
    if not isinstance(contents, list):
        return ""
    parts: list[str] = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
        elif "blob" in item:
            mime = item.get("mimeType") or item.get("mime_type") or "unknown"
            parts.append(f"[Binary content omitted ({mime})]")
    return "\n".join(parts)


def _split_long_lines(lines: list[str], max_chars: int) -> list[str]:
    r"""Break any line longer than *max_chars* into chunks so every line is pageable.

    Each chunk ends with ``\\n`` so the downstream line-boundary logic works.
    """
    chunk_size = max(max_chars - 1, 1)
    out: list[str] = []
    for line in lines:
        while len(line) > max_chars:
            out.append(line[:chunk_size] + "\n")
            line = line[chunk_size:]
        out.append(line)
    return out


def _paginate_text(
    text: str, uri: str, *, offset: int, limit: int, max_chars: int
) -> str:
    """Slice *text* by line offset/limit, then char-truncate to *max_chars*.

    Long lines are pre-split into chunks so every line fits within the
    char budget. ``next_offset`` is derived from lines actually shown,
    not from the requested ``limit``, so no lines are skipped on resume.
    """
    lines = _split_long_lines(text.splitlines(keepends=True), max_chars)
    total_lines = len(lines)
    offset = max(offset, 0)
    limit = max(limit, 1)
    page = lines[offset : offset + limit]
    result = "".join(page)

    if not result and offset > 0:
        return (
            f"[No content at offset={offset}. "
            f"Resource {uri} has {total_lines} lines (offsets 0–{max(total_lines - 1, 0)}).]"
        )

    truncated_by_lines = offset + limit < total_lines
    truncated_by_chars = False

    if len(result) > max_chars:
        cut = result[:max_chars]
        last_nl = cut.rfind("\n")
        if last_nl >= 0:
            result = cut[: last_nl + 1]
            lines_shown = result.count("\n")
        else:
            result = cut
            lines_shown = 0
        truncated_by_chars = True

    if truncated_by_chars and lines_shown > 0:
        next_offset = offset + lines_shown
    elif truncated_by_lines:
        next_offset = offset + limit
    else:
        next_offset = None

    if next_offset is not None:
        notice = (
            f"\n\n[Output truncated. Resource {uri} has {total_lines} lines. "
            f"Use offset={next_offset} and limit={limit} to read the next page.]"
        )
        result += notice
    elif truncated_by_chars:
        result += (
            "\n\n[Output truncated due to size limits. "
            "The resource content is very large. "
            "Consider requesting a smaller portion or a different URI.]"
        )

    return result


def get_mcp_resource_tools(
    *,
    server_names: list[str] | None = None,
    allowed_uris: list[str] | None = None,
) -> list[Any]:
    """Return host resource tools scoped like ``get_mcp_tools``.

    Filters enabled MCP servers (and optional ``mcps:`` names) with the same
    helpers as tool discovery. Does **not** connect or call ``resources/list``
    — the three tools talk to the server at turn time.
    """
    servers = _get_server_configs()
    enabled = {
        k: v
        for k, v in servers.items()
        if isinstance(v, dict) and v.get("enabled", False)
    }
    enabled = _filter_by_names(enabled, server_names)
    return build_mcp_resource_tools(
        allowed_servers=list(enabled.keys()),
        allowed_uris=allowed_uris,
    )


def _uri_allowed(uri: str, allowed_uris: list[str] | None) -> bool:
    """Return True if *uri* is unrestricted, listed, or matches a listed template."""
    if not allowed_uris:
        return True
    if uri in allowed_uris:
        return True
    return any(
        "{" in pattern and _template_matches(pattern, uri) for pattern in allowed_uris
    )


def _template_matches(pattern: str, uri: str) -> bool:
    """Match RFC-6570-style ``{param}`` as a single URI path segment."""
    parts = re.split(r"(\{[^}]+\})", pattern)
    regex = (
        "^"
        + "".join(
            r"[^/]+" if p.startswith("{") and p.endswith("}") else re.escape(p)
            for p in parts
        )
        + "$"
    )
    return re.match(regex, uri) is not None


def _auth_context() -> tuple[str, str | None]:
    return _resolve_mcp_user_id() or "", _current_access_token.get()


def _is_authorization_required(exc: HTTPException) -> bool:
    detail = exc.detail
    return (
        exc.status_code == 401
        and isinstance(detail, dict)
        and detail.get("error") == "authorization_required"
        and isinstance(detail.get("mcp_name"), str)
        and isinstance(detail.get("connect_url"), str)
    )


def _raise_or_format_http(exc: HTTPException) -> str:
    if _is_authorization_required(exc):
        detail = exc.detail
        raise NeedsAuthorization(detail["mcp_name"], detail["connect_url"]) from None
    return f"MCP resource request failed ({exc.status_code}): {exc.detail}"


def _format_generic_failure(exc: BaseException) -> str:
    """Return a stable tool error; do not forward raw exception text to the model."""
    if isinstance(exc, TimeoutError):
        return "MCP resource request failed: timed out"
    logger.exception("MCP resource request failed")
    return "MCP resource request failed"


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _filter_resources(
    payload: dict[str, Any], allowed_uris: list[str] | None
) -> dict[str, Any]:
    if not allowed_uris:
        return payload
    out = dict(payload)
    out["resources"] = [
        item
        for item in (payload.get("resources") or [])
        if isinstance(item, dict)
        and _uri_allowed(str(item.get("uri") or ""), allowed_uris)
    ]
    return out


def _filter_templates(
    payload: dict[str, Any], allowed_uris: list[str] | None
) -> dict[str, Any]:
    if not allowed_uris:
        return payload
    key = (
        "resourceTemplates" if "resourceTemplates" in payload else "resource_templates"
    )
    out = dict(payload)
    out[key] = [
        item
        for item in (payload.get(key) or [])
        if isinstance(item, dict)
        and str(item.get("uriTemplate") or item.get("uri_template") or "")
        in allowed_uris
    ]
    return out


def _reject_server(mcp_name: str, allowed_servers: tuple[str, ...]) -> str:
    allowed = ", ".join(allowed_servers) if allowed_servers else "(none)"
    return f"Unknown or disallowed MCP server {mcp_name!r}. Allowed: {allowed}"


def build_mcp_resource_tools(
    *,
    allowed_servers: list[str],
    allowed_uris: list[str] | None = None,
) -> list[Any]:
    """Return list/templates/read tools, or ``[]`` when no servers are available.

    Args:
        allowed_servers: MCP ``mcp.json`` keys this agent may call.
        allowed_uris: ``None`` = all URIs; non-empty list = allowlist.
            Production callers normalize ``[]`` to ``None`` (falsy = unrestricted,
            matching ``mcps:`` and ``tools:`` behavior).
    """
    servers = tuple(s for s in allowed_servers if s)
    if not servers:
        return []

    max_chars = _get_max_chars()
    server_list = ", ".join(servers)

    class _ListInput(BaseModel):
        mcp_name: str = PydanticField(
            description=f"MCP server name. Allowed: {server_list}"
        )
        cursor: str | None = PydanticField(
            default=None,
            description="Pagination cursor from a previous list call",
        )

    class _ReadInput(BaseModel):
        mcp_name: str = PydanticField(
            description=f"MCP server name. Allowed: {server_list}"
        )
        uri: str = PydanticField(
            description="Resource URI from resources/list or a template"
        )
        offset: int = PydanticField(
            default=DEFAULT_READ_OFFSET,
            description="Line offset (0-indexed). Use for pagination of large resources.",
        )
        limit: int = PydanticField(
            default=DEFAULT_READ_LIMIT,
            description="Max lines to return. Default 100. Pass a higher value for large resources.",
        )

    async def _list(mcp_name: str, cursor: str | None = None) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        user_id, sso = _auth_context()
        try:
            payload = await list_resources(
                mcp_name, cursor=cursor, user_id=user_id, sso_token=sso
            )
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return _format_generic_failure(exc)
        return _dump(_filter_resources(payload, allowed_uris))

    async def _templates(mcp_name: str, cursor: str | None = None) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        user_id, sso = _auth_context()
        try:
            payload = await list_resource_templates(
                mcp_name, cursor=cursor, user_id=user_id, sso_token=sso
            )
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return _format_generic_failure(exc)
        return _dump(_filter_templates(payload, allowed_uris))

    async def _read(
        mcp_name: str,
        uri: str,
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        if not _uri_allowed(uri, allowed_uris):
            return f"Resource URI not allowed: {uri}"
        user_id, sso = _auth_context()
        try:
            payload = await read_resource(mcp_name, uri, user_id=user_id, sso_token=sso)
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return _format_generic_failure(exc)
        text = _extract_text(payload)
        return _paginate_text(
            text, uri, offset=offset, limit=limit, max_chars=max_chars
        )

    list_desc = (
        "List MCP resources (resources/list). Returns catalog metadata for concrete "
        "URIs, not file contents. Also call mcp_list_resource_templates when listing "
        "what resources are available. "
        f"mcp_name must be one of: {server_list}."
    )
    templates_desc = (
        "List MCP resource templates (resources/templates/list). These are URI "
        "patterns with {param} placeholders, not readable files. Fill a pattern "
        "then call mcp_read_resource. "
        f"mcp_name must be one of: {server_list}."
    )
    read_desc = (
        "Read an MCP resource (resources/read) by URI. Use after listing. "
        "By default reads up to 100 lines starting from the beginning. "
        "Use offset/limit to page through large resources. "
        "Binary blob contents are omitted. "
        f"mcp_name must be one of: {server_list}."
    )

    return [
        StructuredTool(
            name=LIST_TOOL,
            description=list_desc,
            coroutine=_list,
            args_schema=_ListInput,
        ),
        StructuredTool(
            name=TEMPLATES_TOOL,
            description=templates_desc,
            coroutine=_templates,
            args_schema=_ListInput,
        ),
        StructuredTool(
            name=READ_TOOL,
            description=read_desc,
            coroutine=_read,
            args_schema=_ReadInput,
        ),
    ]
