#!/usr/bin/env python
"""Generate the config reference from the Pydantic models.

35 config files with no index means answering "what can I change?"
requires reading Python. Hand-written docs would answer it once and
then drift; generating from the models means the reference cannot
disagree with the code that enforces it.

    make config-reference
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "docs" / "config-reference.md"


def _type_name(annotation: Any) -> str:
    """Render a type annotation the way a config author thinks about it."""
    if annotation is None or annotation is type(None):
        return "null"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)

    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        inner = [a for a in args if a is not type(None)]
        rendered = " | ".join(_type_name(a) for a in inner)
        return f"{rendered} | null" if len(inner) < len(args) else rendered

    if origin in (list, set, frozenset, tuple):
        return f"list[{', '.join(_type_name(a) for a in args)}]" if args else "list"

    if origin is dict:
        return f"map[{_type_name(args[1])}]" if len(args) == 2 else "map"

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return annotation.__name__
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def _default(field: Any) -> str:
    from pydantic_core import PydanticUndefined

    if field.default_factory is not None:
        try:
            produced = field.default_factory()
        except Exception:
            return "_(computed)_"
        if isinstance(produced, BaseModel):
            return "_(section)_"
        return f"`{produced!r}`" if produced not in ([], {}) else "_(empty)_"

    if field.default is PydanticUndefined:
        return "**required**"
    return f"`{field.default!r}`"


def _describe(model: type[BaseModel], prefix: str, seen: set[str]) -> list[str]:
    """One table row per field, recursing into nested models."""
    rows: list[str] = []

    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation

        nested = None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested = annotation

        description = (field.description or "").replace("\n", " ").strip()
        # A union renders as `a | b`, and a bare pipe ends the table cell.
        rendered_type = _type_name(annotation).replace("|", "\\|")
        rows.append(
            f"| `{path}` | {rendered_type} | {_default(field)} | {description} |"
        )

        if nested is not None and nested.__name__ not in seen:
            rows.extend(_describe(nested, path, seen | {nested.__name__}))

    return rows


def _section(title: str, where: str, model: type[BaseModel], doc: str) -> str:
    header = [
        f"### {title}",
        "",
        f"**File:** {where}",
        "",
    ]
    if doc:
        header += [doc.strip().split("\n\n")[0], ""]
    header += [
        "| Key | Type | Default | Notes |",
        "|---|---|---|---|",
    ]
    return "\n".join(header + _describe(model, "", {model.__name__})) + "\n"


def build() -> str:
    """Render the config reference table from the settings models."""
    from deep_agent.src.agent.config.cache import CacheFileConfig
    from deep_agent.src.agent.config.filesystem import FilesystemFileConfig
    from deep_agent.src.agent.config.middleware import (
        ExecutionConfig,
        MiddlewareDefaults,
    )
    from deep_agent.src.agent.config.opa import OpaFileConfig
    from deep_agent.src.agent.config.otel import OtelFileConfig
    from deep_agent.src.agent.config.providers import ProvidersFileConfig
    from deep_agent.src.routing.config import RouterConfig

    sections: list[tuple[str, str, type[BaseModel]]] = [
        ("Middleware", "`runtime/agent.yaml` → `middleware:`", MiddlewareDefaults),
        ("Providers", "`runtime/agent.yaml` → `providers:`", ProvidersFileConfig),
        ("Cache", "`runtime/agent.yaml` → `cache:`", CacheFileConfig),
        ("Filesystem", "`runtime/agent.yaml` → `filesystem:`", FilesystemFileConfig),
        ("Policy engine (OPA)", "`runtime/agent.yaml` → `opa:`", OpaFileConfig),
        ("Execution", "`runtime/agent.yaml` → `execution:`", ExecutionConfig),
        ("Observability", "`runtime/observability.yaml`", OtelFileConfig),
        ("Model routing", "`runtime/router.yaml`", RouterConfig),
    ]

    out = [
        "# Config reference",
        "",
        "**Generated — do not edit.** Regenerate with `make config-reference`.",
        "",
        "Every key the agent reads from `config/agent/`, with its type and",
        "default, derived from the Pydantic models that validate it. If a key",
        "is not here, the agent does not read it.",
        "",
        "Files not covered below are not schema-validated: `PROMPT.md` and",
        "`subagents/*.md` are Markdown with YAML frontmatter, `mcp.json` is",
        "JSONC, `skills/` are directories, and `compliance/policies/*.rego`",
        "are OPA policies. `make validate-config` checks all of them.",
        "",
        "Run `make validate-config` after editing anything here.",
        "",
    ]

    for title, where, model in sections:
        out.append(_section(title, where, model, model.__doc__ or ""))
        out.append("")

    # Exactly one trailing newline. The sections each append a blank
    # line, which left the file ending in two — and `end-of-file-fixer`
    # trims it back to one on every commit, so the generator and the
    # hook disagreed forever and the staleness test failed either way.
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> int:
    """Write the config reference to disk. Returns a process exit code."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build())
    keys = build().count("\n| `")
    print(f"Wrote {OUTPUT.relative_to(REPO)} — {keys} keys documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
