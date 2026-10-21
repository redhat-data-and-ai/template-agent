"""Why an answer looks the way it does.

A reply is shaped by decisions the reader cannot see: which model
produced it, whether routing picked a cheaper tier, whether the
primary model failed and a fallback answered, whether the reply was
cut short, and whether a guardrail or PII rule touched it. Each is
invisible in the transcript, and each changes how much weight the
answer deserves.

Derived in one place rather than annotated by each middleware. The
alternative — every component appending its own key — is how three
of them end up using three names for "truncated", and how the fourth
is forgotten. Everything here is read off the message the model
returned, plus whatever the middleware recorded on it.

Nothing here is a secret: a model name, a tier name and a finish
reason. Endpoints, project ids and prompts stay out.
"""

from __future__ import annotations

from typing import Any

# LangChain normalises some of this and passes the rest through, so
# the same fact arrives under different names depending on provider.
_MODEL_KEYS = ("model_name", "model", "model_version")
_FINISH_KEYS = ("finish_reason", "finishReason", "stop_reason")

# Provider spellings for "the model ran out of room". Anything else
# is an ordinary stop and not worth reporting.
_TRUNCATED = {"max_tokens", "maxtokens", "length", "max_output_tokens"}


def _first(meta: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_truncated(finish_reason: str) -> bool:
    return finish_reason.lower().replace("-", "_") in _TRUNCATED


def describe_message(msg: Any) -> dict[str, Any]:
    """Provenance for one AI message, or `{}` when there is nothing to say.

    Empty rather than a dict of nulls: the UI renders this line only
    when it carries something, and an always-present block of
    "unknown" is noise on every single turn.
    """
    meta = getattr(msg, "response_metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    extra = getattr(msg, "additional_kwargs", None)
    if not isinstance(extra, dict):
        extra = {}

    out: dict[str, Any] = {}

    # Integrations this turn could not reach, stamped onto the
    # message by McpAuthNoticeMiddleware. This is what the inline
    # Connect button reads.
    #
    # Read off the message rather than a contextvar: the run executes
    # in a worker task and the reply is serialised somewhere else, so
    # a contextvar set during the run is gone by the time anyone
    # reads it. The message is the one thing that crosses that
    # boundary — which is why routed_tier and pii_scrubbed travel the
    # same way.
    auth_required = extra.get("auth_required")
    if isinstance(auth_required, list) and auth_required:
        out["auth_required"] = auth_required

    model = _first(meta, _MODEL_KEYS)
    if model:
        out["model"] = model

    finish_reason = _first(meta, _FINISH_KEYS)
    if finish_reason:
        # Reported even when ordinary: "stopped normally" is the
        # answer to "was this cut off", and absent is not.
        out["finish_reason"] = finish_reason
        if _is_truncated(finish_reason):
            out["truncated"] = True

    # Recorded by middleware onto the message it produced. Absent
    # means the feature is off or did not fire, which is why these
    # are read rather than defaulted.
    for key in ("routed_tier", "routed_by", "fallback_from", "cached", "pii_scrubbed"):
        value = extra.get(key)
        if value not in (None, "", False):
            out[key] = value

    # An empty reply with no error is the failure mode that reads as
    # the agent having nothing to say. Saying so costs one boolean.
    content = getattr(msg, "content", None)
    if not content and not getattr(msg, "tool_calls", None):
        out["empty"] = True

    return out
