"""PII middleware package.

Public surface:
  init_pii_middleware(config, hash_key) → PIIScrubber   # call once at startup
  get_scrubber() → PIIScrubber | None                   # returns the global instance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from deep_agent.src.pii.config import PIIConfig
    from deep_agent.src.pii.scrubber import PIIScrubber

_scrubber: Optional["PIIScrubber"] = None


def init_pii_middleware(config: "PIIConfig", hash_key: bytes = b"") -> "PIIScrubber":
    """Initialise the global PII scrubber. Call once at process startup."""
    global _scrubber  # noqa: PLW0603
    from deep_agent.src.pii.scrubber import PIIScrubber

    _scrubber = PIIScrubber(config, hash_key)
    return _scrubber


def get_scrubber() -> Optional["PIIScrubber"]:
    """Return the global PIIScrubber, or None if not yet initialised."""
    return _scrubber


__all__ = [
    "init_pii_middleware",
    "get_scrubber",
]
