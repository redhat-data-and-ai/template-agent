"""Granite Guardian guardrails — content safety error hierarchy and public API."""


class ContentSafetyError(ValueError):
    """Raised by GraniteGuardianCallbackHandler when content is flagged unsafe."""


class InputContentSafetyError(ContentSafetyError):
    """Raised when the user's input is flagged unsafe."""


class ToolContentSafetyError(ContentSafetyError):
    """Raised when a tool result is flagged unsafe."""
