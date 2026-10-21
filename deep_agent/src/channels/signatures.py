"""Request signature verification for inbound channels.

This is the whole security boundary for a channel. The endpoint is
reachable from the internet by design, carries no SSO token, and starts
real runs — so a mistake here is not a degraded feature, it is an open
door.

Three rules, each of which has been a real CVE in somebody's webhook
handler:

* Compare with :func:`hmac.compare_digest`, never ``==``. A plain
  comparison leaks the correct prefix through timing.
* Enforce a timestamp window. A valid signature is valid forever
  otherwise, so one captured request replays indefinitely.
* Sign the raw body, never the parsed and re-serialised one. Key order
  and whitespace change under a round trip and the digest stops matching
  — or worse, matches something the handler did not actually see.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    """Whether a signature checked out, and why not when it did not."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        """Truthy when verification passed, so callers can use `if result:`."""
        return self.ok


def _fresh(timestamp: str, max_age_seconds: int, now: float) -> VerificationResult:
    try:
        sent_at = float(timestamp)
    except (TypeError, ValueError):
        return VerificationResult(False, "timestamp is not a number")

    # `float()` accepts "nan", "inf" and "-inf". Every comparison
    # against NaN is False, so a NaN timestamp fell through both
    # bounds below and was reported fresh — which removes the replay
    # window entirely for anyone who captures one signed request.
    if not math.isfinite(sent_at):
        return VerificationResult(False, "timestamp is not a finite number")

    age = now - sent_at
    if age > max_age_seconds:
        return VerificationResult(False, f"timestamp too old ({int(age)}s)")
    # Clock skew cuts both ways; a request from the future is as
    # suspicious as a stale one and the window should be symmetric.
    if age < -max_age_seconds:
        return VerificationResult(
            False, f"timestamp too far in the future ({int(-age)}s)"
        )
    return VerificationResult(True)


def verify_slack(
    *,
    body: bytes,
    timestamp: str,
    signature: str,
    secret: str,
    max_age_seconds: int = 300,
    now: float | None = None,
) -> VerificationResult:
    """Verify Slack's v0 request signature.

    Slack signs ``v0:{timestamp}:{raw_body}`` with the app's signing
    secret and sends the hex digest as ``v0={digest}``.
    """
    if not secret:
        return VerificationResult(False, "no signing secret configured")
    if not signature or not timestamp:
        return VerificationResult(False, "missing signature or timestamp header")

    fresh = _fresh(timestamp, max_age_seconds, time.time() if now is None else now)
    if not fresh:
        return fresh

    basestring = b"v0:" + timestamp.encode() + b":" + body
    digest = hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"

    # Bytes, not str. `compare_digest` on two strings raises
    # TypeError when either holds a non-ASCII character, and the
    # signature arrives in a header an attacker controls, so a
    # crafted one turned a verification failure into a 500.
    if not hmac.compare_digest(expected.encode(), signature.encode("utf-8", "replace")):
        return VerificationResult(False, "signature mismatch")
    return VerificationResult(True)


def verify_webhook(
    *,
    body: bytes,
    timestamp: str,
    signature: str,
    secret: str,
    max_age_seconds: int = 300,
    now: float | None = None,
) -> VerificationResult:
    """Verify the generic webhook signature.

    Same shape as Slack's, minus the vendor prefix: HMAC-SHA256 over
    ``{timestamp}.{raw_body}``, sent as a bare hex digest in
    ``X-Signature``. A ``sha256=`` prefix is accepted because several
    senders add one.
    """
    if not secret:
        return VerificationResult(False, "no signing secret configured")
    if not signature or not timestamp:
        return VerificationResult(False, "missing signature or timestamp header")

    fresh = _fresh(timestamp, max_age_seconds, time.time() if now is None else now)
    if not fresh:
        return fresh

    basestring = timestamp.encode() + b"." + body
    digest = hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    candidate = signature[7:] if signature.startswith("sha256=") else signature

    if not hmac.compare_digest(digest, candidate):
        return VerificationResult(False, "signature mismatch")
    return VerificationResult(True)


def sign_webhook(*, body: bytes, timestamp: str, secret: str) -> str:
    """Produce a signature the way a sender would.

    Exists so tests sign the way the verifier expects rather than
    re-implementing the scheme and agreeing with themselves.
    """
    basestring = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
