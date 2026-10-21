"""Signature verification is the whole security boundary for a channel.

These endpoints are reachable from the internet, carry no SSO token and
start real runs. Each test here corresponds to a failure that has been a
real CVE in somebody's webhook handler.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from deep_agent.src.channels.signatures import (
    sign_webhook,
    verify_slack,
    verify_webhook,
)

SECRET = "s3cr3t"
NOW = 1_700_000_000.0
BODY = b'{"hello":"world"}'


def slack_signature(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    base = b"v0:" + timestamp.encode() + b":" + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


class TestSlack:
    def test_a_genuine_request_verifies(self):
        ts = str(int(NOW))
        assert verify_slack(
            body=BODY,
            timestamp=ts,
            signature=slack_signature(BODY, ts),
            secret=SECRET,
            now=NOW,
        )

    def test_a_tampered_body_fails(self):
        """The signature covers the body, so changing it must break."""
        ts = str(int(NOW))
        result = verify_slack(
            body=b'{"hello":"evil"}',
            timestamp=ts,
            signature=slack_signature(BODY, ts),
            secret=SECRET,
            now=NOW,
        )
        assert not result
        assert "mismatch" in result.reason

    def test_the_wrong_secret_fails(self):
        ts = str(int(NOW))
        assert not verify_slack(
            body=BODY,
            timestamp=ts,
            signature=slack_signature(BODY, ts, secret="other"),
            secret=SECRET,
            now=NOW,
        )

    def test_a_stale_request_fails_even_with_a_valid_signature(self):
        """Otherwise one captured request replays forever."""
        ts = str(int(NOW - 3600))
        result = verify_slack(
            body=BODY,
            timestamp=ts,
            signature=slack_signature(BODY, ts),
            secret=SECRET,
            max_age_seconds=300,
            now=NOW,
        )
        assert not result
        assert "too old" in result.reason

    def test_a_future_request_fails(self):
        """Clock skew cuts both ways; the window must be symmetric."""
        ts = str(int(NOW + 3600))
        result = verify_slack(
            body=BODY,
            timestamp=ts,
            signature=slack_signature(BODY, ts),
            secret=SECRET,
            max_age_seconds=300,
            now=NOW,
        )
        assert not result
        assert "future" in result.reason

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"signature": ""},
            {"timestamp": ""},
            {"secret": ""},
        ],
        ids=["no-signature", "no-timestamp", "no-secret"],
    )
    def test_missing_pieces_fail_closed(self, kwargs):
        ts = str(int(NOW))
        args = {
            "body": BODY,
            "timestamp": ts,
            "signature": slack_signature(BODY, ts),
            "secret": SECRET,
            "now": NOW,
        }
        assert not verify_slack(**{**args, **kwargs})

    def test_a_non_numeric_timestamp_fails(self):
        assert not verify_slack(
            body=BODY,
            timestamp="not-a-number",
            signature="v0=whatever",
            secret=SECRET,
            now=NOW,
        )


class TestWebhook:
    def test_a_genuine_request_verifies(self):
        ts = str(int(NOW))
        assert verify_webhook(
            body=BODY,
            timestamp=ts,
            signature=sign_webhook(body=BODY, timestamp=ts, secret=SECRET),
            secret=SECRET,
            now=NOW,
        )

    def test_a_sha256_prefix_is_accepted(self):
        """Several senders add one; rejecting it is a support ticket."""
        ts = str(int(NOW))
        signature = sign_webhook(body=BODY, timestamp=ts, secret=SECRET)
        assert verify_webhook(
            body=BODY,
            timestamp=ts,
            signature=f"sha256={signature}",
            secret=SECRET,
            now=NOW,
        )

    def test_a_tampered_body_fails(self):
        ts = str(int(NOW))
        assert not verify_webhook(
            body=b'{"hello":"evil"}',
            timestamp=ts,
            signature=sign_webhook(body=BODY, timestamp=ts, secret=SECRET),
            secret=SECRET,
            now=NOW,
        )

    def test_a_slack_signature_does_not_verify_as_a_webhook_one(self):
        """The two schemes must not be interchangeable."""
        ts = str(int(NOW))
        assert not verify_webhook(
            body=BODY,
            timestamp=ts,
            signature=slack_signature(BODY, ts),
            secret=SECRET,
            now=NOW,
        )

    def test_whitespace_changes_break_the_signature(self):
        """Why the raw body is signed, never a re-serialised one.

        A handler that parses then re-dumps JSON before verifying will
        pass on bodies it never actually saw.
        """
        ts = str(int(NOW))
        signature = sign_webhook(body=b'{"a":1}', timestamp=ts, secret=SECRET)
        assert not verify_webhook(
            body=b'{"a": 1}',
            timestamp=ts,
            signature=signature,
            secret=SECRET,
            now=NOW,
        )
