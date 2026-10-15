"""SSRF guard on outbound fetches.

The agent fetches URLs a user supplies, from inside the cluster. The
address that matters most is 169.254.169.254 — the cloud metadata
endpoint, which hands out credentials to anything that asks.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from deep_agent.src.knowledge.fetch import (
    UnsafeURLError,
    _is_blocked,
    assert_safe_url,
)


class TestIsBlocked:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",  # cloud metadata
            "::1",
            "fe80::1",
        ],
    )
    def test_blocks_everything_off_the_public_internet(self, address):
        assert _is_blocked(address) is True

    @pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "2606:4700::1111"])
    def test_allows_public_addresses(self, address):
        assert _is_blocked(address) is False

    def test_blocks_anything_that_is_not_an_address(self):
        # Fail closed: an unparseable value is not demonstrably safe.
        assert _is_blocked("not-an-ip") is True


def _addrinfo(*addresses: str):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
        for address in addresses
    ]


class TestAssertSafeUrl:
    def test_allows_a_public_https_url(self):
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            assert_safe_url("https://example.com/page")

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x", "ftp://x/y"])
    def test_rejects_other_schemes(self, url):
        with pytest.raises(UnsafeURLError, match="http and https"):
            assert_safe_url(url)

    def test_rejects_a_url_with_no_host(self):
        with pytest.raises(UnsafeURLError, match="no host"):
            assert_safe_url("http:///just-a-path")

    def test_rejects_a_name_that_does_not_resolve(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            with pytest.raises(UnsafeURLError, match="Could not resolve"):
                assert_safe_url("https://nowhere.invalid/")

    def test_rejects_a_name_resolving_to_a_private_address(self):
        with patch("socket.getaddrinfo", return_value=_addrinfo("10.1.2.3")):
            with pytest.raises(UnsafeURLError, match="non-public"):
                assert_safe_url("https://internal.example.com/")

    def test_rejects_when_any_answer_is_private(self):
        # A host can return one public and one private answer, and
        # which one httpx connects to is not ours to predict.
        with patch(
            "socket.getaddrinfo",
            return_value=_addrinfo("93.184.216.34", "169.254.169.254"),
        ):
            with pytest.raises(UnsafeURLError, match="non-public"):
                assert_safe_url("https://dns-rebind.example.com/")
