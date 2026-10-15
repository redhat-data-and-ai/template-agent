"""URL ingestion is a server-side fetcher driven by user input — SSRF.

The agent sits inside the cluster with reach to Postgres, Redis, the
MCP servers and the cloud metadata endpoint. Every test here is a way
an attacker would try to turn "add a source" into a proxy.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_agent.src.knowledge.fetch import (
    UnsafeURLError,
    _html_to_text,
    assert_safe_url,
)


def resolving_to(address: str):
    """Pretend DNS resolves every name to one address."""
    return patch(
        "socket.getaddrinfo",
        return_value=[(2, 1, 6, "", (address, 0))],
    )


class TestBlockedDestinations:
    @pytest.mark.parametrize(
        "address,what",
        [
            ("127.0.0.1", "loopback"),
            ("10.0.0.5", "private class A"),
            ("172.16.0.1", "private class B"),
            ("192.168.1.1", "private class C"),
            ("169.254.169.254", "cloud metadata"),
            ("0.0.0.0", "unspecified"),
            ("::1", "IPv6 loopback"),
            ("fd00::1", "IPv6 unique local"),
        ],
    )
    def test_non_public_addresses_are_refused(self, address, what):
        with resolving_to(address), pytest.raises(UnsafeURLError):
            assert_safe_url(f"https://anything.example.com/{what}")

    def test_a_public_address_is_allowed(self):
        with resolving_to("93.184.216.34"):
            assert_safe_url("https://example.com/page")

    def test_every_resolved_address_is_checked_not_just_the_first(self):
        """A host can answer with one public and one private address."""
        with (
            patch(
                "socket.getaddrinfo",
                return_value=[
                    (2, 1, 6, "", ("93.184.216.34", 0)),
                    (2, 1, 6, "", ("10.0.0.1", 0)),
                ],
            ),
            pytest.raises(UnsafeURLError),
        ):
            assert_safe_url("https://split-horizon.example.com")


class TestSchemes:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://example.com",
            "ftp://example.com/x",
            "data:text/plain,hello",
        ],
    )
    def test_only_http_and_https_are_fetched(self, url):
        with pytest.raises(UnsafeURLError):
            assert_safe_url(url)

    def test_a_url_with_no_host_is_refused(self):
        with pytest.raises(UnsafeURLError):
            assert_safe_url("http:///nohost")

    def test_an_unresolvable_host_is_refused(self):
        import socket as socket_module

        with patch("socket.getaddrinfo", side_effect=socket_module.gaierror):
            with pytest.raises(UnsafeURLError):
                assert_safe_url("https://does-not-exist.invalid")


class TestHtmlToText:
    def test_the_title_is_extracted(self):
        title, _ = _html_to_text(
            "<html><title>Leave policy</title><body>x</body></html>", "u"
        )
        assert title == "Leave policy"

    def test_the_url_is_the_fallback_title(self):
        title, _ = _html_to_text("<html><body>x</body></html>", "https://e.com/p")
        assert title == "https://e.com/p"

    def test_script_and_style_content_is_dropped(self):
        """Neither is prose, and both poison retrieval."""
        _, text = _html_to_text(
            "<body><script>var secret=1</script><style>.a{}</style><p>real words</p></body>",
            "u",
        )
        assert "real words" in text
        assert "secret" not in text
        assert ".a{}" not in text

    def test_entities_are_decoded(self):
        _, text = _html_to_text("<p>caf&eacute; &amp; bar</p>", "u")
        assert "café & bar" in text

    def test_block_tags_become_line_breaks(self):
        _, text = _html_to_text("<p>one</p><p>two</p>", "u")
        assert "one" in text and "two" in text
        assert "onetwo" not in text.replace(" ", "").replace("\n", "X")
