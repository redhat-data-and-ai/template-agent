"""Fetch a URL to add as a source.

This is a server-side fetcher driven by a user-supplied URL, which is
the textbook SSRF primitive. The agent runs inside the cluster with
network access to Postgres, Redis, the MCP servers and — on OpenShift —
the cloud metadata endpoint. A naive implementation here hands an
attacker a proxy into all of it.

So: resolve the host first, check every resolved address against the
private ranges, and re-check after each redirect. Checking the hostname
alone is not enough — DNS can point anywhere, and a redirect can cross
from public to private between the check and the request.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

MAX_BYTES = settings.KNOWLEDGE_FETCH_MAX_BYTES
TIMEOUT_SECONDS = settings.KNOWLEDGE_FETCH_TIMEOUT_SECONDS
MAX_REDIRECTS = settings.KNOWLEDGE_FETCH_MAX_REDIRECTS


class UnsafeURLError(ValueError):
    """The URL resolves somewhere we refuse to fetch from."""


def _is_blocked(address: str) -> bool:
    """True for anything not on the public internet.

    `is_global` covers private, loopback, link-local, multicast and
    reserved in one check — including the 169.254.169.254 metadata
    address, which is link-local.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return not ip.is_global


def assert_safe_url(url: str) -> list[str]:
    """Reject anything that is not plain HTTP(S) to a public address.

    Returns the validated addresses so the caller can connect to one
    of them rather than resolving the name a second time.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"Only http and https URLs are supported, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise UnsafeURLError("URL has no host")

    try:
        # Every address the name resolves to, not just the first: a
        # host can return one public and one private answer, and which
        # one httpx connects to is not ours to predict.
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve {parsed.hostname}") from exc

    addresses: list[str] = []
    for info in infos:
        # sockaddr is (host, port) for IPv4 and (host, port, flow, scope)
        # for IPv6, so the element is typed `str | int` even though the
        # first one is always the address.
        address = str(info[4][0])
        if _is_blocked(address):
            raise UnsafeURLError(
                f"{parsed.hostname} resolves to a non-public address and will not be fetched"
            )
        addresses.append(address)
    return addresses


def fetch_url(url: str) -> tuple[str, str]:
    """Fetch a page. Returns ``(title, text)``.

    Redirects are followed manually so each hop is re-validated —
    `follow_redirects=True` would check the first URL and then happily
    follow a 302 into the cluster.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        addresses = assert_safe_url(current)

        # Connect to the address that was checked, not to whatever the
        # name resolves to next.
        #
        # `assert_safe_url` resolves once and httpx used to resolve
        # again on its own, so a host that answers with a public
        # address for the first lookup and 169.254.169.254 for the
        # second passed the check and was then fetched anyway. The
        # address goes in the URL; `Host` and the TLS SNI keep the
        # original name, so virtual hosting and certificate
        # verification both still work.
        parsed = httpx.URL(current)
        hostname = parsed.host
        pinned = parsed.copy_with(host=addresses[0])

        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
            with client.stream(
                "GET",
                pinned,
                headers={"User-Agent": "template-agent/1.0", "Host": hostname},
                extensions={"sni_hostname": hostname},
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafeURLError("Redirect without a location")
                    current = str(httpx.URL(current).join(location))
                    continue

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    raise UnsafeURLError(
                        f"Unsupported content type: {content_type or 'unknown'}"
                    )

                # Read to the cap and stop. `response.content` buffered
                # the whole body first and truncated afterwards, so a
                # server that never stops sending could exhaust memory
                # before the slice ever ran.
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= MAX_BYTES:
                        break
                body = b"".join(chunks)[:MAX_BYTES]

        return _html_to_text(body.decode("utf-8", errors="replace"), current)

    raise UnsafeURLError("Too many redirects")


def _html_to_text(html: str, url: str) -> tuple[str, str]:
    """Strip markup, keeping the title and readable body.

    Deliberately regex rather than a parser dependency: the corpus only
    needs words, and script/style removal plus tag stripping gets
    99% of the way there for a fraction of the surface area.
    """
    import html as html_module
    import re

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = html_module.unescape(title_match.group(1)).strip() if title_match else url

    # Script and style content is not prose and poisons retrieval.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return title.strip() or url, text.strip()
