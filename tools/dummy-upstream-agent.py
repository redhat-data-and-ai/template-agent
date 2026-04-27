"""Dummy upstream agent that calls the template-agent via A2A protocol.

Simulates an upstream agent sending a request with authentication and
identity headers, then prints what came back.

Usage:
    python tools/dummy-upstream-agent.py [--url URL] [--message MSG]

Defaults:
    --url     http://localhost:8082/a2a/
    --message "What is 2 multiplied by 3?"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import httpx

AGENT_ID = "dummy-upstream-agent"
BEARER_TOKEN = "upstream-bearer-token-xyz"
CORRELATION_ID = f"corr-{uuid.uuid4().hex[:8]}"


def send_a2a_message(url: str, message: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "X-Calling-Agent-ID": AGENT_ID,
        "X-Correlation-ID": CORRELATION_ID,
    }

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": message}],
                "metadata": {
                    "user_id": "upstream-user-1",
                    "session_id": "upstream-session-1",
                },
            },
        },
    }

    print(f"{'='*60}")
    print(f"UPSTREAM AGENT: {AGENT_ID}")
    print(f"Target: {url}")
    print(f"Correlation ID: {CORRELATION_ID}")
    print(f"Bearer token: {BEARER_TOKEN}")
    print(f"Message: {message}")
    print(f"{'='*60}\n")

    print("Sending request...")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)

    print(f"HTTP Status: {resp.status_code}")
    print(f"Response Headers:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")

    body = resp.json()
    print(f"\nResponse Body:")
    print(json.dumps(body, indent=2))

    result = body.get("result", {})
    parts = result.get("parts", [])
    for part in parts:
        if part.get("kind") == "text":
            print(f"\nAgent replied: {part['text'][:200]}")

    return body


def main():
    parser = argparse.ArgumentParser(description="Dummy upstream A2A agent")
    parser.add_argument("--url", default="http://localhost:8082/a2a/",
                        help="Template-agent A2A endpoint URL")
    parser.add_argument("--message", default="What is 2 multiplied by 3?",
                        help="Message to send")
    args = parser.parse_args()

    try:
        send_a2a_message(args.url, args.message)
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to {args.url}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
