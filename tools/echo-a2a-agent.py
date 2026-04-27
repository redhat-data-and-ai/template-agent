"""Minimal A2A echo agent that returns all received HTTP headers in the response.

Run standalone:
    python tools/echo-a2a-agent.py

Or in a container:
    podman run --rm --network dataverse-shared-network --name echo-a2a-agent \
        -p 9090:9090 -v ./tools:/app python:3.12-slim \
        sh -c "pip install starlette uvicorn && python /app/echo-a2a-agent.py"
"""

from __future__ import annotations

import json
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

AGENT_CARD = {
    "name": "Echo A2A Agent",
    "description": "Returns received headers for testing header propagation",
    "url": "http://echo-a2a-agent:9090/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "echo-headers",
            "name": "Echo Headers",
            "description": "Echoes back all received HTTP headers",
            "tags": ["echo", "debug"],
            "examples": ["echo headers"],
        }
    ],
    "supportedInterfaces": [
        {"protocolBinding": "JSONRPC", "url": "http://echo-a2a-agent:9090/"}
    ],
}


async def agent_card(request: Request) -> JSONResponse:
    return JSONResponse(AGENT_CARD)


async def jsonrpc(request: Request) -> JSONResponse:
    headers_received = dict(request.headers)
    body = await request.body()
    try:
        rpc = json.loads(body)
    except Exception:
        rpc = {}

    rpc_id = rpc.get("id", str(uuid.uuid4()))
    params = rpc.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])
    user_text = parts[0].get("text", "") if parts else ""

    print(f"\n{'='*60}")
    print(f"RECEIVED REQUEST: method={rpc.get('method')}")
    print(f"User text: {user_text}")
    print(f"Headers received:")
    for k, v in sorted(headers_received.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    headers_summary = "\n".join(f"  {k}: {v}" for k, v in sorted(headers_received.items()))
    echo_text = (
        f"Echo agent received your message: '{user_text}'\n\n"
        f"HTTP Headers received:\n{headers_summary}"
    )

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "artifacts": [
                {"parts": [{"kind": "text", "text": echo_text}]}
            ],
            "status": {"state": "completed"},
        },
    })


app = Starlette(
    routes=[
        Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
        Route("/", jsonrpc, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9090)
