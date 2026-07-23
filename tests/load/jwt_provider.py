"""Self-signed JWT provider for Locust isolation tests.

Generates an RSA-2048 key pair at import time and runs a lightweight
JWKS HTTP server on a random port in a daemon thread. Each Locust user
calls ``create_user_token(user_id)`` to get a JWT with a unique ``sub``
claim, validated by the app through the real JWKS auth path.
"""

import base64
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "locust-isolation-test"
_TOKEN_LIFETIME_SECONDS = 3600

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()

_public_pem = _public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

_public_numbers = _public_key.public_numbers()


def _int_to_base64url(n: int) -> str:
    byte_length = (n.bit_length() + 7) // 8
    raw = n.to_bytes(byte_length, byteorder="big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


_JWKS_RESPONSE = json.dumps(
    {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "locust-test-key",
                "n": _int_to_base64url(_public_numbers.n),
                "e": _int_to_base64url(_public_numbers.e),
            }
        ]
    }
).encode("utf-8")


class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/jwks":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_JWKS_RESPONSE)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


JWKS_PORT = _find_free_port()
_server = HTTPServer(("127.0.0.1", JWKS_PORT), _JWKSHandler)
_thread = threading.Thread(target=_server.serve_forever, daemon=True)
_thread.start()

JWKS_URL = f"http://127.0.0.1:{JWKS_PORT}/jwks"


def create_user_token(user_id: str) -> str:
    """Sign a JWT with the given user_id as the ``sub`` claim."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iss": ISSUER,
        "iat": now,
        "exp": now + _TOKEN_LIFETIME_SECONDS,
        "name": f"Locust {user_id}",
    }
    return jwt.encode(
        payload,
        _private_key,
        algorithm="RS256",
        headers={"kid": "locust-test-key"},
    )
