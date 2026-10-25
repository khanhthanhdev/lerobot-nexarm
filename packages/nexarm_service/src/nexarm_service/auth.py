# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Bearer-token protection for an engine that listens beyond loopback.

On 127.0.0.1 the engine needs no token (only local processes reach it). A remote-engine role (a station
hosting its follower for a remote operator) opts in to a non-loopback bind, and must then be started with
a token: every HTTP request needs ``Authorization: Bearer <token>`` and every WebSocket ``?token=<token>``.
"""

import hmac
import ipaddress
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_token_for_bind(host: str, token: str | None) -> None:
    """Refuse to listen beyond loopback without a token."""
    if not is_loopback(host) and not token:
        raise SystemExit(
            f"refusing to listen on {host} without --token: an engine reachable from other machines must "
            "require a token (or keep --host 127.0.0.1 and use an SSH tunnel)"
        )


def tokens_match(given: str | None, expected: str) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


class TokenAuthMiddleware:
    """Pure ASGI middleware, so it covers WebSocket upgrades as well as HTTP."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return
        given = self._bearer(scope) or self._query_token(scope)
        if tokens_match(given, self._token):
            await self._app(scope, receive, send)
            return
        if kind == "websocket":
            await receive()  # the connect event, then refuse the upgrade
            await send({"type": "websocket.close", "code": 4401})
            return
        body = json.dumps({"detail": "a valid bearer token is required"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _bearer(scope: Scope) -> str | None:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                text = value.decode("latin-1")
                if text.lower().startswith("bearer "):
                    return text[7:].strip()
        return None

    @staticmethod
    def _query_token(scope: Scope) -> str | None:
        # Only WebSocket clients (which cannot set headers from a browser) may pass the token in the URL.
        if scope["type"] != "websocket":
            return None
        values = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("token")
        return values[0] if values else None
