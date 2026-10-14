# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""A remote GPU server profile and the SSH commands used to reach its engine.

The remote engine binds 127.0.0.1 and is reached through an SSH tunnel, so no port is exposed on a
rented machine. Keys are never read by the engine: ssh uses the OS agent, or the identity file named in
the profile.
"""

import re
import socket
import subprocess
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")

# Written by scripts/nexarm-gpu-bootstrap.sh on the server; starts the engine if it is not running.
REMOTE_START_COMMAND = "$HOME/.nexarm/start-engine.sh"


class RemoteServer(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    host: str
    user: str
    ssh_port: int = Field(22, gt=0, lt=65536)
    # Optional identity file; empty means "use the SSH agent / default keys".
    key_path: str | None = None
    engine_port: int = Field(8765, gt=0, lt=65536)

    @field_validator("host")
    @classmethod
    def _host(cls, v: str) -> str:
        # No leading dash: a host must never be parsed by ssh as an option.
        if not _HOST_RE.match(v):
            raise ValueError("host must be a hostname or IPv4 address")
        return v

    @field_validator("user")
    @classmethod
    def _user(cls, v: str) -> str:
        if not _USER_RE.match(v):
            raise ValueError("user must be a plain account name")
        return v

    @field_validator("key_path")
    @classmethod
    def _key(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        if v.startswith("-") or "\n" in v:
            raise ValueError("key_path is not a valid path")
        return v


_COMMON = [
    "-o",
    "BatchMode=yes",  # never prompt: a password prompt would hang a background engine
    "-o",
    "StrictHostKeyChecking=accept-new",  # trust a new host once, refuse if its key later changes
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]


def _target(server: RemoteServer) -> list[str]:
    base = ["-p", str(server.ssh_port), *_COMMON]
    if server.key_path:
        base += ["-i", server.key_path]
    return [*base, "--", f"{server.user}@{server.host}"]


def tunnel_command(server: RemoteServer, local_port: int) -> list[str]:
    """``ssh -N -L``: forwards 127.0.0.1:local_port to the server's loopback engine port."""
    forward = f"127.0.0.1:{local_port}:127.0.0.1:{server.engine_port}"
    head = ["ssh", "-N", "-L", forward, "-o", "ExitOnForwardFailure=yes"]
    return head + _target(server)


def start_engine_command(server: RemoteServer) -> list[str]:
    return ["ssh", *_target(server), REMOTE_START_COMMAND]


def free_local_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class SshTunnel:
    """Context manager around one ``ssh -N -L`` process."""

    def __init__(self, server: RemoteServer, popen: Any = subprocess.Popen, wait_s: float = 25.0) -> None:
        self._server = server
        self._popen = popen
        self._wait_s = wait_s
        self._proc: Any = None
        self.base_url = ""

    def __enter__(self) -> str:
        port = free_local_port()
        self._proc = self._popen(
            tunnel_command(self._server, port), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        deadline = time.monotonic() + self._wait_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                error = (self._proc.stderr.read() or b"").decode("utf-8", "replace").strip()
                raise ConnectionError(f"ssh to {self._server.host} failed: {error or 'exited early'}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                self.base_url = f"http://127.0.0.1:{port}"
                return self.base_url
            except OSError:
                time.sleep(0.2)
        self.__exit__()
        raise ConnectionError(f"ssh tunnel to {self._server.host} did not come up within {self._wait_s:.0f}s")

    def __exit__(self, *exc: object) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
