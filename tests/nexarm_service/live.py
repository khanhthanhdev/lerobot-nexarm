# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Run an engine app on a real loopback socket, for tests that need a second engine or WebSocket clients."""

import socket
import threading
import time

import uvicorn


def wait_for(predicate, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class LiveEngine:
    def __init__(self, app):
        self.app = app
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        assert wait_for(lambda: self.server.started, 20)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(20)
