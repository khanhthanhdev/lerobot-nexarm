# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``lerobot-nexarm-service`` entry point."""

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lerobot-nexarm-service", description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; keep 127.0.0.1 unless this engine is an explicit remote role",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--token",
        default=os.environ.get("NEXARM_ENGINE_TOKEN"),
        help="bearer token every request must carry; required when --host is not loopback "
        "(also read from NEXARM_ENGINE_TOKEN)",
    )
    parser.add_argument("--home", type=Path, default=None, help="state directory (profiles, jobs)")
    parser.add_argument(
        "--dump-openapi",
        action="store_true",
        help="print the OpenAPI schema (docs/service/openapi.json) and exit",
    )
    parser.add_argument(
        "--dump-events",
        action="store_true",
        help="print the WebSocket event schema (docs/service/events.json) and exit",
    )
    args = parser.parse_args(argv)

    if args.dump_events:
        from nexarm_service.events import EVENT_SCHEMAS

        sys.stdout.write(json.dumps(EVENT_SCHEMAS, indent=2, sort_keys=True) + "\n")
        return

    if args.dump_openapi:
        from nexarm_service.app import openapi_schema

        sys.stdout.write(json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n")
        return

    import uvicorn

    from lerobot.utils.import_utils import register_third_party_plugins
    from nexarm_service.app import create_app
    from nexarm_service.auth import require_token_for_bind

    # Policy checkpoints name their type (`turbovla`, `bspline`, ...); plugin types resolve only once imported.
    register_third_party_plugins()

    require_token_for_bind(args.host, args.token)
    app = create_app(args.home, token=args.token)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    # POST /api/v1/shutdown asks uvicorn to exit gracefully so the lifespan hook releases the arm.
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
