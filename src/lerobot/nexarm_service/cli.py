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
    parser.add_argument("--home", type=Path, default=None, help="state directory (profiles, jobs)")
    parser.add_argument(
        "--dump-openapi",
        action="store_true",
        help="print the OpenAPI schema (docs/service/openapi.json) and exit",
    )
    args = parser.parse_args(argv)

    if args.dump_openapi:
        from lerobot.nexarm_service.app import openapi_schema

        sys.stdout.write(json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n")
        return

    import uvicorn

    from lerobot.nexarm_service.app import create_app

    uvicorn.run(create_app(args.home), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
