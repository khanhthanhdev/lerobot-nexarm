"""Extract the NexArm robot and workcell from Fusion 360 over MCP, then merge it into the MuJoCo models.

Runs ``sim/exporter/fusion_export_all.py`` inside Fusion (meshes + ``fusion_extract.json``), then
``sim/apply_fusion_extract.py`` on this checkout. Works from Windows, or from WSL with Fusion on the Windows host.

Usage:
    uv run python sim/run_export.py
    uv run python sim/run_export.py --sim-dir /tmp/nexarm_trial --skip-apply
"""

import argparse
import json
import os
import platform
import subprocess  # nosec B404 - runs wslpath only
import sys
import urllib.request
from pathlib import Path

MCP_URL = os.environ.get("FUSION_MCP_URL", "http://127.0.0.1:27182/mcp")
SIM_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = SIM_DIR / "exporter" / "fusion_export_all.py"
PLACEHOLDER = 'EXPORT_SIM_DIR = ""'


def fusion_visible_path(path: Path) -> str:
    """Return ``path`` as Fusion (a Windows process) sees it."""
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()  # nosec B603 B607
    return str(path)


def build_script(sim_dir: Path) -> str:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    if script.count(PLACEHOLDER) != 1:
        raise RuntimeError(f"{SCRIPT_PATH} must contain exactly one line '{PLACEHOLDER}'")
    return script.replace(PLACEHOLDER, f"EXPORT_SIM_DIR = {fusion_visible_path(sim_dir)!r}")


def _post(payload: dict, session_id: str | None = None, timeout: float = 5) -> tuple[dict | None, str | None]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session_id:
        headers["MCP-Session-Id"] = session_id
    request = urllib.request.Request(MCP_URL, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # nosec B310 - local MCP endpoint
        body = resp.read().decode("utf-8")
        new_session = resp.headers.get("MCP-Session-Id")
    if not body.strip():
        return None, new_session
    if body.lstrip().startswith("{"):
        return json.loads(body), new_session
    # Server-sent events: take the last data line.
    data = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
    return (json.loads(data[-1]) if data else None), new_session


def call_fusion(script: str) -> str:
    """Run ``script`` in Fusion and return its printed output; raise if it failed."""
    init, session_id = _post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nexarm-exporter", "version": "2.0"},
            },
        }
    )
    if not session_id:
        raise RuntimeError("MCP server did not return a session ID")
    server = (init or {}).get("result", {}).get("serverInfo", {}).get("name", "MCP server")
    print(f"Connected to {server}")
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id)

    result, _ = _post(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fusion_mcp_execute",
                "arguments": {"featureType": "script", "object": {"script": script}},
            },
        },
        session_id,
        timeout=600,
    )
    if result is None or "error" in result:
        raise RuntimeError(f"MCP call failed: {result}")
    texts = [
        item.get("text", "") for item in result["result"].get("content", []) if item.get("type") == "text"
    ]
    output = "\n".join(texts)
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if result["result"].get("isError") or (isinstance(parsed, dict) and parsed.get("success") is False):
        raise RuntimeError(f"Fusion script failed:\n{parsed.get('error', output) if parsed else output}")
    return parsed.get("message", output) if isinstance(parsed, dict) else output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sim-dir", type=Path, default=SIM_DIR, help="Output sim/ directory (default: this repo)"
    )
    parser.add_argument("--skip-apply", action="store_true", help="Only extract; do not update the models")
    args = parser.parse_args()
    sim_dir = args.sim_dir.resolve()

    print(f"Extracting from Fusion into {sim_dir} via {MCP_URL} ...")
    try:
        print(call_fusion(build_script(sim_dir)))
    except OSError as exc:
        print(f"Failed to reach the Fusion MCP server: {exc}")
        print("Make sure Fusion is running with the MCP add-in active.")
        sys.exit(1)

    if args.skip_apply:
        return
    from apply_fusion_extract import apply_extract  # noqa: PLC0415 - sibling script

    apply_extract(sim_dir / "fusion_export" / "fusion_extract.json", sim_dir)


if __name__ == "__main__":
    main()
