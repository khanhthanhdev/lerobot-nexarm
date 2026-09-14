"""CLI Runner to trigger Fusion 360 robot export from Linux via MCP.

Usage:
    uv run python sim/run_export.py
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

MCP_URL = os.environ.get("FUSION_MCP_URL", "http://127.0.0.1:27182/mcp")
SCRIPT_PATH = Path(__file__).parent / "exporter" / "fusion_export_all.py"


def run_export() -> None:
    if not SCRIPT_PATH.exists():
        print(f"Error: Exporter script not found at {SCRIPT_PATH}")
        sys.exit(1)

    with open(SCRIPT_PATH, encoding="utf-8") as f:
        script_code = f.read()

    print(f"Connecting to Fusion 360 MCP server at {MCP_URL}...")

    # 1. Initialize MCP session
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "nexarm-exporter", "version": "1.0"},
        },
    }

    try:
        init_req = urllib.request.Request(
            MCP_URL,
            data=json.dumps(init_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(init_req, timeout=5) as resp:  # nosec B310
            session_id = resp.headers.get("MCP-Session-Id")
            init_res = json.loads(resp.read().decode("utf-8"))
            server_info = init_res.get("result", {}).get("serverInfo", {})
            print(f"Connected to MCP Server: {server_info.get('name', 'MCP Server')}")

        if not session_id:
            raise RuntimeError("MCP server did not return a session ID")

        # 2. Send notifications/initialized
        notif_req = urllib.request.Request(
            MCP_URL,
            data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "MCP-Session-Id": session_id},
        )
        with urllib.request.urlopen(notif_req, timeout=5) as resp:  # nosec B310
            pass

    except Exception as e:
        print(f"Failed to connect to Fusion 360 MCP server: {e}")
        print("Please ensure Autodesk Fusion 360 is running with the MCP add-in active.")
        sys.exit(1)

    # 3. Call fusion_mcp_execute
    print("Executing export script inside Autodesk Fusion 360...")
    call_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "fusion_mcp_execute",
            "arguments": {
                "featureType": "script",
                "object": {"script": script_code},
            },
        },
    }

    call_req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(call_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "MCP-Session-Id": session_id},
    )
    with urllib.request.urlopen(call_req, timeout=60) as resp:  # nosec B310
        call_res = json.loads(resp.read().decode("utf-8"))

    content = call_res.get("result", {}).get("content", [])
    for item in content:
        if item.get("type") == "text":
            raw_text = item.get("text", "")
            try:
                parsed = json.loads(raw_text)
                print(parsed.get("message", raw_text))
            except Exception:
                print(raw_text)

    print("Export completed successfully!")


if __name__ == "__main__":
    run_export()
