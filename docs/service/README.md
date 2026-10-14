# NexArm engine

`lerobot-nexarm-service` is the local engine behind the NexArm desktop app (`nexarm-desktop`, Lab pages). It runs the hardware sessions and the ML work on your machine and talks to the app over HTTP and one WebSocket. The app can start and stop it for you; you can also run it yourself.

## Run it

```bash
uv sync --extra service                       # fastapi, uvicorn, websockets and the dataset and NexArm extras
uv run lerobot-nexarm-service                 # http://127.0.0.1:8765
uv run lerobot-nexarm-service --home D:/nexarm-state   # profiles, jobs and curation live here
```

State lives under `~/.cache/lerobot/nexarm_service` by default (override with `--home` or `LEROBOT_NEXARM_HOME`). Datasets are read from and written to `$HF_LEROBOT_HOME`, like `lerobot-record`.

## The contract

| File                           | What it is                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------- |
| [`openapi.json`](openapi.json) | Every HTTP route, committed. A test fails when it is stale.                           |
| [`events.json`](events.json)   | The `WS /api/v1/ws/events` event types (OpenAPI does not describe WebSockets).        |
| `/api/v1/health`               | `version`, `api_version` and `capabilities`; the desktop refuses another API version. |

Regenerate the snapshots after a route change, then refresh the desktop's copies with its `scripts/sync-engine-contract`:

```bash
uv run lerobot-nexarm-service --dump-openapi > docs/service/openapi.json
uv run lerobot-nexarm-service --dump-events > docs/service/events.json
```

## What it does

- **Sessions** (one at a time, kept alive by a 20 s heartbeat; 60 s of silence safety-stops it): teleoperation, recording, replay, rollout, eval, coaching (DAgger), hosting and remote teleoperation.
- **Datasets**: list, inspect, episode joints and decoded video frames, curation (an excluded-episodes list; nothing is deleted), edits that always write a new dataset (delete, split, merge, remove features, rename), Hub upload and import.
- **Training jobs**: a queue with a local runner (`lerobot-train` as a subprocess) and a remote runner (below), persisted logs and metrics, checkpoints, and a guard that refuses a checkpoint trained on another robot type.
- **Calibration check**: reports leader and follower differences at the vendor Home pose. Zero offsets live in the servos; the engine's calibration files are always identity (older ones are backed up and reset).

## Safety rules the engine enforces

- Every session captures a **rest pose** from validated reads and, on stop, ramps the follower back to it before torque is cut. The ramp steps from its own last command, never from a raw read, so a corrupt read cannot make a joint jump.
- Commands are clamped to soft joint limits derived from the URDF (not yet validated on hardware) and the gripper range 1195 to 2833 ticks.
- A stop requested while a session is still starting waits for the start to finish, then stops it.
- The driver's idle auto-relax is off during sessions.

## Remote training

A GPU server you own or rent runs its own engine. The app opens an SSH tunnel to it, so the server's engine stays on `127.0.0.1`; it uploads the dataset, runs the job there and downloads the checkpoints. Prepare a fresh Linux server once:

```bash
scripts/nexarm-gpu-bootstrap.sh            # from a lerobot-nexarm checkout on the server
scripts/nexarm-gpu-bootstrap.sh --repo URL # or clone it
```

Add the server in the app (Lab: Train, GPU servers). SSH keys stay with your SSH agent. `ssh` uses `BatchMode=yes`, so a missing key fails fast instead of hanging.

## Remote teleoperation

The operator's engine streams its leader over a WebSocket to a station engine that hosts the follower. An engine that listens beyond `127.0.0.1` refuses to start without a token:

```bash
uv run lerobot-nexarm-service --host 100.64.0.7 --token "$(openssl rand -hex 24)"   # or NEXARM_ENGINE_TOKEN
```

Every request then needs `Authorization: Bearer <token>` (WebSocket: `?token=`). The station keeps the follower parked at rest until an operator engages, limits speed, and parks again when the operator leaves or goes quiet for a second.

## Tests

```bash
uv run pytest tests/nexarm_service -q      # the engine, with fake hardware (no robot needed)
```

`tests/fixtures/nexarm_protocol_vectors.json` holds golden CommProtocol frames shared with the desktop app; `scripts/gen-nexarm-protocol-vectors.py` regenerates it.

## Hardware not yet verified

Soft limits from the URDF; rest capture happening after torque is enabled; the follower ESP32 restart when bridge mode is exited; the leader following a policy through CMD 97/98; a two-machine remote teleop run. The desktop repo keeps the checklist.

## Behaviour changes for existing CLI users

`NexArmLeader` now exposes `feedback_features`, `send_feedback`, `enable_torque` and `disable_torque` (CMD 97 and 98), so the library's actuated-teleop handovers apply to it: `lerobot-rollout --strategy.type=dagger` (and `episodic` with a leader) now drive the leader arm with torque on while the policy runs and free it for a correction, instead of treating it as a passive device. Plain `lerobot-teleoperate` and `lerobot-record` never call `send_feedback` for the NexArm leader and are unchanged.

## Tokens and logs

The desktop and the operator engine send the token in an `Authorization` header. A `?token=` query string is still accepted on WebSocket URLs for clients that cannot set headers, but it shows up in access logs, so prefer the header.
