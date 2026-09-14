<img src="assets/banner.png" alt="Embodied Metal Hackathon" width="100%">

# Embodied Metal Hackathon - Awesome List

Every project built at the Embodied Metal Hackathon (July 17–19, 2026, San Francisco) — SO-101 arms, fine-tuned VLA models, and the harnesses around them. Use this repo to find tools you can hack into your own projects. Point your agents at it and use them as reference implementations.

Ordered by how cloneable they are: clone-and-play first, then runnable-with-setup, then reference-code. Within a grade, the ones with the most reusable pieces come first.

---

## 🏆 Winners

**Main Track**

- 🥇 **1st — [Physical Agents](#-physical-agents--we-built-a-robot-that-cleans-so-the-world-doesnt-have-to)** — $3,000 + $3,000 Modal compute
- 🥈 **2nd — [G1 Goalie](#-g1-goalie--autonomous-goalkeeper-unitree-g1-that-picks-and-throws-a-soccer-ball)** — $2,000 + $2,000 Modal compute
- 🥉 **3rd — [Ctrl-Z](#-ctrl-z--the-physical-undo-tools-you-orchestrate-to-reset-physical-spaces)** — $1,000 + $1,000 Modal compute

**Rerun**

- **Best Rerun Viewer — [Lama](#-lama--the-amazing-xylophone-robo-player)** — $2,000
- **Best Query API — [Physical Agents](#-physical-agents--we-built-a-robot-that-cleans-so-the-world-doesnt-have-to)** — $2,000
- **Best Port of the Rerun Starter — [Physical Agents](#-physical-agents--we-built-a-robot-that-cleans-so-the-world-doesnt-have-to)** — $1,000

**Arduino hardware prizes:** [Dreamscale](#-dreamscale--fast-cloud-inference-for-robotics), [ODIEV3](#-odiev3--odie-is-your-replacement-pet-companionship-without-the-care), and [Cornerf](#-cornerf--an-autonomous-desk-attendant-that-sees-clutter-and-puts-it-away), plus YCSF (see [Inaccessible](#inaccessible-at-harvest-time) below) and Boosterbot (sheet-only — no submission on record).

---

## All projects

### 🥇 Physical Agents — We built a robot that cleans, so the world doesn't have to.

Two SO-101 leader arms drive two i2rt YAM followers through a calibrated joint-space mapping; three RealSense cameras stream to a Rerun dashboard; a web panel records episodes to MP4+NPZ. You triage with `winnow` (real DataFusion SQL over per-episode metrics), convert to a LeRobotDataset, train with LeRobot ACT or fine-tune MolmoAct2 LoRA on Modal, and deploy back through a dry-run-by-default runner with velocity limits and gravity-comp shutdown.

**You get:**

- SO-101-leader → any-follower mapping — the porting seam off SO-101 (`leader_yam_bridge/leader_yam_bridge.py`)
- Multi-leader calibration with atomic config writes (`scripts/calibrate.py`)
- Rerun camera dashboard that records episodes over HTTP (`leader_yam_bridge/_v1/camera_dashboard.py`)
- Stdlib-HTTP connect/teleop/record panel, Rerun view embedded (`control_panel.py`)
- Episode writer, temp-then-rename so interrupted takes never look complete (`episode_writer.py`)
- `winnow` curation — a real DataFusion SQL keep/reject predicate, Query API not viewer logging (`curation/pipeline.py`, `_query.py`)
- LeRobotDataset converter, `--format act`/`molmoact2` (`convert_to_lerobot.py`) + resumable HF Hub push (`push_dataset.py`)
- Autonomous runner, one contract across ACT and VLA — dry-run default, velocity limits, gravity-comp shutdown (`autonomous/`) + a Modal serving endpoint (`autonomous/modal_vla.py`)
- Agent skill for discovery/calibration/mapping validation, won't start motion itself (`skills/connect-yam-leader/`)
- One-command bootstrap (`scripts/initialize_new_project.bash`) + a per-stage Jekyll docs site

**Stack:** Python 3.12 (uv), i2rt YAM/CAN driver, Feetech SO-101, RealSense + OpenCV, Rerun SDK (blueprint, RRD, DataFusion SQL via `winnow`), LeRobot ACT, MolmoAct2 LoRA on Modal, HF Hub, pytest
**Readiness:** clone-and-play — `curl | bash` installer + a quickstart (calibrate → teleop → record → curate → convert → train) + an agent skill that runs setup. Hardware-gated (SO-101 leaders, YAM over CAN, RealSense): complete and runnable, not runs-without-a-robot.
Repo: https://github.com/godbrigero/YamDualArmController
Demos: https://youtu.be/eFW11dJSGGs · https://winnow.adamxu.workers.dev/ · https://youtu.be/Y-z-bsvWtfg · https://godbrigero.github.io/YamDualArmController/
Entered: Save the world, Rerun Viewer, Rerun Query API, Non-SO-101 port

---

### Jags — WE build houses.

A house-builder for a SO-101 arm. A regex parser (no LLM) turns a spoken or typed color request — or a scan of a hand-built model house from a third camera — into three pick-and-place instructions, and a state machine runs them one layer at a time against a Modal-hosted MolmoAct2/ACT server. An OpenCV/HSV verifier exists and matches the writeup, but the README and code comments say it's off by default — the shipped loop is operator-paced (a human hits Pause when a layer looks done), not autonomous verify-and-retry.

**You get:**

- Regex color-request parser, no LLM (`src/house_builder/parser.py`)
- Planner turning colors into the exact instruction strings the policy was trained on (`src/house_builder/planner.py`)
- Build state machine with an explicit allowed-transition table (`src/house_builder/state_machine.py`)
- HSV placement verifier (color band + centroid + support-alignment + multi-frame stability) — off by default (`src/house_builder/verifier.py`)
- Rerun Blueprint: two camera views + verification log + joint time series (`src/house_builder/rr_blueprint.py`) — Viewer only, no Query API
- Reference-scan path: a third-camera hub + FastAPI route + React panel reading colors off a physical model house (`backend/camera_hub.py`, `backend/routes/cam2.py`, `frontend/src/components/ReferenceScan.tsx`)
- Voice entrypoint with a `--text` fallback (`voice_control.py`)
- FastAPI + React/Vite dashboard: controls, live monitor, highlights reel, 3D house preview
- Offline `simulate.py` harness running the full pipeline against fakes — what the tests exercise
- Vendored so100-hackathon stack: Feetech driver, SO-101 calibration JSONs, Modal MolmoAct2 (LoRA + Action Expert)/ACT/Pi0.5/SmolVLA scripts

**Stack:** Python 3.11 (FastAPI, OpenCV/HSV, Rerun SDK, SpeechRecognition/PyAudio) + React/Vite/TS + Modal-hosted MolmoAct2/ACT/Pi0.5/SmolVLA + vendored SO-100/101 Feetech driver
**Readiness:** runnable-with-setup — real install+run path with an offline `simulate.py` mode and passing tests, but the demo needs a physical SO-101, two calibrated cameras, the pixi env, and teammate-hosted Modal. The headline "closed-loop auto-verify-and-retry" is disabled in shipped code (comments say verification and auto-homing "were removed").
Repo: https://github.com/gracexu24/embodiedmetalhack
Entered: Solve a puzzle, Rerun Query API, Rerun Viewer

---

### Nero — Production-grade robots need marked zones and direction so humans nearby stay safe.

Nero drives a Booster K1 Geek humanoid toward a spoken target ("go to the chair") using ORB-SLAM3 for pose, a QNN-accelerated YOLO-World detector (or an ArUco fallback), and pure-pursuit or A\*-over-map — fail-closed the moment any sensor, SLAM state, or detector drops. A separate projector paints a live floor overlay from a ceiling RealSense + HTC Vive pose; it only visualizes state over an HTTP/WebSocket contract, never commands motors. Two pitch claims don't match the code: the safety ring is a fixed 0.28m radius (not ANSI/RIA reach+stopping-distance with auto-inflate), and goals come from a manual x/y/yaw web form, not from pointing at an object.

**You get:**

- Booster K1 Geek object-nav stack: native ORB-SLAM3 (IMU-RGBD) + fail-closed preflight + pure-pursuit and A\* controllers (`src/nero/navigation/`, `src/nero/agents/orb_slam_agent.py`)
- QNN/Qualcomm AI Hub deploy of a text-conditioned YOLO-World detector on the K1 NPU, with CPU and Modal-GPU fallbacks (`src/nero/perception/`, `deploy/modal_perception.py`)
- Deterministic ArUco object-nav fallback on the same pipeline (`src/nero/perception/aruco_detector.py`)
- Ceiling-RealSense floor-calibration projector: homography UI, live floor overlay, HTTP/WebSocket contract that never touches motor control (`src/nero/projector/`)
- HTC Vive → ROS 2 pose bridge with fail-closed staleness/validity gating (`src/nero/vive/`, `deploy/vive_ros/`)
- Rerun bridge + blueprint for full RGB/depth/IMU/SLAM/detection/safety telemetry, browser-servable off the robot network (`src/nero/observability/rerun_bridge.py`, `scripts/run_robot_web.sh`)
- Booster Studio sim scenes + a sim-vs-SLAM benchmark (ATE/RPE/scale drift) (`src/nero/simulation/`)
- Gaussian-splat/COLMAP room-capture → occupancy-map pipeline for the A\* planner (`src/nero/mapping/`)
- 24-file pytest suite + ruff + a Docker IMU-RGBD CI smoke test

**Stack:** Python 3.10 (uv) + ROS 2 Humble + native ORB-SLAM3 (IMU-RGBD) + OpenCV ArUco + YOLO-World (QNN HTP / Modal L4 / CPU) + Rerun SDK (Viewer + blueprint, no Query API found) + aiohttp; targets Booster K1 Geek, RealSense D435i, HTC Vive Lighthouse (Raspberry Pi + libsurvive), Booster Studio sim
**Readiness:** runnable-with-setup — the dev loop (`uv sync --all-groups --locked && uv run pytest -q`, plus a macOS simulator) runs out of the box, but the full demo needs a K1 Geek, a D435i, Vive base stations + a Pi bridge, a ceiling projector, and a Qualcomm AI Hub QNN export.
Repo: https://github.com/nimarez/nero
Entered: Surprise us, Non-SO-101 port, Rerun Query API, Rerun Viewer

---

### 🥉 Ctrl-Z — The Physical Undo: tools you orchestrate to reset physical spaces.

A "reset my desk" agent on NT's so100-hackathon starter kit. A zero-shot vision tool calls Claude once per frame to place objects in table coordinates (no thresholds, no trained classifier); a desk-diff Rerun view compares the live layout to a saved reference with ghost rings + move arrows; an MCP server exposes goto/detect/pick/drop plus a trained grasp policy, so Claude Desktop/Code can say "put the red block back" and it happens. The pickup is a from-scratch ACT policy (also tried as SmolVLA), trained on Modal from ~40–50 teleop episodes recorded slower through the grasp — the part ACT needed more frames of.

**You get:**

- MCP server (`tools/mcp_server.py`, FastMCP) exposing 6 tools to Claude Desktop/Code: detect_objects, goto_location, drop_object, pick_up_underneath, capture_table_reference, compare_table_with_reference
- Zero-shot tabletop detector — one `claude -p` (Sonnet) vision call per frame, no trained model (`tools/detect_objects.py`)
- ArUco calibration CLI: homography from 4 markers, top-down warp, pixel→table coords (`tools/table_vision.py`)
- Hand-taught arm-to-table mapping: torque-off teach of 4 corners → FK plane fit → IK goto, no gripper-offset (`tools/table_arm.py`)
- Live desk-diff Rerun viewer: ghost rings, move arrows, displacement time series that falls to zero as the desk is restored (`tools/desk_diff.py`)
- Modal ACT training (± image aug) + a SmolVLA variant (`act_train.py`, `smolvla_train.py`)
- Checkpoint-fetch script sanitizing a LeRobot `pretrained_model` dir for the kit's `export` pixi env (`scripts/get_checkpoint.sh`)

**Stack:** Python + pixi/uv, LeRobot v3 (ACT + SmolVLA) on Modal (A10/A100), Rerun SDK (viewer + blueprint), OpenCV + ArUco, MCP (FastMCP), Claude (Sonnet) as zero-shot vision detector, SO-100/101 over Feetech
**Readiness:** runnable-with-setup — every tool has a precise usage docstring (commands, env vars, error bounds, failure modes) and the MCP wiring is real, but `README.md` is byte-identical to the upstream kit and never mentions the MCP server, vision/desk-diff pipeline, or trained policies. Rebuilding the demo means reading docstrings + AGENTS.md.
Repo: https://github.com/plotline-insights/so101-hackathon
Entered: Save the world, Rerun Viewer

---

### 🏆 Dreamscale — Fast cloud inference for robotics.

The writeup pitches DROID bringup, VR teleop, and live serving; the repo is one thing — a Modal pipeline running the frozen MolmoAct2-DROID checkpoint over ~40h of selected DROID episodes on 4× B200 GPUs, caching action chunks, prompt-context latents, and ResNet-18 visual tokens into a lazy parquet frame/anchor join. That cache trains a separate 20–25M-param "A2C2" action-chunk-correction transformer — which, with the hardware/teleop stack and the live endpoint, isn't in this repo.

**You get:**

- Modal job graph: metadata/selection → 4× B200 MolmoAct2 inference+caching → finalize/validation, resumable safetensors+SHA256 per part (`modal_app.py`)
- Attention-bias fix for batched MolmoAct2: `build_prefill_attention_bias()` replaces the checkpoint's `_build_native_attention_bias`, which breaks at batch > 1 (`src/a2c2_droid_cache/molmo.py`)
- DROID selection/splitting: stable-hash 80/10/10 with completeness filtering — reusable for large video-dataset subsampling (`selection.py`)
- Lazy anchor/frame join: parquet indices + safetensors shards over full materialization; tau encoding `[sin(2πk/15), cos(2πk/15)]` (`anchors.py`, `batching.py`, `features.py`)
- Pinned HTTP-range video decoding reading DROID intervals without full MP4 shards (253.6GB avoided) (`hf_source.py`)
- pytest over the whole pipeline — real coverage, not a stub

**Stack:** Python 3.11, PyTorch 2.7 + torchvision + transformers (MolmoAct2-DROID from HF), Modal (B200 jobs, Volumes, Secrets), pyarrow/parquet + safetensors for the cache, uv/pyproject
**Readiness:** runnable-with-setup — real commands (`modal volume create`, `modal run ::metadata`, `modal run --detach ::run`) + a full pytest suite, but hard-requires a Modal account, a gated HF token, and B200 GPU quota.
Repo: https://github.com/RedCatAFK/embodied-metal
Demos: https://www.linkedin.com/posts/chris-yoo_how-often-do-you-get-to-set-up-a-40k-ugcPost-7484532445866799104--Ti4/
Entered: Surprise us

---

### 🏆 Lama — The amazing xylophone robo player.

A fine-tuned MolmoAct2 policy that knows one move — strike a single xylophone bar on one instruction — turned into a melody sequencer: `run.py` parses note text, and `sequencer.py` swaps the instruction prompt mid-session over one long-lived WebSocket instead of reconnecting per note. A separate `rerun/` toolkit annotates recorded episodes — strike/retract from joint kinematics alone (no mic or force sensor) — and builds Viewer blueprints comparing takes.

**You get:**

- Prompt-injection sequencer: one newt-SDK `run("")` over a single WebSocket, swapping the obs `"prompt"` per note instead of reconnecting (`sequencer.py`)
- `notes.py`: regex-first text-to-label parser with a constrained-enum Claude fallback, so the model never gets an untrained instruction
- Modal WebSocket policy server for a LeRobot-format MolmoAct2 checkpoint (`server/modal_ws.py`) + `codec.py` (msgpack wire format from the newt SDK)
- `tests/test_contract.py`: downloads a checkpoint's `config.json` and fails if image/chunk size or normalization drift from the client's constants
- `rerun/` scripts: strike-detection (scipy peak-finding on joint velocity/accel), impact-ripple annotations, multi-take comparison blueprints on real `.rrd`
- `GUIDE.md`: a runbook for swapping in any MolmoAct2 checkpoint — 3 edits, contract checks, and a table of real failures (Modal ASGI 500s, stale calibration, silent mismatches)

**Stack:** Python (uv) — newt SDK + vendored lerobot-nt (SO-101 driver), Modal serving MolmoAct2 (LeRobot format), rerun-sdk (`RrdReader`, BarChartView, Spatial3DView), Anthropic API as optional free-text fallback
**Readiness:** runnable-with-setup — code and tests complete with a real run path, but hard-depends on a physical SO-101 + calibrated cameras + a deployed Modal server (or NT hosted inference). The `rerun/` toolkit alone runs on a laptop against sample `.rrd`.
Repo: https://github.com/nro-bot/lama-hack
Demos: https://photos.app.goo.gl/G9X8pUpRRjd1vedU9 · https://docs.google.com/presentation/d/1L19RGcJZgJ3UnwINjAbeo7gdjcYsdeQwwmN35AAgn9I/edit
Entered: Rerun Viewer, Rerun Query API

---

### 🥈 G1 Goalie — Autonomous goalkeeper (Unitree G1) that picks and throws a soccer ball.

A real Unitree G1 picks a soccer ball off a table with an ACT policy fine-tuned on 121 teleop episodes, then runs a hardcoded FSM: turn 180°, watch for a person via YOLO on a depth camera, shuffle side-to-side while they're there, throw once clear. Inference runs directly over DDS on the robot's dev PC (subscribe `rt/lowstate`, publish `rt/arm_sdk`) — no sim, no extra robot-side server. A separate Rerun pipeline streams live joint/camera/skeleton telemetry and repackages the episodes into recordings + a LeRobot v3 export.

**You get:**

- ACT-on-G1 direct-DDS inference loop: subscribes `rt/lowstate`, publishes `rt/arm_sdk`, no extra robot-side service (`local-vla-inference/`)
- Live Rerun telemetry with a 3D FK arm skeleton + camera feed + measured/target/commanded joint traces (`local-vla-inference/telemetry.py`, `scripted-behavior/g1_arm_fk.py`)
- Teleop→Rerun→LeRobot round trip via Rerun's dataframe Query API (`data-pipeline/`)
- Hardcoded goalkeeper FSM: YOLO human avoidance + a throw primitive self-verifying against real G1 URDF joint limits at import, 9-test suite (`scripted-behavior/`)
- Modal fine-tuning launcher over vendored HF LeRobot — CLI for ACT/GR00T/MolmoAct2 (`training/main.py`)

**Stack:** Python uv workspace; vendored HF LeRobot; ACT checkpoints on HF Hub (`ajkoder/g1-pickup-ball-act`); Unitree unitree_sdk2py + CycloneDDS; ultralytics YOLO; Modal for fine-tuning; Rerun SDK (live logging + dataframe Query API)
**Readiness:** runnable-with-setup — thorough install/run docs (HANDOVER.md, how_to_run.md, AGENTS.md, per-stage READMEs), but the demo is hard-wired to one physical rig (G1 + teleimager at fixed LAN IPs). Some module READMEs lag the code (`scripted-behavior/README.md` calls turn/avoid/shuffle "not yet built" though they exist; `data-pipeline/README.md` flags its scripts "not run end to end").
Repo: https://github.com/arjuncoder1/soccerbot
Demos: https://www.youtube.com/watch?v=B6rOcKDwPHQ
Entered: Surprise us, Rerun Viewer

---

### 🏆 Cornerf — An autonomous desk attendant that sees clutter and puts it away.

A closed-loop desk-cleaner on a reBot B601-DM arm (7-DOF, Damiao CAN, non-SO-101): an overhead-camera VLM (or CV fallback) plans pick targets, an IK/calibration layer turns pixels into arm poses, and the arm runs home→photo→plan→pick→drop with retries. Separately, a full Rerun pipeline for this non-standard robot — record teleop to `.rrd`, filter with the Query API (goal-vs-position error, success/fail labels), export to LeRobot v3, replay — then fine-tune MolmoAct 2 (LoRA) on Modal against a scripted baseline.

**You get:**

- Rerun Query API quality gate for non-SO-101 robots (`p5_rerun_port/challenge/`) scoring 100+ HF-hosted episodes into a checksummed manifest + HTML/MD report
- Rerun Blueprint: synchronized front/side cameras, a 3D URDF view, a goal-vs-position TimeSeriesView (`p5_rerun_port/viewer_blueprint.py`)
- record/log/export/replay CLI porting the so100-hackathon Rerun loop to a 7-joint Damiao-CAN arm (`p5_rerun_port/`)
- Vendored LeRobot packages for the Seeed reBot B601 follower + reBot-102 leader — a working non-SO-101 robot/teleoperator pair
- ArUco pixel-to-arm calibration chain with a ≤10min recalibration runbook (`p2_vision_calibration/`)
- VLM closed-loop pick/place orchestrator with a CV-only fallback and a pre-staged canned run for demo safety (`p3_vlm_orchestrator/`)
- MolmoAct 2 LoRA fine-tune + scripted-vs-learned bake-off on Modal (`p5_training/modal_finetune.py`, `bakeoff.py`)
- macOS operator GUI for two-camera teleop, recording, dataset review (`rebot_operator_kit/`)

**Stack:** Python 3.10+, LeRobot (vendored Seeed reBot fork), Rerun SDK (Query API + Blueprint + `rr.server.Server`), MolmoAct 2 (LoRA on Modal), OpenCV/ArUco, reBotArm_control_py SDK (Damiao CAN), HF Hub
**Readiness:** reference-code — complete and unusually well-documented (per-track docs, recalibration runbook, Query API design doc) with real HF-hosted data and passing tests, but the install path is hard-blocked on physical reBot B601-DM + 102-leader hardware, specific USB/CAN ports, and a Google-deck-hosted SDK repo.
Repo: https://github.com/aarochu/DeskPartner
Demos: https://mint-thread-32418156.figma.site · https://docs.google.com/presentation/d/1NN-bJ-LfKU94xZ4SmXe-Ko7Bcp5jQ6J_aqam3j9i4PI/edit
Entered: Non-SO-101 port, Rerun Query API, Rerun Viewer, Surprise us

---

### 🏆 ODIEV3 — Odie is your replacement pet: companionship without the care.

A 12-DOF quadruped (Pi 5 + Servo2040 on MicroPython, Hailo-8L NPU for YOLOv8) that walks, turns, sits, waves, dances, tracks an orange ball via OpenCV HSV, greets people via YOLOv8, and streams live joint/IMU/URDF telemetry to a browser over Rerun. **Critical caveat:** the repo has zero source files — it's eight markdown documents _describing_ a codebase (main.py, odie_rerun.py, ball_detect_cv.py, etc.) that isn't checked in at any commit or branch.

**You get (design-doc prose only, no implementation in the tree):**

- Serial protocol between Pi and Servo2040: command set (stand/walk/left/right/sit/wave/dance/reset90), ball-tracking commands (`track,<cx>,<cy>` / `noball`), telemetry line format
- Rerun telemetry schema: entity paths for 12 joint angles, IMU pitch/roll, fall flag, ball position, behavior log, URDF model over gRPC
- Sim-to-real gait method: diagonal-trot phase logic (FL+RR / FR+RL, sine-driven hip/knee offsets) + servo-deg→MuJoCo-rad conversion (`deg_to_ctrl`)
- HSV ball-tracking pipeline (dual red/orange ranges, erode/dilate, largest-contour circle fit, 20Hz serial output) + a Flask dashboard
- IMU fall-detection (pitch/roll ~50Hz, >45° → forced stand) and PS3 button-to-gait mapping

**Stack:** Raspberry Pi 5 + Hailo-8L NPU (YOLOv8s), Pimoroni Servo2040 (MicroPython), MuJoCo/MJCF, Rerun SDK, OpenCV, pygame, Flask — described as Python throughout, but no `.py`/`.xml`/`.stl` files exist, only prose about them.
**Readiness:** reference-code (generous) — the repo has no code at all (verified via GitHub tree, contents endpoint, and raw-file fetch on the sole `main` branch); 8 markdown files describe scripts and a MuJoCo model, none checked in. Nothing to clone or run — only ideas to take.
Repo: https://github.com/ML-bot-2012/ODIE-V3
Entered: Surprise us, Solve a puzzle

---

## Inaccessible at harvest time

Repos that 404'd or were private when this list was built. Fix your visibility and open a PR to move your entry up.

- **YCSF** — `github.com/ycsf955/ycsf-cloth-folding` 404s on the web UI, REST API, and GraphQL (authenticated); the user `ycsf955` doesn't exist either, so it's not a permissions problem — the URL maps to no real account or repo. Pitch: a voice-driven bimanual robot that folds laundry and talks you through it. Nothing to verify without a repo. The Arduino prize stands — open a PR with a working link and we'll add the full entry.

---

## Add or fix your entry

PRs welcome. One entry per team, in the `## All projects` section, ordered by readiness grade (clone-and-play → runnable-with-setup → reference-code). Match the card format:

```
### Team — one-line pitch

Two or three sentences on what the repo actually is — the shipped code, not the writeup's claims.

**You get:**
- Reusable piece, with the file path in backticks
- ...

**Stack:** one line
**Readiness:** grade — one honest sentence on what it takes to run
Repo: https://github.com/...
Demos: link · link   (omit if none)
Entered: Track, Track
```

Readiness grades: **clone-and-play** (complete, one documented path to run), **runnable-with-setup** (real run path, gated on hardware/infra/keys), **reference-code** (read it for ideas; won't run standalone). Be honest about the gap between pitch and code — that's the whole point of the list.
