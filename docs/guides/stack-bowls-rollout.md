# Rolling out the stack-bowls models on the rig

Step by step: download the published stack-bowls models once, check them in simulation, set up the rig, and run them on the real arm. It is written for the collection rig (`marinelab`: Logitech C270 front camera, icSpring wrist camera, RealSense D435i top camera). Every rollout option is explained in [Rolling out a trained policy](rollout.md); this page is the short path. Run every command from the repository root.

| Name       | Model                                                                          | Prompt                                   |
| ---------- | ------------------------------------------------------------------------------ | ---------------------------------------- |
| `planbind` | B-spline Plan-Bind, `thanhkt/plan-bind-d1`, `train_planbind` step 60000        | None; always red bottom, blue, black top |
| `bspline`  | B-spline Policy, `thanhkt/bspline-nexarm-stack-bowls-single-order`, 100k steps | None; always red bottom, blue, black top |
| `turbovla` | TurboVLA, `thanhkt/nexarm_stack_bowls_turbovla`, final EMA weights             | Follows the prompt (`--task`)            |

All three use the front, wrist and top cameras. Plan-Bind and B-spline were trained only on simulated data, so **expect a lower success rate on the real arm than in simulation**; [Calibration](calibration.md) is how to close that gap.

## 1. Update and install

```bash
git pull
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense --extra turbovla
```

## 2. Download the weights (once)

```bash
for p in planbind bspline turbovla; do
    HF_HUB_DISABLE_XET=1 uv run python examples/nexarm/infer_hub_policy.py "$p" --download-only
done
```

Each model becomes a plain policy folder (about 1.2 GB each; the download resumes if interrupted):

```text
outputs/models/stack_bowls/
├── planbind/    config.json, model.safetensors, policy_*processor*
├── bspline/
├── turbovla/    converted from the former train_turbovla.py checkpoint
└── .hub/        raw downloads; delete once the three folders exist
```

After that the rollouts need no network. `HF_HUB_DISABLE_XET=1` switches to plain HTTP downloads; on `marinelab` the default Xet transfer stalled. TurboVLA's DINOv3 and BERT backbones load from the Hugging Face cache, where they already are on `marinelab`. To keep another checkpoint of a repo, add `--subfolder` or `--checkpoint-file` ([`--list` shows them](rollout.md#models-already-on-the-hub-turbovla-b-spline-policy-plan-bind)); it gets its own folder.

## 3. Check them in simulation (optional, GPU)

```bash
MUJOCO_GL=egl uv run python examples/nexarm/infer_hub_policy.py planbind --n-episodes 20
```

It prints the success rate (`pc_success`) and how far failed episodes got, and saves videos in `outputs/eval/hub_planbind`. A model that fails here will not work on the arm.

## 4. Set up the rig

1. **Power and USB.** Follower on its 12V supply, follower USB-C plugged in. The leader is not needed. Check the port with `ls /dev/ttyUSB*` (or `uv run lerobot-find-port`); the launcher uses `/dev/ttyUSB1`, so pass `--follower-port /dev/ttyUSB0` if the follower is the only arm plugged in.
2. **Cameras.** `./scripts/nexarm/collect.sh --test-cameras` checks all three cameras together and saves snapshots under `outputs/collection/<timestamp>/`. Compare them with the ones from your collection sessions: a moved camera is the most common cause of a policy that reaches to the wrong place.
3. **Bowls.** Put the three bowls about where the simulation starts them, measured from the base's pan axis: red 11 cm to the robot's left and 27 cm forward, blue centred 18 cm forward, black 11 cm to the robot's right and 27 cm forward. Keep the lighting as during collection.
4. **Space.** Nothing else in reach of the arm, and a hand near the power switch.

## 5. Run

```bash
./scripts/nexarm/rollout_stack_bowls.sh planbind
./scripts/nexarm/rollout_stack_bowls.sh bspline
./scripts/nexarm/rollout_stack_bowls.sh turbovla
```

[`rollout_stack_bowls.sh`](../../scripts/nexarm/rollout_stack_bowls.sh) passes the rig's camera paths and pixel formats, the follower port and the red-blue-black prompt, and downloads the model first if step 2 was skipped. A Rerun window shows the cameras and actions.

- `Ctrl+C` stops. The arm then moves back to its start pose and its torque turns off, so support it or start from a resting pose.
- Any [`rollout.py` flag](rollout.md#run-it-on-the-real-arm) overrides the defaults, for example `--follower-port /dev/ttyUSB0`, `--duration 90` (stop after 90 s) or `--rerun-save-path outputs/rerun/planbind_trial1.rrd` (replay later with `uv run rerun <file>`).
- TurboVLA follows the prompt. Another order: `TASK="Stack the bowls with blue on bottom, red in middle, and black on top." ./scripts/nexarm/rollout_stack_bowls.sh turbovla`. Plan-Bind and B-spline ignore it.
- To keep every trial as a dataset, add `--strategy sentry --repo-id <hf_user>/eval_stack_bowls_planbind`; it uploads to the Hub every few episodes, so log in first (`uv run hf auth login`).

## 6. Compare the models

Run the same protocol for each model so the results compare:

1. Write down 10 start layouts (bowl positions within a few centimetres of step 4) before you start.
2. For each model and layout, reset the bowls, run with `--duration 90`, and note success (all three stacked in the right order) or how it failed: missed grasp, dropped bowl, wrong order, timeout, collision.
3. Report successes out of 10 per model.

How to read the failures and which calibration step fixes each: [Calibration, Step 10](calibration.md#step-10-run-on-the-real-arm-and-iterate).

## Troubleshooting

| Symptom                                                   | Fix                                                                                                    |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `Rollout setup failed` naming the top camera or RealSense | The D435i is unplugged or busy; all three models need it. Close other programs using it and replug     |
| Camera not found at `/dev/v4l/by-id/...`                  | `ls /dev/v4l/by-id/`, then pass `--front-cam` / `--wrist-cam` with the path you see                    |
| `Permission denied: '/dev/ttyUSB1'`                       | Wrong port, or not in `dialout`: see [Install](../getting-started/install.md#linux-serial-permissions) |
| The arm does not move, or the loop runs at 5 Hz           | Follower not powered or not in bridge mode: check the 12V supply and restart                           |
| Jerky or slow motion                                      | Check the logged control rate stays near 30 Hz; stop other jobs on the GPU (`nvidia-smi`)              |
| Reaches to the wrong place                                | Cameras moved or bowls outside the trained area (step 4); then [Calibration](calibration.md)           |

Everything else: [Rolling out a trained policy, Troubleshooting](rollout.md#troubleshooting).
