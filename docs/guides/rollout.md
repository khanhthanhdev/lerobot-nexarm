# Rolling out a trained policy

How to run a trained policy on the real NexArm follower and in simulation, record what it does, and measure its success rate. This page owns rollout; the [pipeline guide](pipeline.md) covers recording and training before it. Run every command from the repository root with `uv run`.

## Before you start

- **A checkpoint folder.** `lerobot-train` writes `outputs/train/<run>/checkpoints/<step>/pretrained_model/` with `config.json`, `model.safetensors` and the pre/post-processor files. `last` points to the latest step. TurboVLA also writes `pretrained_model_ema/`; use that one. A Hub repo id such as `<user>/nexarm_act` works too.
- **The cameras the policy was trained on.** The checkpoint records its image inputs (`front`, `wrist`, `top`) and `rollout.py` exits at startup when the opened cameras differ. Use `--no-top-cam` for a checkpoint trained on front and wrist only, such as one trained on `thanhkt/nexarm_stack_bowls`. Find camera indexes and ports as in [Running NexArm](../running.md#2-find-the-serial-ports).
- **The training prompt.** Language-conditioned policies (TurboVLA, SmolVLA, pi0) need `--task` identical to the prompt they were trained with. ACT ignores it.
- **The same scene and settings as the training data.** Same camera positions, lighting, object start area, and follower `motion_speed` / `motion_acc`. A policy trained only on simulated data also needs the [calibration checks](calibration.md#step-10-run-on-the-real-arm-and-iterate).
- **Power and space.** The follower on 12V, nothing in reach of the arm, and a hand near the power switch the first time a new checkpoint moves it. Only the follower is needed; the leader can stay unplugged (except for DAgger).

## Run it on the real arm

```bash
uv run python examples/nexarm/rollout.py \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 --wrist-cam 1 \
    --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model
```

On Windows pass your COM port, for example `--follower-port COM19`, and write the command on one line. A Rerun window shows the cameras, joint states and actions. The policy runs until you press `Ctrl+C`; the arm then moves smoothly back to the pose it had at startup, and the follower's torque turns off when it disconnects, so start from a resting pose or support the arm when you stop.

A TurboVLA checkpoint with its prompt:

```bash
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

`rollout.py` flags:

| Flag                                                             | Default        | Use                                                                                                   |
| ---------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| `--follower-port`                                                | `/dev/ttyUSB1` | Follower serial port                                                                                  |
| `--leader-port`                                                  | `/dev/ttyUSB0` | Leader serial port; connected only with `--strategy dagger`                                           |
| `--policy-path`                                                  | required       | Checkpoint folder or Hub repo id                                                                      |
| `--front-cam`, `--wrist-cam`, `--front-fourcc`, `--wrist-fourcc` | `0`, `1`, auto | Camera index or device path, pixel format (`MJPG` when frames drop)                                   |
| `--top-cam`, `--no-top-cam`                                      | auto-detect    | RealSense serial when several are connected; disable for 2-camera checkpoints                         |
| `--task`                                                         | empty          | Prompt for language-conditioned policies                                                              |
| `--duration`                                                     | `0` (no limit) | Stop after this many seconds                                                                          |
| `--fps`                                                          | `30`           | Control loop and camera rate; keep the dataset's rate                                                 |
| `--strategy`, `--repo-id`                                        | `base`         | Record the rollout as a dataset; see [Strategies](#strategies)                                        |
| `--inference-type`                                               | `sync`         | `rtc` for real-time chunking with policies that support it; others fail at startup with a clear error |
| `--rerun-save-path`                                              | none           | Save the session to an `.rrd` file you can replay with `uv run rerun <file>`                          |

### Pick the checkpoint

- Start with `checkpoints/last`, then compare earlier steps (`checkpoints/050000/...`) on the same protocol; a later step is not always better on the real arm.
- Checkpoints from the former `train_turbovla.py` need a one-time conversion: [TurboVLA, section 4](turbovla.md#checkpoints-from-the-former-train_turbovlapy).
- A checkpoint on another machine: copy the whole `pretrained_model` folder, or push it to the Hub and pass its repo id.

### Models already on the Hub: TurboVLA, B-spline Policy, Plan-Bind

[`examples/nexarm/infer_hub_policy.py`](../../examples/nexarm/infer_hub_policy.py) runs the published stack-bowls models by name. The first time, it downloads only the chosen checkpoint into a plain folder, `outputs/models/stack_bowls/<preset>` (`--models-dir` moves it, `--download-only` stops there), converting the TurboVLA checkpoint into a LeRobot policy; then it runs the model from that folder in the MuJoCo stack-bowls env (default) or on the arm. On the collection rig, follow [Rolling out the stack-bowls models](stack-bowls-rollout.md) step by step:

| Preset     | Hub repo                                          | Model                                                                                                                         | Trained on                                                                        |
| ---------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `turbovla` | `thanhkt/nexarm_stack_bowls_turbovla`             | TurboVLA, `final_ema_pytorch_model.pt` (former `train_turbovla.py` format)                                                    | Stack bowls, prompt-conditioned, 3 cameras (the repo does not record its dataset) |
| `bspline`  | `thanhkt/bspline-nexarm-stack-bowls-single-order` | B-spline Policy (diffusion), 100k steps                                                                                       | Sim, red-blue-black only, 3 cameras                                               |
| `planbind` | `thanhkt/plan-bind-d1`                            | B-spline Plan-Bind, `train_planbind/checkpoints/060000` ([experiment](../../experiments/261027-plan-bind-two-step/README.md)) | Sim, red-blue-black only, 3 cameras                                               |

```bash
# In simulation: success rate and per-stage rates over 20 episodes, videos in outputs/eval/hub_<preset>
MUJOCO_GL=egl uv run python examples/nexarm/infer_hub_policy.py planbind
MUJOCO_GL=egl uv run python examples/nexarm/infer_hub_policy.py turbovla --n-episodes 50

# On the real arm: every other flag goes to rollout.py
uv run python examples/nexarm/infer_hub_policy.py bspline --target real --follower-port /dev/ttyUSB1
uv run python examples/nexarm/infer_hub_policy.py turbovla --target real --follower-port /dev/ttyUSB1 \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."

# Other checkpoints of the same repos
uv run python examples/nexarm/infer_hub_policy.py planbind --list
uv run python examples/nexarm/infer_hub_policy.py planbind --subfolder train_onestep/checkpoints/060000/pretrained_model
uv run python examples/nexarm/infer_hub_policy.py planbind --repo-id thanhkt/plan-bind-d1b \
    --subfolder train_planbind_s1001/checkpoints/060000/pretrained_model
uv run python examples/nexarm/infer_hub_policy.py turbovla --checkpoint-file steps_50000_ema_pytorch_model.pt
```

- **Sim flags.** The single-order presets evaluate with `--env.stack_order=red,blue,black`, the order they learned; change it with `--stack-order`. `--policy.device` follows whether a GPU is available. Other flags go to `lerobot-nexarm-eval`, for example Plan-Bind's latency study options `--policy.planbind_latency=6 --policy.planbind_commit=latency`.
- **Real-arm flags.** Ports, cameras, `--strategy`, `--duration` and the rest are the `rollout.py` flags above. TurboVLA needs `--task` with its training prompt on the arm; in sim the env supplies it. B-spline and Plan-Bind take no prompt, so they always stack in the order they learned.
- **Install.** TurboVLA needs `--extra turbovla` in your `uv sync` line, and its first run downloads the DINOv3 and BERT backbones (DINOv3 is gated: accept its licence on the Hub first). On a CPU-only machine each simulated episode takes several minutes.
- **Sim-trained models on the real arm.** B-spline and Plan-Bind were trained only on simulated data, and TurboVLA's three-camera setup matches the simulated datasets too. Calibrate the sim to your arm and check the cameras first ([Calibration, Step 10](calibration.md#step-10-run-on-the-real-arm-and-iterate)); expect a lower real success rate than in sim.

## Strategies

The strategy decides whether the rollout is recorded and how a person can step in.

| Strategy    | Records                                                                                                                           | Needs                                            | Use it to                                   |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------- |
| `base`      | Nothing                                                                                                                           | —                                                | Watch the policy, first runs, demos         |
| `sentry`    | Everything, split into episodes by video size                                                                                     | `--repo-id`                                      | Keep evaluation runs as a dataset           |
| `highlight` | Only what you save: `s` flushes the last 10 s and keeps recording until `s` again, `h` uploads                                    | `--repo-id`                                      | Keep the interesting moments of a long run  |
| `episodic`  | Episodes of `episode_time_s` with `reset_time_s` resets in between, like `lerobot-record` (→ ends early, ← re-records, Esc stops) | `lerobot-rollout` with `--dataset.*`             | Fixed-length evaluation trials              |
| `dagger`    | Your corrections (and with `record_autonomous=true` the policy's own frames too)                                                  | `--repo-id` and the leader arm (`--leader-port`) | Collect corrections to fine-tune the policy |

Name evaluation datasets `eval_*`. With `rollout.py`, `sentry` and `highlight` upload to the Hub every few episodes, so log in first (`uv run hf auth login`) and use your user name in the id:

```bash
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --strategy sentry --repo-id "${HF_USER}/eval_nexarm_act" \
    --task "Pick up the red cube and place it into the green zone" --duration 120
```

To keep an evaluation dataset local, or for `episodic`, call `lerobot-rollout` directly as below.

### DAgger: correct the policy with the leader arm

Plug in both arms. The policy drives the follower while the leader follows it with torque on; when the policy goes wrong, take over with the leader and your correction is recorded. Keys: `space` pauses or resumes the policy, `tab` starts and ends a correction, `enter` uploads. By default only the corrections are recorded, each as its own episode.

```bash
uv run python examples/nexarm/rollout.py \
    --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --strategy dagger --repo-id "${HF_USER}/nexarm_dagger" \
    --task "<training prompt>"
```

Like `sentry`, this uploads the dataset to the Hub; for a local-only dataset use the `lerobot-rollout` DAgger command below with `--dataset.push_to_hub=false`. Merge the correction dataset with the original training data and fine-tune; see [Training tips](training-tips.md).

## Run it with `lerobot-rollout`

`rollout.py` builds this command; call it yourself for options the script does not expose. `lerobot-nexarm-rollout` is the same command plus `--rerun_save_path`.

```bash
uv run lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --task="Stack the bowls with red on bottom, blue in middle, and black on top." \
    --display_data=true
```

Remove the `top` entry for a 2-camera checkpoint. Or edit the ports and cameras once in [`examples/nexarm/inference.yaml`](../../examples/nexarm/inference.yaml) (a `sentry` rollout to a local dataset) and run:

```bash
uv run lerobot-rollout --config_path=examples/nexarm/inference.yaml \
    --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model
```

Useful options:

| Option                                                                         | Effect                                                                             |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `--dataset.repo_id`, `--dataset.single_task`, `--dataset.push_to_hub=false`    | Where a recording strategy writes, its task, and keeping it local                  |
| `--dataset.num_episodes`, `--dataset.episode_time_s`, `--dataset.reset_time_s` | Trial count and lengths for `episodic`                                             |
| `--interpolation_multiplier=2`                                                 | Send interpolated commands between policy actions at `fps × 2` for smoother motion |
| `--return_to_initial_position=false`                                           | Leave the arm where it stopped instead of moving it back to the startup pose       |
| `--duration=120`                                                               | Stop after 120 s (`0` runs until `Ctrl+C`)                                         |
| `--inference.type=rtc`                                                         | Real-time chunking, for policies that support it                                   |
| `--device=cpu`                                                                 | Run without a GPU (slower; check the control rate stays near 30 Hz)                |
| `--play_sounds=false`                                                          | Skip spoken event messages, which can hang on machines without a speech engine     |
| `--display_data=false`                                                         | No Rerun stream, which frees CPU on a slow machine                                 |

DAgger with `lerobot-rollout`, keeping the dataset local (add `--strategy.record_autonomous=true` to record the policy's own frames too):

```bash
uv run lerobot-rollout \
    --strategy.type=dagger \
    --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --robot.cameras='<same JSON as above>' \
    --teleop.type=nexarm_leader --teleop.port=/dev/ttyUSB0 \
    --dataset.repo_id="${HF_USER}/nexarm_dagger" \
    --dataset.single_task="<training prompt>" \
    --dataset.push_to_hub=false
```

## Run it in simulation

Check a checkpoint in MuJoCo before it touches the arm. `lerobot-eval` runs seeded episodes and reports the success rate (`pc_success`); `lerobot-nexarm-eval` takes the same flags and adds how far failed episodes got (`pc_picked_middle`, `pc_nested_middle`, ...):

```bash
MUJOCO_GL=egl uv run lerobot-nexarm-eval \
    --policy.path=outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20 \
    --output_dir=outputs/eval/nexarm_turbovla
```

The stack-bowls env gives each episode the prompt of its sampled bowl order, so no task is passed. Up to 10 episode videos are saved under `--output_dir`. Use `--env.task=NexArmPickPlace-v0` for pick-and-place, add `--env.calibration_path` / `--env.joint_mapping_path` for a [calibrated sim](calibration.md#calibration-files-and-where-they-are-used), and compare several pick-and-place checkpoints with `lerobot-nexarm-sim-benchmark` ([pipeline, section 4](pipeline.md#4-benchmark)).

To drive the simulated follower through the same robot interface as the real one (no success metric, a Rerun view of the episode):

```bash
uv run lerobot-nexarm-rollout --strategy.type=base \
    --policy.path=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
    --robot.type=nexarm_sim --robot.model_path=sim/description/mjcf/scene.xml \
    --task="Pick up the red cube" --duration=60 --display_data=true \
    --rerun_save_path=outputs/rollouts/act_nexarm_sim.rrd
```

More simulated rollout options: [`sim/README.md`, section 9](../../sim/README.md#9-run-a-policy-in-mujoco).

## Measure the success rate on the real arm

Simulated success rates do not predict real ones. On the real arm:

1. Write down the trial protocol before you start: 10–20 trials, the start layout of each, the success criterion, and a time limit per trial.
2. Run every checkpoint you compare on the same protocol, ideally as a `sentry` or `episodic` rollout so each trial is saved.
3. Report successes out of trials and note how each failure happened (missed grasp, dropped object, wrong order, timeout, collision).
4. Inspect saved trials with `uv run lerobot-dataset-viz --repo-id <eval_repo_id> --episode-index 0`, or replay an `.rrd` with `uv run rerun <file>`.

What to expect and how to read the result: [Training tips, section 4](training-tips.md#4-evaluation--benchmarks). For policies trained in simulation, [Calibration, Step 10](calibration.md#step-10-run-on-the-real-arm-and-iterate) maps each kind of failure to the calibration step that fixes it.

## Troubleshooting

| Symptom                                            | Fix                                                                                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `Rollout setup failed: ...` naming cameras         | The opened cameras differ from the checkpoint's. Add or remove `--no-top-cam`, or fix `--front-cam` / `--wrist-cam`       |
| The arm does not move, or the loop runs at 5 Hz    | Follower not powered or not in bridge mode: check the 12V supply and restart                                              |
| The arm moves, but aimlessly or to the wrong place | Wrong camera index (front and wrist swapped), a moved camera, a different scene, or a `--task` that differs from training |
| Jerky motion                                       | Try `--interpolation_multiplier=2`; check the logged control rate stays near 30 Hz                                        |
| Control rate well below 30 Hz                      | Slow inference: run on a GPU, use `--display_data=false`, and add `--front-fourcc MJPG` if camera frames drop             |
| `--strategy sentry requires --repo-id`             | Recording strategies need a dataset id                                                                                    |
| `dagger strategy requires --teleop.type to be set` | `lerobot-rollout` needs `--teleop.type=nexarm_leader --teleop.port=...`; `rollout.py` adds them from `--leader-port`      |
| Upload fails during `sentry`                       | Log in with `uv run hf auth login`, or keep the dataset local with `lerobot-rollout --dataset.push_to_hub=false`          |
| Freeze on exit                                     | Add `--play_sounds=false`                                                                                                 |
| TurboVLA: `No module named 'turbovla'`             | Add `--extra turbovla` to your `uv sync` line                                                                             |

Other hardware problems: [Troubleshooting](../reference/troubleshooting.md).
