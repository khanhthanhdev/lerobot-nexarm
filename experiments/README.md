# Experiments

One folder per experiment, so every result can be traced to its config, data and checkpoint.

## Index

Each folder's README holds the status, the commands and the results; update this table when an experiment's status changes. Open work across experiments and plans is tracked in [`plans/README.md`](../plans/README.md).

| Experiment                                                                                   | Question                                                                | Status                                                       |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| [261024-turbovla-nexarm-parity](261024-turbovla-nexarm-parity/README.md)                     | Does TurboVLA through `lerobot-train` match the former training script? | 2k-step parity done; full run, sim eval and 2-GPU check open |
| [261024-turbovla-libero](261024-turbovla-libero/README.md)                                   | TurboVLA on the four LIBERO suites                                      | Pending (Linux GPU with EGL)                                 |
| [261024-bspline-vs-diffusion-stack-bowls](261024-bspline-vs-diffusion-stack-bowls/README.md) | B-spline Policy against diffusion on stack bowls, sim and real arm      | Pending (Linux GPU with EGL, real arm)                       |
| [261026-dem-stack-bowls-single-order](261026-dem-stack-bowls-single-order/README.md)         | Does DEM stack bowls when the order is fixed?                           | Run: 25% success (EMA, 20 episodes)                          |
| [261027-bspline-stack-bowls-single-order](261027-bspline-stack-bowls-single-order/README.md) | B-spline Policy as the non-language baseline for the DEM run            | Results table empty                                          |
| [261027-plan-bind-two-step](261027-plan-bind-two-step/README.md)                             | Plan-Bind D1: does it beat the one-step control under latency?          | Run once: ahead in all 8 cells                               |
| [261028-plan-bind-d2-latency](261028-plan-bind-d2-latency/README.md)                         | Plan-Bind D2: measured inference latency and update rate                | Run                                                          |
| [261028-plan-bind-d3-pusht](261028-plan-bind-d3-pusht/README.md)                             | Plan-Bind D3: does the lead hold on Push-T's human demonstrations?      | Ready, not run                                               |
| [261029-plan-bind-d1b-seeds](261029-plan-bind-d1b-seeds/README.md)                           | Plan-Bind D1b: D1 repeated with 3 seeds and a position-bearing control  | Run                                                          |

## Convention

- Name each folder `YYMMDD-<slug>`, for example `261024-turbovla-nexarm-parity`. Start by copying `_template/`.
- Each folder holds a `README.md` (hypothesis, data, command, checkpoint, results) and the draccus YAML configs it ran. Pass a config to the LeRobot CLIs with `--config_path`, and override single fields on the command line:

  ```bash
  uv run lerobot-train --config_path experiments/261024-my-run/train.yaml --steps=2000
  ```

- Record results with the number of episodes (at least 50 in sim and at least 10 on the real arm), the success rate, and the checkpoint's Hub id.
- Pin the dataset: write down the `repo_id` and the revision (commit hash or tag) you trained on.
- Training outputs stay out of git: set `output_dir` to `experiments/<folder>/outputs/...` or `outputs/...`, which are both ignored. Log runs to W&B with `--wandb.enable=true` and link the run in the README.
