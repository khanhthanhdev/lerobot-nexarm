# Experiments

One folder per experiment, so every result can be traced to its config, data and checkpoint.

## Convention

- Name each folder `YYMMDD-<slug>`, for example `261024-turbovla-nexarm-parity`. Start by copying `_template/`.
- Each folder holds a `README.md` (hypothesis, data, command, checkpoint, results) and the draccus YAML configs it ran. Pass a config to the LeRobot CLIs with `--config_path`, and override single fields on the command line:

  ```bash
  uv run lerobot-train --config_path experiments/261024-my-run/train.yaml --steps=2000
  ```

- Record results with the number of episodes (at least 50 in sim and at least 10 on the real arm), the success rate, and the checkpoint's Hub id.
- Pin the dataset: write down the `repo_id` and the revision (commit hash or tag) you trained on.
- Training outputs stay out of git: set `output_dir` to `experiments/<folder>/outputs/...` or `outputs/...`, which are both ignored. Log runs to W&B with `--wandb.enable=true` and link the run in the README.
