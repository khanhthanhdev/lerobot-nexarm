# NexArm test baseline before the reorganization

Recorded on 2026-10-24 at commit `c96f8b20` (tag `pre-reorg`). Later phases compare against this.

## Windows (this box, Windows Server 2022, CPU)

| Command                                                                 | Result                            |
| ----------------------------------------------------------------------- | --------------------------------- |
| `uv run pytest tests -k nexarm -q --ignore=tests/utils/test_process.py` | 338 passed, 10 failed, 28 skipped |
| `uv run pytest tests/nexarm_service -q`                                 | 128 passed                        |
| `make test-nexarm-ete`                                                  | not run: needs MuJoCo rendering   |

Without `--ignore`, collection fails: `tests/utils/test_process.py` uses `signal.SIGHUP`, which Windows does not have.

All 10 failures are environment problems (MuJoCo has no OpenGL context on this box, and `nexarm_scene.json` is checked out with CRLF line endings):

- `tests/envs/test_nexarm_env.py::test_nexarm_env_gym_checker[pixels_agent_pos]`
- `tests/envs/test_nexarm_env.py::test_nexarm_env_obs_types[pixels_agent_pos]`
- `tests/envs/test_nexarm_env.py::test_nexarm_env_obs_types[pixels]`
- `tests/envs/test_nexarm_env.py::test_nexarm_env_render_rgb`
- `tests/envs/test_nexarm_sim_gym_integration.py::test_nexarm_sim_gym_raw_position_contract`
- `tests/envs/test_nexarm_sim_gym_integration.py::test_nexarm_train_eval_integration`
- `tests/robots/test_nexarm_isaac_assets.py::test_isaac_assets_are_in_sync_with_the_mjcf`
- `tests/robots/test_nexarm_sim.py::test_backend_steps_and_renders`
- `tests/robots/test_nexarm_sim.py::test_sim_robot_matches_physical_feature_contract`
- `tests/scripts/test_generate_nexarm_sim_dataset.py::test_generator_writes_dataset_report_and_episode_log`

## Linux GPU server

Recorded on 2026-10-24 on `marinelab` (Ubuntu 24.04, RTX 5050 8 GB, `MUJOCO_GL=egl`), in fresh worktrees synced with `uv sync --locked --extra test --extra dev --extra viz --extra service --extra training` (the branch adds `--extra turbovla`). Without `--extra training`, `test_nexarm_train_eval_integration` fails because `accelerate` is missing.

| Command                                                                                      | `pre-reorg` (`c96f8b20`) | `chore/upstream-sync` (`86d7f4b5`) |
| -------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------- |
| `uv run pytest tests -k nexarm -q`                                                           | 348 passed, 26 skipped   | 348 passed, 26 skipped             |
| `uv run pytest tests/nexarm_service -q`                                                      | 128 passed               | 128 passed                         |
| `uv run pytest tests -k nexarm -q` at `255d29e5` (adds the 3 `_max_episode_steps` env tests) | n/a                      | 351 passed, 26 skipped             |
| `make test-nexarm-ete`                                                                       | 3 passed                 | 3 passed                           |

On the branch, `uv run pytest policies packages tests/utils/test_rerun_save_path.py tests/scripts/test_record_collection.py` gives 26 passed. With `TURBOVLA_PARITY_DATASET_ROOT` pointing at a full download of `thanhkt/nexarm_stack_bowls@64ce8aeb`, the TurboVLA plugin tests give 14 passed; the 2 `[trained]` inference-parity cases do not apply to the only trained checkpoint on the server, which uses three cameras while that dataset has two.
