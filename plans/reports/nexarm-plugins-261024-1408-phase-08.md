# NexArm plugins: steps 1-3 of phase 08

Date: 2026-10-24. Branch `chore/upstream-sync`, starting from `1466ecdb3`. The work is committed but not pushed. Scope follows the user's decision: the NexArm code moves into `packages/`, and `src/lerobot/` stays with the generic patches. Details are in the phase file's [execution notes](../261024-0016-research-repo-reorganization/phase-08-nexarm-plugins-on-upstream-lerobot.md#execution-notes-2026-10-24).

## Commits

| Commit      | What it does                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `3cc39d267` | moves `nexarm_leader` into `lerobot_teleoperator_nexarm`                                                                                    |
| `eb5db6c8d` | moves the motor driver, the three robots and the collection tools into `lerobot_robot_nexarm`                                               |
| `902cec292` | moves `NexArmEnv`, the gym envs and the sim benchmark into `lerobot_env_nexarm`                                                             |
| `4749ca05a` | moves the desktop-app engine into `nexarm_service`, and the CLI now registers plugins before it serves                                      |
| `3d3c2b8af` | adds the packages to mypy, interrogate and the CI path filters, fixes the Docker copy step, updates the AGENTS.md table and the README tree |

## Result

- `src/lerobot/` contains no NexArm code: `git grep -i nexarm -- src/lerobot` returns nothing. What remains against `upstream/main` is 13 files, +190/-23, all of them the carried generic patches.
- The registered names are unchanged: `nexarm_follower`, `nexarm_sim`, `mobile_bi_nexarm_sim`, `nexarm_leader` and the env `nexarm`. Datasets and checkpoints store only these names, so no compatibility shim was needed.
- Outside `plans/`, `git grep` finds no old module or file path.
- Tests stay in `tests/`, and the test commands are the same as before.

## Tests (Windows, CPU)

| Command                                                                 | Before                           | After                                              |
| ----------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------- |
| `uv run pytest tests -k nexarm -q --ignore=tests/utils/test_process.py` | 338 passed, 10 failed, 5 skipped | same; the same 353 test IDs with the same outcomes |
| `uv run pytest tests/nexarm_service -q`                                 | 128 passed                       | 128 passed                                         |
| `uv run pytest policies -q`                                             | 28 passed, 2 skipped             | 28 passed, 2 skipped                               |

The CLIs work: `lerobot-record --help` lists the four NexArm types, `lerobot-eval --env.type=nexarm --help` resolves the env, and `lerobot-nexarm-service` runs and dumps schemas that match `docs/service/*.json`. An ACT checkpoint trained with `--env.type=nexarm` before the move still loads, and `thanhkt/nexarm_stack_bowls` still loads. The checkpoint gives the same action as before the move. Pre-commit is clean apart from the 8 known Windows `termios` mypy errors.

## Open items

- `lerobot-find-joint-limits` and the `rl/*` HIL-SERL entry points have no plugin hook, as in upstream. NexArm types therefore no longer parse there, although nothing in the NexArm docs or examples uses them.
- Not run: the Linux test runs, `make test-nexarm-ete`, the Docker builds, and the desktop app against the real arm.
- To remove `src/lerobot/`, each of the 13 patched files needs a merged upstream PR or a fallback. After that, pin upstream `lerobot`, move the NexArm tests into the packages, and document the upgrade step.
