# Code review: `chore/upstream-sync` (upstream LeRobot `d3793b7e3` into the NexArm fork)

## Scope

- Merge `e7cd531e9` plus the follow-ups `2caaee217..25680fe41`. The working tree is at `02d3d980c`, one commit past that range (uv workspace); I did not review that commit.
- Focus: fork-specific code only. `examples/nexarm/record_collection.py`, `tests/scripts/test_record_collection.py`, the launchers (`record.py`, `prepare_collection.py`, `collect.sh`), the re-applied patches, the `gradient_accumulation` rename, `nexarm_service` against upstream APIs, and the registration imports.
- Contract: `plans/261024-0016-research-repo-reorganization/phase-02-upstream-sync-and-retire-core-edits.md` (disposition table and "Execution notes").

## Overall assessment

The merge itself looks correct. The wrapper keeps all five collection flags. The manual path, the merge after recording, the merge error path with its retry command, resume, and the Hub-push suppression all behave like the old `lerobot_record.py` on `main`. No doc or script passes a collection flag to plain `lerobot-record`. Every `lerobot.*` import in fork code still resolves, and `nexarm_service` matches the current upstream rollout and DAgger APIs.

There are two real problems:

1. `--rerun_save_path` writes an `.rrd` file with no logged data. This bug was already on `main`; the merge did not cause it.
2. Plain `lerobot-record` no longer prints the timestamped dataset id, but several guides still tell users to copy it. The merge caused this.

## Confirmed findings

### H1. `--rerun_save_path` saves an `.rrd` that contains no logged data. Bug was already on `main`; the merge did not cause it.

- **Where:** `src/lerobot/utils/rerun_visualization.py:67-75`.
- **Code:** `init_rerun` calls `rr.save(path)` and then `rr.connect_grpc(...)` (remote viewer) or `rr.spawn(...)` (local viewer). In rerun 0.36 each of these calls replaces the recording's sink. The later call therefore drops the file sink, and everything logged after it goes only to the viewer.
- **Evidence:** I ran a test script under `uv run --no-sync python -I` with rerun 0.36.3. It logs 200 scalars in three modes:

  | Mode | File size |
  | --- | --- |
  | `save` only | 11748 B |
  | `save` then `connect_grpc` | 3226 B (header only) |
  | `rr.set_sinks(rr.GrpcSink(...), rr.FileSink(path))` | 11732 B |

  The `spawn` path is inferred, not tested: `rr.spawn(connect=True)` calls `connect_grpc` internally.
- **Who is affected:** every `--rerun_save_path` user:
  - `examples/nexarm/record_collection.py` (non-manual path through `_rerun_saved_to`, and `_record_manual`)
  - `examples/nexarm/record.py:90`
  - `src/lerobot/scripts/lerobot_rollout.py:234-250` (documented in `sim/README.md:467-474`)
  - `examples/nexarm/rollout.py:132`
  - `examples/nexarm/teleoperate.py:67`
- **Not caused by the merge:** `main` has the same `init_rerun` code, and `main`'s lock already pinned rerun-sdk 0.36.2.
- **Failure scenario:** in the plan's real-arm smoke test, `record.py --rerun-save-path out.rrd` shows a live viewer, but the replay file is empty.
- **Why tests pass:** `tests/scripts/test_record_collection.py::test_rerun_save_path_reaches_the_visualization` only checks that the keyword argument reaches a stubbed `init_visualization`. It never checks that the file receives data.
- **Severity:** High. A feature fails silently, but no recorded dataset is lost.
- **Fix (verified above):**

  ```python
  if save_path is not None:
      Path(save_path).parent.mkdir(parents=True, exist_ok=True)
      if ip and port:
          rr.set_sinks(rr.GrpcSink(url=f"rerun+http://{ip}:{port}/proxy"), rr.FileSink(str(save_path)))
      else:
          rr.spawn(memory_limit=memory_limit, connect=False)
          rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(save_path)))
  elif ip and port: ...  # unchanged
  ```

  Add a test that logs a few scalars with `save_path` set and checks the file grows beyond the header size. Use a closed port for the gRPC sink, as in the experiment.

### M1. Plain `lerobot-record` no longer prints `Dataset repo_id:` / `Dataset root:`, but the guides still tell users to copy them. Caused by the merge.

- **What changed:** the old fork `record()` printed the final stamped id (`main:src/lerobot/scripts/lerobot_record.py:548-550`). That print now exists only in `examples/nexarm/record_collection.py:302-304`.
- **Upstream never shows the stamped id:**
  - `record()` logs `pformat(asdict(cfg))` before calling `cfg.dataset.stamp_repo_id()`, so the logged config has the unstamped id.
  - `stamp_repo_id` itself does not log.
  - Nothing in `datasets/` logs the created repo id or root.
- **Affected docs:** these show a plain `uv run lerobot-record ...` command and then say to use the printed id:
  - `AGENT_GUIDE.md:119-131`
  - `run.md:387-402` (the prose covers both `record.py` and `lerobot-record`; only `record.py` still prints)
  - `sim/README.md:343-364`
  - `docs/source/nexarm/pipeline.md:54`
  - `SERVER_TRAINING_GUIDE.md:78`
- **Failure scenario:** a user follows AGENT_GUIDE §4, records with `lerobot-record`, and sees no id. The follow-up replay, visualize and train commands need `<repo_id>_<YYYYMMDD_HHMMSS>`. The user has to guess it from `~/.cache/huggingface/lerobot/`.
- **Severity:** Medium (documented workflow broken).
- **Fix (either):**
  - Point those commands at `examples/nexarm/record_collection.py`, which accepts every `lerobot-record` flag.
  - Or rewrite the prose to say plain `lerobot-record` does not print the id. Tell users to look under `$HF_LEROBOT_HOME`, or to pass `--dataset.no_stamp=true` (upstream has `no_stamp`) to keep a fixed id.

### M2. The manual-control path (the production collection path) has no tests in the repo.

- **Why it matters:** `scripts/nexarm/collect.sh` runs `prepare_collection.py`, which always passes `--manual_control=true` (`prepare_collection.py:345`). That routes to `_record_manual` (`record_collection.py:182-258`), about 80 lines that copy upstream `record()`'s dataset opening, connection, finalize and push. Neither `tests/scripts/test_record_collection.py` nor `tests/datasets/test_collection_merge.py` sets `manual_control`. `main` had no such test either.
- **Behaviour I confirmed matches the old code:**
  - Logs are redirected to `recording_log` and `play_sounds` is forced off.
  - The mergeability check runs before any connection.
  - The connection order is teleop then robot, matching current upstream.
  - `control_time_s=inf` passes upstream's new "required" check.
  - Resume skips `stamp_repo_id`.
  - The `push_to_hub` short-circuit is equivalent to the old `and not root_repo_id`.
  - `MERGING DATA...` / `FINISHED` dashboards still show, the retry command prints on merge failure, and the exception is re-raised.
- **Scratch test:** it drives `_record_manual` with a mock robot and teleop, a fake listener and two simulated key presses. Both cases pass: without a root, and with `root_repo_id` + `merge_root`, where the root ends up with 1 episode. File: `C:/Users/Administrator/AppData/Local/Temp/claude/C--Users-Administrator-code-lerobot-nexarm/95291e05-9bdf-4792-91ad-42e0b55b54a8/scratchpad/test_manual_scratch.py`.
- **Failure scenario:** a later upstream sync changes `record_loop`, `LeRobotDataset.create/resume` or `init_keyboard_listener`. The copied manual path breaks with no failing test until someone is at the arm.
- **Severity:** Medium.
- **Fix:** move the scratch test into `tests/scripts/test_record_collection.py`.

### L1. In the non-manual path, `--root_repo_id` builds the robot twice.

- `record_collection.py:290-296` calls `make_robot_from_config` for the mergeability check, then `record()` builds its own robot.
- Harmless for `nexarm_follower`: its constructor does no I/O (`nexarm_follower.py:55-61`). The RealSense camera may do a device lookup.
- For `nexarm_sim` it loads the MuJoCo model twice.
- The check also runs before `init_logging()`, so its log lines use the default handler.
- **Fix:** acceptable as is. Optionally compute features once from a config-only path, or document it.

### L2. `examples/nexarm/record.py:19-22` header says "Enter: start / confirm next episode", but `record.py` never passes `--manual_control`.

In the timed flow Enter does nothing. This was already on `main`. **Fix:** change the text to Right/`n`, or add a `--manual` option.

### L3. Plan execution note: the rerun version jump is misstated.

The plan says "rerun-sdk 0.33.1 -> 0.36.3", but `main:uv.lock` already had 0.36.2. The actual change is 0.36.2 to 0.36.3. **Fix:** correct the line.

## Suspected / low-confidence findings

- **S1. `local_files_only` does not apply to remote object-store roots** (`lerobot_dataset.py:242-246`). When `root` is a remote URI, `_storage_root` localization runs before the metadata `local_files_only` guard, so it still uses the network. This only matters if someone combines the flag with an `s3://`-style root, which no fork caller does. Low.
- **S2. `lerobot_rollout.py:233-234` uses `getattr(cfg, "rerun_save_path", None)` on a dataclass field.** It was added for a test stub that lacks the field. It also hides a future rename of the field. Prefer updating the stub. Low.
- **S3. `_rerun_saved_to` monkeypatches `lerobot_record.init_visualization`** (`record_collection.py:113-124`). If upstream changes how `record()` reaches `init_visualization`, the `.rrd` would silently not be requested. The existing test catches this, so the risk is low.
- **Info:** upstream `record_loop` now supports multi-teleop for LeKiwi only. `main`'s `NexArmLeader` + `KeyboardTeleop` list path (for `mobile_bi_nexarm_sim`) is gone. `git grep` on `main` and HEAD finds no caller, so nothing breaks today.

## Verified OK (evidence)

1. **Wrapper flags.**
   - `draccus.parse(CollectionRecordConfig, ...)` accepts the full `DATA_COLLECTION_GUIDE.md` command plus `--recording_log`, `--resume` and `--merge_root`.
   - Upstream `RecordConfig` rejects each of `--manual_control`, `--recording_log`, `--root_repo_id` and `--merge_root`.
   - `examples/nexarm/record.yaml` parses with the wrapper.
2. **No caller passes collection flags to plain `lerobot-record`.** I checked `git grep` hits in the docs, `run.md`, `AGENT_GUIDE.md`, `DATA_COLLECTION_GUIDE.md`, `scripts/`, `src/lerobot/nexarm_service/` and `examples/`. The only users are `prepare_collection.py`, `record.py` and `DATA_COLLECTION_GUIDE.md:158`, all through `record_collection.py`. The `rerun_save_path` hits in `sim/README.md` and `examples/nexarm/rollout.py` are for `lerobot-rollout`, which has the field (`rollout/configs.py:332`).
3. **Re-applied patches** (`git diff upstream/main`):
   - The eval stage rates use upstream's `done_indices`, `ACC_KEYS` and `_append`, and `defaultdict` is imported. `test_lerobot_eval_reports_stack_bowls_success_and_stage_rates` passes.
   - `local_files_only` in metadata and dataset passes `tests/datasets/test_lerobot_dataset.py -k local_files_only`.
   - The lazy `groot` imports are kept and `test_optional_policy_imports.py` passes.
   - `import_utils.py` and `streaming_dataset.py` are identical to upstream. Upstream's `get_safe_default_video_backend` already falls back to PyAV on `ImportError`, `OSError` or `RuntimeError`.
   - `robot_type=robot.robot_type` is consistent with upstream `sanity_check_dataset_robot_compatibility`, which compares `robot.robot_type`.
4. **`gradient_accumulation_steps`.** The only remaining hits are upstream's own prose (`multi_gpu_training.mdx` formula, `lerobot_train.py` comments) and its test assertions. The fork YAML, docs and tests use `accelerator.gradient_accumulation.steps`. `test_train_stack_bowls_act.py` passes.
5. **`nexarm_service`.**
   - An AST walk over all `from lerobot... import` statements in `nexarm_service/**`, `examples/nexarm/*.py`, the `*nexarm*` modules, `collection_merge.py`, `recording_session.py` and `lerobot_nexarm_sim_benchmark.py` found that every module and name resolves.
   - These names still exist upstream with matching signatures:
     - the `RolloutConfig`, `DAggerStrategyConfig` and `DatasetRecordConfig` keyword arguments
     - `build_rollout_context(cfg, shutdown_event, ...)` and `create_strategy(config)`
     - the patched `episodic.init_keyboard_listener`, `dagger._init_dagger_keyboard(events, cfg)` and `dagger._init_dagger_pedal(events, cfg)`
     - `strategy._events`, `robot_wrapper.inner` and `supports_hold`
     - the `LeRobotDataset.create/resume` keyword arguments
   - The train-log metric parser matches upstream's `MetricsTracker.__str__` format.
   - Upstream's new stdin interactive listener is off by default (`interactive: bool = False`).
6. **Registration imports.** `nexarm_follower`, `nexarm_sim` and `mobile_bi_nexarm_sim` are imported in all six scripts: `lerobot_record`, `rollout`, `teleoperate`, `calibrate`, `replay` and `find_joint_limits`. `nexarm_leader` is imported in all of them except `replay`, which also lacked it on `main`. They are also in `rl/actor.py`, `learner.py`, `eval_policy.py` and `gym_manipulator.py`, with factory branches in `robots/utils.py` and `teleoperators/utils.py`. `envs/__init__.py` exports `NexArmEnv`.

## Tests run (Windows, CPU, `uv run --no-sync`)

| Tests | Result |
| --- | --- |
| `tests/scripts/test_record_collection.py`, `tests/datasets/test_collection_merge.py`, `tests/scripts/test_prepare_collection.py` | 19 passed |
| `tests/nexarm_service`, `tests/policies/test_optional_policy_imports.py`, `tests/utils/test_visualization_utils.py` | 133 passed |
| `tests/envs/test_nexarm_env.py` | 28 passed, 4 failed (all `mujoco.FatalError: gladLoadGL`, the plan's known no-OpenGL environment) |
| `tests/scripts/test_train_stack_bowls_act.py`, `tests/envs/test_nexarm_sim_gym_integration.py` | 4 passed, 2 failed (same OpenGL cause) |
| `tests/datasets/test_lerobot_dataset.py -k local_files_only` | passed |
| Scratch manual-path test (M2) | 2 passed |
| Scratch rerun sink experiment (H1) | confirms the defect and the fix |

## Recommended actions (priority order)

1. Fix `init_rerun` to tee sinks with `rr.set_sinks`, and add a test that checks the file contains data (H1). Do this before the real-arm smoke test that uses `--rerun_save_path`.
2. Update the five docs that promise a printed id from plain `lerobot-record`, or route them through `record_collection.py` (M1).
3. Commit the manual-path test (M2).
4. Low items as time allows (L1-L3, S2).

## Unresolved questions

- Partway through this review, `uv run` failed once with "Workspace member `policies/lerobot_policy_bspline` is missing a pyproject.toml". The `policies/*` glob comes from `02d3d980c`, and the directory appeared while another agent was working. A retry worked. This is outside the reviewed range and does not block, but a half-created workspace member breaks every `uv run` in this checkout.
- Should H1 be fixed on this branch, or separately on `main`, given that it was already there before the merge?
