# Repository Comparison: Hiwonder `finsh` vs `lerobot-nexarm`

## Source manifest

- Source: `/home/thanh/code/vinuni/finsh`
- Local project: `/home/thanh/code/vinuni/lerobot-nexarm`
- Mode: comparison only
- Source version: LeRobot `0.5.2`
- Local version: LeRobot `0.6.1`
- Source commit: unavailable because the packaged `.git` directory lacks repository metadata such as `HEAD` and `config`
- Local commit at comparison: `e3d4a938f163bbd7ec328a035dff2ede9b15a653`

The source package was inspected as untrusted, read-only data. No source commands
were executed and no dependencies were installed.

## Summary

`finsh` is the older Hiwonder NexArm distribution described by the vendor
tutorial. The local project retains the same serial protocol and core
leader/follower abstractions, but it has been rebased onto a newer LeRobot,
adapted to newer device-factory and typing conventions, hardened for hardware
errors, and extended with MuJoCo simulation, firmware, and newer examples.

## Head-to-head

| Aspect | `finsh` | Current project | Recommendation |
| --- | --- | --- | --- |
| LeRobot base | 0.5.2 | 0.6.1 | Keep the current base |
| Package quality | Contains `.venv`, caches, egg-info, and incomplete `.git` data | Normal working Git repository with `uv.lock` | Do not treat `finsh` as a clean upstream repository |
| NexArm protocol | Commands 68, 96, 97, and 98 | Same commands plus command 56 motion acceleration/speed | Keep the current protocol implementation when using matching firmware |
| Follower startup | Enables torque with no motion-profile configuration | Sends command 56, then enables torque | Prefer current behavior if the AT32 firmware supports command 56 |
| Serial feedback | Reads positions after every action | Returns commanded action; observations perform separate reads and sanitize corrupt values | Prefer current separation for LeRobot action/observation semantics |
| Corrupt readings | Clamps only to the raw range | Reuses prior values near 0/4095 and logs large jumps | Prefer current behavior, but add tests |
| Disconnect | Torque remains enabled by default | Holds position briefly, then disables torque by default | Current is safer for shutdown, but users must support the arm because it will relax |
| Camera timeout | Falls back from `read_latest()` to `async_read()` | Uses the newer camera API directly | Keep current behavior unless Windows stale-frame failures recur |
| Device factory | Explicit NexArm branches in utility factories | New generic device-class factory | Keep the current LeRobot convention |
| Examples | Four editable YAML files matching the vendor tutorial | Argument-driven Python wrappers plus direct CLI examples | Current wrappers are more portable; YAML templates remain useful documentation |
| Simulation | None | `nexarm_sim`, MuJoCo model, viewer, and tests | Keep current simulation support |
| Firmware/docs | Not included | Vendor chapters and organized ESP32/binary bundle included | Current project is more complete |
| NexArm tests | Mock serial and follower tests | Only NexArm simulation tests | Port and update the useful protocol/hardware mock tests |
| Dependency extra | Defines `nexarm = [pyserial, av]` and includes it in `all` | `nexarm` extra is missing | Restore a current-compatible `nexarm` extra |

## Important inconsistencies

1. `finsh` defines `disable_torque_on_disconnect = False`, while its own test
   asserts that the default is `True`. The package therefore contains stale or
   unverified tests.
2. The current project documentation still describes a `nexarm` optional
   dependency, but the current `pyproject.toml` does not define it.
3. `finsh` examples assume fixed COM ports and camera indices. These are sample
   values, not hardware discovery.
4. Current command-56 motion tuning depends on newer follower/AT32 firmware and
   should not be assumed compatible with every factory image.

## Challenge questions

| Question | Source answer | Local answer | Risk if wrong |
| --- | --- | --- | --- |
| Is a wholesale copy needed? | Provides an older complete NexArm fork | Already contains and extends the same integration | Copying would regress LeRobot and simulation work |
| Are the serial protocols compatible? | Uses 68/96/97/98 | Retains them and adds 56 | Old firmware may ignore or mishandle motion tuning |
| Should YAML examples replace wrappers? | Vendor docs depend on YAML | Wrappers expose ports and cameras explicitly | Replacing wrappers would make stale machine-specific values easier to run accidentally |
| Are source tests trustworthy as-is? | Good protocol coverage, but at least one stale assertion | Hardware mock coverage is missing | Blind copying would create failing or misleading tests |
| Can users follow `pip install -e ".[nexarm]"` locally? | Yes | No extra currently exists | Fresh installs may miss `pyserial` or video dependencies |
| Should action calls read feedback immediately? | Yes | Observations read feedback separately | Restoring read-after-write would add serial latency and blur action semantics |

## Decision matrix

| Decision | Source's way | Local way | Recommendation |
| --- | --- | --- | --- |
| Base repository | LeRobot 0.5.2 | LeRobot 0.6.1 | Local |
| Hardware driver | Basic bridge | Hardened bridge plus motion parameters | Local, gated by firmware compatibility |
| User configuration | YAML templates | Python wrappers and CLI flags | Hybrid: keep wrappers, optionally restore sanitized YAML examples |
| Tests | Protocol and physical-driver mocks | Simulation tests | Hybrid: adapt the mock tests to current behavior |
| Dependencies | Dedicated `nexarm` extra | Missing dedicated extra | Restore the source concept using current dependency versions |
| Documentation | Vendor-specific guide | General repo guide plus vendor chapters | Local as authority; vendor docs as hardware reference |

## Recommendation

Do not replace the current project with `finsh`. Use `finsh` as a vendor
reference and selectively recover two pieces: a current-compatible `nexarm`
dependency extra and updated mock tests for framing, mapping, serial I/O, and the
follower lifecycle. YAML templates are optional convenience assets, not core
architecture.

Risk is **medium** because dependency installation and firmware command-56
compatibility must be resolved before claiming that the current checkout is a
drop-in replacement for the vendor package.
