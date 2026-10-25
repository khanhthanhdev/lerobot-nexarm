# Real dataset analysis: `thanhkt/nexarm_stack_bowls`

This report analyses the joint data in the real dataset (no video) to check the assumptions in
`plans/261014-1423-nexarm-sim-fidelity/`. All values are in raw servo ticks; 4096 ticks = 2π rad. The
analysis scripts are in the session scratchpad and are not committed.

## Dataset facts

- LeRobot v3.0, 30 fps, `robot_type="nexarm_follower"`, 163 episodes and 46,765 frames.
  - Episodes are 144–449 frames long (median 269, about 9 s).
  - Task 0 (blue on the bottom) has 22 episodes; task 1 (red on the bottom) has 141.
- Cameras are `front` and `wrist` (OpenCV, 640×480). **There is no `top` D435i stream**, so this dataset cannot
  check the D435i field of view.
- `meta/collection_sessions.json` lists only 70 of the 163 episodes, so the settings for the rest are not
  recorded. The default `record.yaml` uses `motion_speed=2000` and `motion_acc=100`.
- Observed joint ranges (state):
  - pan 1327–2667, lift 1149–2451, elbow 963–3058, wrist flex 883–3087, wrist roll 683–2243;
  - gripper 1195–2785. The command reaches 2833, but the measured jaw never does.

## Finding 1: the real arm is not capped at 2000 ticks/s (assuming a true 30 Hz time base)

> **Caveat:** LeRobot writes `timestamp` as frame index ÷ fps, not as measured clock time, so every dt is exactly
> 0.0333 s. If the record loop ran slower than 30 Hz, every speed below is inflated by 30 ÷ the actual rate. At
> 20 Hz the elbow's 3190 maximum would become about 2130, close to the 2000 cap. The gripper's 4000 ticks/s,
> 17% above the vendor no-load figure, hints at this.
>
> Either check settles it:
>
> - confirm that `lerobot-record` never printed "Record loop is running slower" during recording;
> - or time one joint on the robot with `perf_counter`, at `motion_speed=2000` and again at 0.
>
> **Update (user, 2026-10-14):** the warning was printed often during recording, so the time base is unreliable
> and Finding 1 is unproven.
>
> - Six episodes end at 448–449 frames, which looks like a time limit. At 15 s that would mean an average of
>   about 30 Hz with short stalls; at 20 s it would mean about 22 Hz. The configured `episode_time_s` is unknown.
> - The elbow moves more than 67 ticks in one frame (the most 2000 ticks/s allows at 30 Hz) 946 times. Some
>   steps reach 100–120 ticks, consistent with stretched frames.
> - Only a timed step test on the robot can now settle the cap. Uneven frame spacing also adds noise to the
>   replay and latency fits.
>
> **Update 2 (user: `episode_time_s=15`):** the episodes that ran the full 15 s settle Finding 1.
>
> - Seven episodes ran to the limit with 440–449 frames: 2, 19, 23, 25, 26, 29 and 37. Their loop therefore
>   averaged 29.3–29.9 Hz, and stalls totalled at most about 0.3 s per episode.
> - In those episodes:
>   - wrist flex still reaches 3050 (episode 25), 2700 (episode 26) and 2295 ticks/s over 200 ms;
>   - the elbow reaches 2340 ticks/s;
>   - the gripper reaches 3330–3750 ticks/s.
> - A one-frame stall inside a 200 ms window inflates speed by at most 7/6, which still leaves 2600 for wrist
>   flex.
> - **Conclusion:** `motion_speed=2000` does not cap moves at 2000 ticks/s. The gripper is close to the 3413
>   spec, within about 10%.
> - Fit speeds and latency on these full-length episodes, where frame spacing is trustworthy.

Speeds were measured over 200 ms windows (6 frames) to suppress frame jitter.

| Joint        | p95  | p99.9 | max  | Windows above 2100 ticks/s        |
| ------------ | ---- | ----- | ---- | --------------------------------- |
| 1 pan        | 725  | 1601  | 2495 | 7                                 |
| 2 lift       | 925  | 1540  | 1965 | 0                                 |
| 3 elbow      | 1540 | 2783  | 3190 | 632 windows in 92 of 163 episodes |
| 4 wrist flex | 1080 | 2410  | 3050 | 126 windows in 28 episodes        |
| 5 wrist roll | 810  | 1690  | 2215 | 2                                 |
| gripper      | 2750 | 3770  | 4020 | —                                 |

- **The excess appears across the whole dataset, so mixed recording settings don't explain it.** Every
  10-episode block except episodes 30–39 and 70–79 has an elbow maximum above 2100 ticks/s, and the median per-episode
  elbow p99 is above 2000 in 10 of 17 blocks.
- **The joint follows the command above 2000 ticks/s.** When the elbow command moves at 2500–4000 ticks/s,
  the measured elbow moves at a median of 2335 ticks/s. For wrist flex the figure is 1948.
- **Conclusion:** if these episodes used `motion_speed=2000`, then CMD 56 does not cap CMD 97 position writes in
  ticks/s. Either it is not applied to them, or its units are different. The sim cap planned in phase 3 (2000
  ticks/s, about 3.07 rad/s) would make the sim elbow slower than the real one.
- **The gripper is faster than the spec-derived figure.** It reaches about 3800–4000 ticks/s, above the 3413
  ticks/s no-load maximum taken from 0.2 s/60°.
  - Two clean full strokes took 0.25 s for 80% of the travel. Most closes stop on a bowl, so the sample is small.
  - A full stroke therefore takes about 0.4 s. Acceptance criterion 3 requires at least 0.8 s, which
    contradicts the hardware.

## Finding 2: tracking lag of 2–3 frames

The best shift between command and measured state is 2 frames (89 episodes) or 3 frames (53 episodes), i.e.
67–100 ms. It includes the servo response, not only transport latency, and is the starting latency guess for
`calibrate_sim.py`.

Episodes 69–89 show zero lag. These 21 episodes look like a separate session, roughly matching the 20-episode
`local/` session. Command and state there may have been logged differently, so exclude them from latency fits
until checked.

## Finding 3: the gripper is compliant when closed

While the gripper is commanded closed (above 2700) and has stopped moving:

- the median measured position is 72 ticks short of the command (interquartile range 49–83);
- the 5th percentile is 619 ticks short, where the jaws stop on a bowl rim.

The real gripper is therefore a stiff spring with steady-state error, not a hard position lock. The planned
116 N, kp = 998.22/r² gripper will close to almost zero error, so the steady-state error with no object is a
tuning target for phase 3.

## What this data can and cannot do

- **Can:** a replay-error baseline per joint before and after phase 3 (contact biases it, so it is a check, not
  a fit). It can also fit latency and gain, ideally after free-air sweeps are recorded. It can give bowl start
  positions from forward kinematics at gripper closes, and a front-camera overlay check.
- **Cannot:** verify the D435i `top` camera; that needs a `pyrealsense2` read on the robot PC.

## Unresolved questions

1. Were all 163 episodes recorded with `motion_speed=2000`? Only 70 have session metadata.
2. Why do episodes 69–89 show zero command-to-state lag?
3. Which speed model the sim should use, given Finding 1 (decision pending with the user).

## Finding 4 (blocking): real joint angles do not map to the same arm pose in the sim

Real states were run through sim forward kinematics (`sim/description/mjcf/scene.xml`, mapping
`servo_4096_v1`: rad = (raw − 2048)·2π/4096 with no per-joint sign or offset). The results were compared with
the real camera frames.

| Episode 70 frame | Real state (raw)             | Real camera shows                                        | Sim forward kinematics says                                                                          |
| ---------------- | ---------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 0 (start)        | 1798, 1833, 1479, 2264, 1517 | Wrist camera looks down at the red bowl from close range | Wrist camera at z 0.39 m, looking almost horizontal (direction −0.95 y, −0.28 z)                     |
| 88 (grasp)       | 2166, 1653, 2169, 2717, 1503 | Jaws closing on the blue bowl rim, near the table        | Jaw at (−0.003, −0.015, 0.079), directly above the robot base, with the camera looking backward (+y) |

- **Confirmed on a clean episode.** Episode 150 has a correct side-view `front` stream and normal lag. At frame
  233 (state 1578, 1258, 2772, 3073, 684) the side view shows the jaws on the blue bowl rim near the table, about
  25 cm in front of the base. The sim puts the jaw at (−0.06, 0.07, 0.33): 33 cm up and behind the base.
- In episode 0, the sim path passes through the base column (jaw at about 0, 0, 0.08 m).
- Testing 16 sign patterns and fitting offsets for 2 of them did not produce consistent grasp heights. Those
  grasp events were not checked against video, so this does not rule out a simpler sign-plus-offset fix.
- **Consequences:**
  - real and sim datasets that share `robot_type="nexarm_follower"` encode different arm poses for the same
    joint values, so co-training mixes inconsistent kinematics;
  - `calibrate_sim.py` replay fits and any real→sim replay are invalid until the mapping is fixed.
- **Ground truth needed:** photos of the real follower at a few known raw poses, starting with the vendor Home
  pose (all joints 2048), compared against the sim at the same values.

## Finding 5: camera streams are swapped in episodes 69–92

The `front` video for episodes 69–92 shows the wrist view (jaws in frame). Episodes 143–162 have a correct
side view of the table, with the robot base on the right. These episodes overlap the zero-lag group (69–89), so
that session probably had both its camera indices and its logging set up differently. Check episodes 69–92
before training on them.

## Correction to Finding 1 (lag-aligned audit)

`examples/nexarm/audit_real_dataset.py --episode-time-s 15` compares each command window with the state window
2 frames later. Doing so removes the apparent elbow saturation: the "median 2335 ticks/s at 2500–4000 commands"
figure was response lag.

Measured on the 7 full-length episodes:

| Joint         | p99.9 (ticks/s) | max  | Tracking ratio at fastest commands | Verdict                         |
| ------------- | --------------- | ---- | ---------------------------------- | ------------------------------- |
| shoulder_pan  | 1525            | 1600 | 1.01                               | not saturated; lower bound only |
| shoulder_lift | 1628            | 1690 | 0.98                               | not saturated                   |
| elbow_flex    | 2145            | 2340 | 1.00                               | not saturated                   |
| wrist_flex    | 2926            | 3050 | 0.98                               | not saturated; still above 2000 |
| wrist_roll    | 1625            | 1750 | 0.98                               | not saturated                   |
| gripper       | 3725            | 3750 | 0.66                               | saturated: cap about 3700       |

- `motion_speed=2000` still does not cap moves: wrist flex tracks its command at 3050 ticks/s.
- No arm joint reached its servo limit in these demos. Arm caps stay at the 3413 spec; only the gripper cap
  comes from data.

## Finding 6: real releases do not trigger the sim grasp-assist release

380 of 394 real releases open the gripper command only to a median of 1989, which is above the stack-bowls
assist release threshold of 1600. Replayed in the sim, those bowls would never be let go.
`NexArmStackBowlsTask(assist_release_pos=...)` now makes the threshold configurable; use about 2300 for real
replays.
