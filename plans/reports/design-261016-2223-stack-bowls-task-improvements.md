# Stack-bowls task: design review and improvement plan

Date: 2026-10-16. Repo `lerobot-nexarm` at `88b1df68`. This is a read-only design review: no code was changed.

All measurements below come from scratch probes, run headless with no cameras. They used the committed `NexArmStackBowlsTask`, `NexArmStackBowlsEnv`, and the scripted expert in `examples/nexarm/generate_stack_bowls_dataset.py`. The probes have since been deleted.

Related records:

- `plans/261014-1423-nexarm-sim-fidelity/` (the sim fidelity plan; phases 0 and 3 are pending)
- `plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md` (analysis of the real recordings)

## Outcome to aim for

"Better and more accurate" means three things:

1. The sim task means the same thing as the real `thanhkt/nexarm_stack_bowls` task:
   - the same arm pose for the same joint ticks;
   - the same bowl geometry and the same release behaviour.
2. The grasp is honest enough that a policy trained or evaluated in sim does not learn behaviour the real arm cannot do.
3. The reward, the success test, and the metrics push RL towards the real goal, and they diagnose failures by stage.

## What I measured

| Probe                                                          | Result                                                                                                                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Scripted expert, 24 seeds per jitter value, assist on          | Jitter 0.005 m: 24/24 succeed. Jitter 0.02 m: 24/24. Jitter 0.04 m (the gym env default): 23/24; one `ik_p2_grasp_missed`.                                   |
| Commanded 12 cm lift of the middle bowl, assist **off**        | The bowl's centre rises 6.5–8.1 cm, and it hangs tilted 45–66°.                                                                                              |
| Same lift: grasped rim point vs jaw                            | The jaw rises 11.8–12.0 cm and the grasped rim point rises 12.4–13.0 cm. The bowl does **not slip**: it is held, but it **pivots** about the pinch.          |
| Assist off with `noslip_iterations=3`, jaw `condim=6`, or both | No change: tilt stays at 43–66°.                                                                                                                             |
| Same, with gripper kp ×3                                       | Tilt falls to 38–40°. The bowl still hangs.                                                                                                                  |
| Jaw pitch sweep, 55–90° (default is 75°)                       | Tilt stays at 42–74°. At pitch 30° or lower the IK misses the rim.                                                                                           |
| Jaw-bowl contacts after closing                                | 2–4 contact points with normal forces of 0.3–2 N. The contact normals have z components of 0.23–0.38: vertical jaw faces pressing on the slanted proxy wall. |
| Proxy bowl geometry (`bowl_stack_scene.xml`)                   | The wall is 32 capsules of radius 5.5 mm, so it is **11 mm thick**. It slants about 23° (r 48→61 mm over z −14→+16 mm). Total mass is 33 × 1.818 g = 60 g.   |
| Gripper free-air gap vs raw command                            | 1195 → 50.9 mm, 1600 → 38.3 mm, 2000 → 25.9 mm, 2400 → 13.4 mm, 2833 → 0 mm.                                                                                 |
| Dense reward along the scripted episode (seed 0)               | Holding the top bowl in place pays **6.44 per step**. Right after release it pays **4.45 per step**. Success adds 5 once, and then the episode terminates.   |

**Correction to my earlier message:** I said an unassisted bowl rises only 1–3 mm. That number came from the generator aborting at its held-bowl check before any lift was commanded. With the lift commanded directly, the bowl is lifted, but it hangs at a steep tilt.

## Findings, ordered by impact

### P0: blocks accuracy against the real task

**A. The real-to-sim joint mapping is still wrong.**

- The mapping version is still `servo_4096_v1` (`motors/nexarm/mujoco_mapping.py:40`), and phase 0 of the fidelity plan is pending.
- Until v2 lands, real joint ticks pose the sim arm differently (Finding 4 of the real-data report). Every real-data comparison of stack-bowls is therefore invalid:
  - bowl-layout inference from grasp poses,
  - replay error,
  - co-training on real and sim data.
- Do this first. It needs one robot session: `examples/nexarm/probe_joint_mapping.py`.

**B. The bowl proxy is not measured from the real bowl.** The proxy fixes four things:

- how the jaws meet the wall (11 mm thick and slanted 23°),
- the nesting heights, which the success thresholds `min/max_nesting_delta_m = 0.020/0.038` depend on,
- the rim height used by the scripted IK (`BOWL_RIM_Z_OFFSET = 0.0215`),
- the concentric tolerance (35 mm).

Measure one real bowl with calipers and a scale:

- wall thickness at the rim and at mid-height, and the rim lip if there is one;
- outer and inner rim radius, base radius, height, and wall slant;
- the height step of two and three nested bowls (this sets the nesting thresholds directly);
- mass. The fidelity plan records 55 g measured by the user, but PR #2 and `test_bowls_weigh_60_g` use 60 g.

Then refit the capsule ring (radius, thickness, slant) and the thresholds from those numbers, and add a test that compares the proxy with the recorded measurements.

### P1: training-signal bugs (each can be fixed now and has a test the current code fails)

**C. The dense reward pays more for holding the last bowl than for finishing the task.**

- The cause is in `envs/nexarm.py:460-487`. `stage` (2 per nested bowl) is paid whether or not the bowl has been released. `reach`, `grasp`, and `carry` keep paying while the top bowl is held.
- Since success terminates the episode, holding until timeout earns about 6.44 × the remaining steps (up to 900), while releasing earns 5 once. PPO will learn never to let go.
- Fix, in order of preference:
  1. Make the terms potential differences, `Phi(s') - Phi(s)`. Order Phi so that each stage, including release, is strictly higher than the one before. Keep success as the dominant terminal term.
  2. A minimal fix instead: gate `stage` on `is_released`, and stop paying `grasp`/`carry` once the mover bowl is nested.
- Regression test: walk the scripted trajectory and assert two things:
  - the per-step reward does not drop at release;
  - "hold the nested top bowl until timeout" has a lower return than "release and succeed".

**D. The env cannot release bowls the way real operators do.**

- Real releases open the gripper only to a median raw value of 1989 (Finding 6), which is a 26 mm gap. The assist drops the bowl only when the command falls below 1600 (`stack_bowls_task.py:23`).
- `NexArmStackBowlsEnv._make_task` never passes `assist_release_pos` (`envs/nexarm.py:438-444`). A policy trained on real data and evaluated in the gym env therefore never lets go.
- Fix:
  - expose `assist_release_pos` (and the engage threshold) in `NexArmEnv` config;
  - default it to about 2300, which the analysis recommends for real replays;
  - record both values in dataset metadata.

**E. The success test and the metrics do not say which stage failed.**

- Success already checks that the stack is concentric, nested, released, and settled: linear speed below 0.05 m/s, angular speed below 0.5 rad/s, and tilt under 15° (`stack_bowls_task.py:262-300`). That part is good.
- What is missing:
  - per-stage outcomes in `info`: `picked_middle`, `nested_middle`, `released_middle`, `picked_top`, `nested_top`, `released_top`;
  - the first failing stage;
  - a check that the arm is static.
- Eval reports should give per-stage rates, not only the final success rate. This is the cheapest way to see why a policy fails.

### P2: grasp model (a scope decision; see question 2)

What the probes show: the bowl does not slip. It pivots about a near-point pinch, because vertical jaw faces touch a slanted, 11 mm thick capsule wall at 2–4 points. Point contacts have no lever arm against the tilting moment, so friction cone shape (`condim`), NoSlip, and pitch change nothing, and more gripper force helps only a little.

The fix is contact geometry, not friction:

- make the proxy wall thin and its slant match the real bowl (after measurement B);
- spread each jaw pad as 2–3 small boxes or spheres stacked vertically, so the pinch resists tilting.

There are three ways to model the grasp:

| Option                                    | What it is                                                                                                                 | Pros                                                                                                                                            | Cons                                                                                                                                                                                                                        |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Teleport assist (current)              | The bowl's qpos is set rigidly to the gripper frame every substep (`stack_bowls_task.py:213-226`)                          | 100% scripted success; simple                                                                                                                   | Contacts go through the constraint solver one way only: a carried bowl pressed into another one kicks it with unbounded force, and a held bowl never slips or tilts. Policies can learn closures that fail on the real arm. |
| 2. Weld equality toggled with `eq_active` | Add one disabled `<weld>` per bowl to the gripper body, and enable it at grasp with the same engage and release hysteresis | Finite, solver-consistent forces; the carried bowl interacts physically with other bowls; little code (swap the teleport in `_on_physics_step`) | Still not a physics grasp; the weld's `solref` needs tuning                                                                                                                                                                 |
| 3. Physics only                           | No assist                                                                                                                  | Most honest for sim-to-real transfer                                                                                                            | Needs B, the pad-geometry fix, and the phase-3 servo model first. The fidelity plan lists a physical bowl grasp as a non-goal today.                                                                                        |

Recommendation:

- Move from option 1 to option 2 now. It is small and removes the unbounded-force artefact.
- Keep option 3 as an explicit later milestone. Gate it on the measured bowl, the pad geometry, and phase 3. The phase-3 gripper (116 N, kp_rot/r²) replaces today's kp 300 and 50 N limit, so tuning contacts before phase 3 means tuning them twice.
- Whatever you choose, write `grasp_mode` and both thresholds into the dataset metadata. Assisted sim data can then be filtered or weighted when you co-train.

### P3: scene and episode distribution

**F. The layout distribution is too narrow and is not matched to the real data.**

- Bowls go into three fixed slots, with ±5 mm jitter in the generator and ±40 mm in the gym env (`stack_bowls_task.py:27`, `envs/nexarm.py:434`).
- After mapping v2 (A), infer the real start positions from forward kinematics at the first gripper close of each real episode. Then sample sim layouts from that distribution, with rejection sampling for separation, instead of using fixed slots.
- Yaw randomization does almost nothing, because the proxy bowl is rotationally symmetric. It only matters if bowls get a visible texture or logo.

**G. The stacking order distribution.**

- The sim cycles all 6 orders uniformly by seed (`stack_bowls_task.py:135-137`).
- The real data has 2 orders: 141 episodes with red on the bottom and 22 with blue on the bottom.
- Uniform is better for language generalisation. A matched mode is better for co-training and for comparing sim and real success. Offer both (`order_distribution: uniform | real`).

**H. Real-world variation the sim lacks** (add each only when it matters, and measure it first):

- the bowls' real friction (plastic on the table surface) and the real jaw-pad material;
- small bowl size variation, if your bowls differ;
- start states where bowls are already partly stacked or touching, which happens in real resets;
- the real control timing (2–3 frames of latency, Finding 2). This already exists through `calibrate_sim.py`; use it in the stack-bowls generator by default.

### P4: scripted expert (data quality)

- Its success already looks good (23–24 of 24). Two improvements:
  - make it robust at jitter 0.04 m: retry IK with another seed on `ik_p2_grasp_missed`;
  - vary the approach. Currently every demo approaches the rim on the line from the base, at fixed heights (+0.08 m and +0.12 m) and fixed step counts. Randomize the approach heights, speeds, and rim angle within the reachable set, so imitation policies do not overfit one motion.
- Keep recording what was sent, after clamping, as the action. It already does this.

## Suggested sequence

```diagram
┌────────────────────────────────┐   ┌──────────────────────────────┐
│ Now, no robot needed           │   │ Needs the robot / real bowl   │
│ C reward fix + test            │   │ A joint mapping v2 (phase 0)  │
│ D release threshold in env     │   │ B bowl measurements           │
│ E per-stage info + metrics     │   └──────────────┬───────────────┘
│ Grasp option 2 (weld toggle)   │                  │
│ G order_distribution option    │                  ▼
└───────────────┬────────────────┘   ┌──────────────────────────────┐
                │                    │ Refit bowl proxy + thresholds │
                │                    │ F real layout distribution    │
                │                    │ Phase 3 servo model           │
                │                    └──────────────┬───────────────┘
                └──────────────┬───────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ Pad geometry, then option 3  │
                │ (physics-only grasp), if     │
                │ chosen; compare sim vs real  │
                │ per-stage success            │
                └──────────────────────────────┘
```

Acceptance checks for the whole effort:

1. The reward regression test from C passes: the return for holding is less than the return for releasing and succeeding.
2. The gym env releases on a replayed real-style release, with a gripper command of about 1990.
3. Eval output reports a per-stage success table.
4. After A and B, real-episode grasp poses put the jaws at the measured rim height, within 1 cm.
5. After A and B, the sim nesting thresholds come from the measured nesting steps.
6. The scripted-expert success rate stays at 95% or higher on fixed seeds after each change.

## Unresolved questions

1. Bowl mass: the plan records 55 g measured by you, but PR #2 and its test use 60 g. Which is right?
2. Grasp model: keep the teleport assist, switch to a weld toggle (recommended for now), or aim for physics only? Physics only is a non-goal in the current fidelity plan.
3. Should the sim's order distribution match the real data (mostly red on the bottom) for co-training, or stay uniform?
4. Should I turn this into an executable plan under `plans/` and start with the items that need no robot (C, D, E)?
