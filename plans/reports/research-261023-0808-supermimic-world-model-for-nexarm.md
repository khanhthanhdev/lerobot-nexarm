# SuperMimic (arXiv 2610.28480) and its use for the NexArm policy stack

Status: research note, 2026-10-23. Nothing here has been run. Facts are tagged
(paper), (repo), (real data) or (estimate).

## 1. What the paper actually does

SuperMimic (Liao et al., UC Berkeley, CoRL 2026) trains one humanoid
motion-tracking policy for the Unitree G1 across 129,514 mocap clips.

- **World model** (paper, §3.2, Table A.4): a deterministic MLP (3 x 2048) that
  predicts the next state from (state, action, sim context xi). xi is the
  per-episode domain-randomization vector (friction, calibration offsets, torso
  CoM, Table A.3). Reward, observation and termination heads are learned or
  analytic. It is fit on 8-step autoregressive windows sampled from a replay
  buffer of MuJoCo Warp transitions (Eq. 1), collected by the current policy
  with N(0, 0.1^2) action noise.
- **Policy** (paper, §3.3): deterministic, trained by backpropagation through
  time over H = 32 imagined steps through the frozen world model, plus a GAE
  critic (lambda 0.95, gamma 0.99) that sees xi and bootstraps beyond H. The
  actor sees the last 10 noisy proprioceptive frames and K = 10 future
  reference frames; it never sees xi. Actions are joint setpoints under
  deliberately low PD gains, so a setpoint is a torque proxy (Appendix A).
- **Why it is fast** (paper, §4.2): one model amortizes the dynamics shared by
  all motions (8-step error stays 0.007 to 0.011 from 1 h to 264 h of data);
  world-model rollouts run 11x to 62x faster than MuJoCo Warp; analytic
  gradients reach PPO's best return with about 64x fewer imagined steps.
- **Headline results** (paper, §4.1): surpasses the SONIC checkpoint (128 L40
  GPUs for 168 h) after 3 h on one H100; 25 failed motions vs 179, and 0 after
  9.8 h on two GPUs. In a controlled LAFAN1 comparison, PPO in the simulator
  reaches 0.60 / 0.70 / 0.72 return on 1 / 4 / 8 GPUs in 18 h, while SuperMimic
  reaches 0.82 in 1.2 h on one GPU.
- **Critic matters** (paper, §4.3): without it, training fails at H <= 16 and
  the policy never recovers from pushes.
- **Real robot** (paper, §4.4): four motions on a G1 (skip jump, side roll,
  90 and 360 degree flips), with the same domain randomization as BeyondMimic
  minus root-velocity pushes.
- **Stated limitations** (paper, §6): the world model learns only from
  simulator transitions; the authors say real-robot data could refine it and
  that its differentiability could support system identification. It is
  trained on flat ground with **no object interaction**. Pushes were removed
  from randomization because a deterministic model cannot capture state
  discontinuities (Appendix A).

The honest one-line summary: SuperMimic is a compute-efficiency result for
**large motion libraries** on a **torque-authority** robot with a **dense
tracking reward**. Those three ingredients are what make the gains real.

## 2. Does the premise transfer to the NexArm?

| Ingredient in the paper | NexArm situation | Verdict |
|---|---|---|
| Thousands of motions; cluster-scale PPO is the bottleneck | 6-DOF arm, one task family; PPO in MuJoCo with 4096 envs trains a 6-DOF tracker in minutes (estimate) | The headline gain does not apply |
| Low-gain PD so setpoints act as torque | Bus servos with a stiff internal position loop; the bridge exposes only CMD 97 position write and CMD 96 position read (repo, `motors/nexarm/nexarm.py`) | The learned controller can only reshape setpoints |
| Dense per-step tracking reward | Available for joint tracking; end-effector reward needs correct forward kinematics | Possible, but see §3.3 |
| Contact-free (flat ground, no objects) | Stack-bowls is contact-rich; grasp and release are discontinuities | The paper's own open problem |
| Sim context xi with friction, offsets, CoM | Delay, payload, gripper compliance, firmware motion profile | Needs a servo model that does not exist yet |

## 3. Critique of the pasted "SuperMimic-style tracker" proposal

### 3.1 Step 0 will almost certainly say "stop" (real data)

The real stack-bowls dataset already answers most of Step 0
(`plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md`):

- Command-to-state lag is 2 frames (67 ms) in most episodes, 3 in some.
- No arm joint saturates in the demos. At the fastest commands the tracking
  ratio is 0.98 to 1.01 on all five arm joints. Only the gripper saturates
  (cap about 3700 ticks/s).
- The closed gripper settles 72 ticks short of its command; the error is
  compliance, not lag.

So the tracking "problem" is a 2-frame pure transport delay plus gripper
compliance. A pure delay is corrected by sending the chunk's target for
t + 2 instead of t. The action chunk already contains those future targets,
so this is a one-line change in the rollout loop, not a learned controller.

### 3.2 The residual bound defeats the purpose

The proposal bounds the residual Delta to "a few degrees". With a 2-frame
delay and demo speeds up to about 3000 ticks/s (4.6 rad/s), the lead needed at
peak speed is about 2 x 4.6 / 30 = 0.31 rad, about 17 degrees (estimate from
real data). A few degrees of residual cannot compensate the lag where it
matters, and where it does not matter the residual is not needed. If a
residual tracker is ever built, bound it in proportion to commanded velocity,
or let it shift the reference in time instead of in angle.

### 3.3 Blocking: sim and real joint values do not describe the same pose

Finding 4 of the real-data report: the sim's 2048 is the CAD pose, the real
follower stands upright at 2048, and the mapping has not been fixed yet
(phase 0 of `plans/261014-1423-nexarm-sim-fidelity/`). Until that lands,
every end-effector reward, every real-to-sim replay for system identification
and every sim-trained tracker is computed on the wrong arm pose. The proposal
does not mention this dependency; it must come first.

### 3.4 The telemetry it relies on does not exist in this fork

- The proposal reads current and a "status byte 0x41". There is no such
  read path here. Register 41 is the HX servo **acceleration** register used
  by the reg-write path (repo, `REG_ACC = 41`). LeRobot bridge mode has no
  current, velocity, temperature or load read (repo, CMD 96 returns six
  positions only).
- The "gripper telemetry note" and "Loop-Bench-Edge" are not in this
  repository. Do not plan against them until they are checked in.
- Writable PID gains on HX-series bus servos are unverified. Without them the
  "setpoint-only authority" caveat in the proposal is the actual situation.

Consequence: the overload-penalty reward has no measurable real counterpart,
and the sim-context parameters kp, damping, friction, backlash and overload
threshold are **not identifiable from position-only logs**. The servo is a
black box from setpoint to measured position. Fit that map directly rather
than a five-parameter physical model whose parameters cannot be separated.

### 3.5 A PD actuator is the wrong servo model

HX bus servos execute a goal position with their own speed and acceleration
registers (repo, `REG_GOAL_SPEED = 46`, `REG_ACC = 41`, and CMD 56 sets the
LeRobot-mode acceleration). That is a profiled move, not a PD response. The
current MJCF (`sim/description/mjcf/nexarm.xml`) uses position actuators with
kp 20 to 80 and torque-limited force ranges; it reproduces stall torque but not
the firmware's motion profile. A rate-and-acceleration-limited first-order
response fitted per joint from real step data is closer to the hardware and
has two parameters per joint that *are* identifiable.

### 3.6 The world model buys nothing on a 6-DOF arm

The paper's model is worth it because 29-DOF contact dynamics are expensive to
simulate and shared across 129k motions. A 6-DOF arm in free space with a
first-order servo model is cheap to simulate and nearly decoupled per joint.
Options that are strictly simpler than a learned world model, in order:

1. the analytic servo model is itself differentiable, so BPTT works directly
   through it (SHAC style, which the paper cites as its inspiration);
2. MJX is differentiable if you want the full rigid-body chain;
3. PPO on the plain simulator converges in minutes at this scale (estimate).

The proposed five-condition evaluation (raw, hand-tuned lead, PPO, SuperMimic,
no-delay ablation) is well designed, but the most likely outcome is that
conditions 2 to 5 tie and condition 2 took an afternoon.

### 3.7 A risk the proposal misses: the policy was trained on lagged data

The imitation policy learned from demos in which the follower state trails the
command by 2 frames. Any tracker that removes that lag changes the state
distribution the policy sees at inference. It can help (states closer to the
leader's intent) or hurt (policy now sees states 2 frames "ahead" of what it
expects). Either way, the tracker has to be in the loop **during data
collection** as well, or the policy must be retrained on tracker-on rollouts.
This is a stronger argument against a bolt-on tracker than the compute cost.

### 3.8 Things in the proposal that are right

- Measuring before building. Keep that; the real-data report already covers
  half of it.
- Residual output with a hard bound (fix the bound as in §3.2).
- Payload randomized per episode, never mid-episode, which is exactly the
  paper's push-removal argument.
- Setpoint history in the state so a deterministic model can represent delay.
- Keeping a hand-tuned classical baseline as the sanity check.

## 4. Where the paper's ideas do help the NexArm stack

### 4.1 Cheap imports into the existing sim training (no world model needed)

- **Future-conditioned goal**: condition the tracker or policy on K = 10
  future reference frames rather than one. The action chunk is already this.
- **Return-based reset sampling** (paper, Appendix D.1): 50 % uniform, 40 %
  softmax over low-return 1 s bins, 10 % by past weight. Apply it to episode
  starts in the stack-bowls gym env so training concentrates on the grasp and
  release windows, where failures concentrate.
- **Asymmetric critic**: the critic sees the randomization vector (payload,
  delay, gripper compliance), the actor does not. This is standard, but the
  paper shows it is what makes short horizons work.

### 4.2 A real-data world model for the servo chain (the honest adaptation)

The paper's own limitation (§6) is that the model learns only from the
simulator. The NexArm is the opposite case: free-air arm motion is safe,
cheap and low-dimensional, so the world model can be fit from **real** logs.

1. Record 30 to 60 minutes of free-air chirps, random smooth splines and demo
   replays at 30 Hz with 0 / 100 / 250 g held in the gripper. Log commanded
   and measured positions only, which is all the bridge provides.
2. Fit a per-joint model: next position from the last 4 setpoints and last 4
   positions, payload as context. Start with a linear ARX; move to a small MLP
   only if the residual is structured.
3. Train the residual tracker by BPTT through this model (H about 16 to 32
   steps, critic optional at this horizon). This is SuperMimic's algorithm with
   the simulator replaced by the real robot, which is the direction the paper
   lists as future work and which RWM and DayDreamer already demonstrated.

This path has no sim-to-real gap for the actuator, needs no MJCF fix, and
still only pays off if §3.1 shows a problem worth solving. It is the version
of the proposal I would build, and only after the cheap lead-shift baseline
loses.

### 4.3 For the VLA itself: world-model RL fine-tuning is research, not plumbing

Using a learned differentiable world model to fine-tune an action-chunk head
on the stack-bowls task is attractive on paper: dense reward from bowl-pose
tracking against the demo, gradients through dynamics, no sim rollouts per
update. Two facts argue for caution:

- The paper removed pushes because its deterministic model cannot represent
  discontinuities. Grasp closure and release are discontinuities. Object
  interaction is explicitly listed as unsolved (§6).
- The 55 g bowl and a 72-tick compliant gripper make the contact dynamics
  dominated by the gripper's steady-state error, which the sim currently models
  as a near-rigid 116 N jaw (real-data report, Finding 3).

If pursued, scope it as: fixed grasp-assist in sim, a tracking reward on the
demo's bowl trajectory, a residual on the chunk only, and a small horizon.
Treat it as an experiment with a kill criterion, not as a deployment path.

### 4.4 Spend the three weeks where the failures are

The same budget (about three weeks in the proposal) buys more task success
from, in order:

1. the joint-mapping fix (phase 0), which unblocks everything sim-related;
2. a 2-frame lead shift in the rollout loop, and a tracker-on re-recording if
   it helps;
3. using the measured gripper position as a grasp detector in the policy
   observation (empty closed 2785 vs holding a bowl 2740 to 2760, real data),
   which turns the compliance "problem" into a free force sensor;
4. more demonstrations and action-chunk temporal ensembling.

## 5. Recommendation

- Do not build a SuperMimic-style simulated tracker for the NexArm. The
  paper's gains come from library scale, torque authority and dense rewards,
  none of which the arm has.
- Do the two measurements that remain open: a timed 2-frame lead-shift A/B on
  task success, and a free-air response fit per joint. Both reuse existing
  scripts (`examples/nexarm/audit_real_dataset.py`, `calibrate_sim.py`).
- Import return-based reset sampling and the K-frame future goal into the
  existing gym training now; they are a day of work and independent of any
  world model.
- Revisit a real-data world model (§4.2) only if the lead shift leaves a
  measurable tracking error at task speed.

## Unresolved questions

1. Do HX-30HM / HX-65HM servos expose writable position-loop gains, and does
   the AT32 firmware forward them in LeRobot mode? Nothing in this repo says so.
2. Does the AT32 apply the register-41 acceleration profile to CMD 97 writes
   at 30 Hz, or does a new goal every 33 ms override the profile? This decides
   whether a first-order or a profiled servo model is right.
3. Was the imitation policy's inference loop also lagged by 2 frames during
   evaluation, or only the recorded demos? The answer decides whether a lead
   shift helps or hurts before retraining.

## Update 2026-10-23: firmware deep dive

`plans/reports/research-261023-0808-nexarm-firmware-control-path.md` answers
unresolved questions 1 and 2 from the AT32 disassembly:

1. Servo gains are writable through CMD 59 (register 21 P/D/I block) and
   readable through CMD 60. In bridge mode the write reaches the AT32 but the
   readback reply is dropped by the ESP32 until `bridge_at32_callback` relays it.
2. CMD 97 is a direct sync write with acc 0 and speed 0. The AT32's 10 ms
   interpolator serves only coordinate moves and is silenced by the first
   CMD 96. `motion_speed` and `motion_acc` therefore have no effect on LeRobot
   writes, which is the mechanism behind Finding 1 of the real-data report.

This strengthens §3.1 and §3.6: the arm's dynamic response is a proportional
servo loop whose gain is a register, so the first "tracker" to try is a P/D
sweep with a step test, not a learned controller.
