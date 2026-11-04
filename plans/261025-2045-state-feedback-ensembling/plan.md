# Plan: state-feedback ensembling (idea 1 of the 20:28 report), v0 research code

Status: v0 done in simulation, 2026-10-25; arm rollout not started. Results are in the new repo's README.

Outcome of v0, which supersedes decision 1's choice of method below:

- The method that works is the **integrator**: Median TE plus a servo-model disturbance observer on the gravity-loaded joints (shoulder_lift, elbow_flex, wrist_flex).
- In the fitted sim it cuts error while holding from 3.09 to 1.71 ticks for a constant sag the size of the bowl's measured median shift (8 ticks; the real holding error is pose-dependent with a wider spread), and from 20.6 to 6.9 ticks for a hypothetical 60-tick sag. Its cost in free motion is 0.06 ticks.
- It does nothing for perception-side common-mode error, which matches the Median TE theorem.
- Age-selective offsetting equals the integrator by construction. Switching estimates on the measured contact mode is worse. Ghost alignment hurts under payload.
- Candidate-aware variants remain open only if candidates disagree on contact timing. Code lives in a new local repo, `C:\Users\Administrator\code\state-feedback-ensembling` (package `sfe`). This plan stays in the NexArm repo because it records the design decisions; the new repo's README describes usage.

## Outcome

A training-free executor for action-chunk policies that uses the measured joint state to remove the part of the tracking error that every overlapping candidate shares and that no candidate-only aggregator can remove (Median TE, 2609.27167, Theorem 1 and Corollary 2). v0 must show, in a simulated servo plant fitted to the real NexArm data, whether that executor beats both robust temporal ensembling and a plain command-offset integrator.

## Design decisions (after the advisor review of 2026-10-25)

1. **What the state adds.** For an executed tick, a candidate's prefix error splits into three terms. The first is a plant disturbance: the measured state minus the lag-model response to the command actually sent, which is identical for every candidate. The second is a prediction-space term, the candidate minus the executed aggregate. The third is a deterministic lag term. Only the first comes from the sensor.
   - The "state-weighted median" variant is dropped: its weight could only depend on the prediction-space term, so it needs no sensor and inherits the blind spot.
   - The plain "offset all candidates" variant equals adding one offset after a translation-equivariant aggregator, so it is a disturbance observer on the command. It is kept as the **integrator baseline**, not as the method.
   - The method is **age-selective offsetting**: candidate i, issued at tick t_i, gets offset `d(now) - kappa * d(t_i)`, where `d` is the disturbance estimate and `kappa` in [0, 1] is how much of a disturbance a policy corrects from what it observed. With kappa 0 it reduces to the integrator; with kappa above 0 the offsets differ by candidate, so the candidate set matters. Whether a real policy self-corrects for sag is one of the experiments (sweep kappa in the sim, estimate it on the arm).
2. **Lag model.** Per joint: integer command-to-state delay `d`, proportional gain `alpha`, velocity saturation `vmax`. They are fitted by least squares on the full-length episodes of `thanhkt/nexarm_stack_bowls`, per episode, and the distribution is reported (2 frames typical, 0 in episodes 69 to 92 per the hardware notes).
3. **Disturbance observer.** Inverting the model for one step, `g = (c[t-d] - q[t]) - (q[t+1] - q[t]) / alpha` estimates the steady offset. It is skipped when the joint is saturated and filtered with an EMA.
4. **Gate.** The servos read load but CMD 96 relays only position, so the default gate is quasi-static: update the estimate only while the command moves slowly. A load gate takes a per-joint scalar so a firmware change can plug in later.
5. **Gripper excluded.** The closed gripper's persistent error is the grasp (commanded 2833, settles at 2740 to 2785), so it is never compensated.
6. **Plant sim.** A proportional loop with delay, velocity saturation, a payload-dependent sag term, noise and integer quantization. The synthetic policy has per-candidate noise, sparse corrupted queries and a self-correction gain. An optional common-mode perception shift is the negative control: no method should fix it.

## Scope v0

The scope is:

- the candidate cache;
- aggregators: latest, ACT exponential mean, median;
- UAD-style ghost correction;
- the integrator baseline and age-selective offsetting;
- the lag fitter, disturbance observer and gates;
- the plant sim;
- the benchmark and dataset-fitter CLIs;
- a LeRobot adapter with duck-typed pre- and post-processors (`temporal_ensemble_coeff=None`, so the cache owns aggregation);
- tests and a README.

Deferred until the age-selective variant beats the integrator: the mode guard (idea 10) and the tracking-error replan trigger (idea 11).

## Acceptance

- `uv run pytest` passes in the new repo.
- `sfe-fit-lag` runs on the cached `thanhkt/nexarm_stack_bowls` and reports per-joint delay, gain and saturation.
- `sfe-bench` prints a table with seeds and intervals. Its conditions are clean, payload sag, sparse corruption, both, and a common-mode perception shift, and it covers every method.

## Risks

- The effect on a quasi-static task may be small.
- The sim result is only as good as the fitted plant.
- Kappa is unknown for real policies.
- Nothing here has run on the arm.
