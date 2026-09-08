# Real-robot readiness — Tim's two questions, answered (2026-08-10)

## Q1: Is our code compatible with controlling a real robot?

**Yes, structurally — and the harness now exists** (scripts/
trossen_real_executor.py, purely additive, sim paths untouched).

Why it composes cleanly:
- The planner (BiRegraspPlanner, unchanged) outputs Cartesian grasp links +
  metre targets; the twin already converts those to joint trajectories via
  real IK + velocity-limit chunking. The real arm consumes exactly that.
- Driver: `pip install trossen-arm` (libtrossen_arm, Python 3.10-3.13, no
  C++ needed). One TrossenArmDriver per arm (two IPs, factory 192.168.1.2/.3).
  Joint positions rad + gripper m (LAST slot, 7-dim); set_arm_positions /
  set_gripper_position with firmware quintic interpolation via goal_time;
  external_effort mode for force-closing on the rope (±20 N demo default).
  Driver+controller firmware versions must match.
- The harness reuses the twin's OWN kinematics (TrossenBimanualEnv IK/FK/
  capsule chains/reach map) as a "safety shadow" — the validated geometry,
  not a reimplementation.

What is still MISSING before first real motion (the honest list):
1. **Rope-state perception** — the planner needs 70 node positions per
   step. The harness takes a provider callback and hard-gates on staleness,
   but the perceptor itself does not exist. Biggest gap.
2. **Gripper unit calibration** — sim carriage q vs driver 0-0.04 m
   coordinate mapping is plausible-not-verified; real mode REFUSES to start
   until GRIPPER_CAL_CONFIRMED is set after a bench calibration.
3. **Base-frame calibration** — place both robot bases in the shadow world
   frame (table z=0, bases 92 cm apart on y). Startup probes FK-vs-driver
   cartesian and refuses on >8 mm disagreement, but someone must do the
   measurement. NOTE: the whole hand-B stack assumes the bases differ by
   pure TRANSLATION (no yaw) — if the real mount is rotated, stop and tell
   me; that needs code, not calibration.
4. **No hardware e-stop exists in the driver API** (verified) — the
   physical power switch is the e-stop; a human must be present. Firmware
   does idle (brake) the arms on connection loss and on any error.
5. Speed ramp protocol: SPEED_SCALE starts at 0.15 (15% of joint limits);
   raise only after supervised slow cycles.

## Q2: Safety — IK everywhere, arm-arm and table collision. THE PRIORITY.

Design: in sim every safety signal was telemetry; in the harness every one
is an interlock, fail-closed, LATCHED. Two chokepoints — check_leg() for
arm motion, _gripper() for jaws — and nothing reaches a driver except
through them.

Per interpolated sample (<= 2 deg joint motion between samples), on EVERY
leg including all flights and retreats (which the sim never checked):
- **IK acceptance** on every waypoint: residual <= 2 mm, phase-aware tilt
  (5 deg at grasp poses, 60 deg transit), joint-limit-clamped solutions are
  VETOED (sim accepted them), achieved-pose lowest-tip bound.
- **Arm-arm collision**: the twin's capsule-chain clearance (per-segment
  radii + 10 mm) between BOTH arms at every sample. The drag is checked
  exactly as executed (A moves with B frozen, then B with A at target).
- **Table**: distal-chain (elbow..wrist) z floor 5 mm ALWAYS — never
  disabled, per arm, with rise-grace; moving fingertip floor 20 mm in
  transit, −4 mm only on the intentional grasp descend/drag.
- **Rope**: chain-vs-rope clearance from a rope state RE-FETCHED before
  every leg group (25 mm for arm segments, 8 mm for the tool, exempt only
  while intentionally grasping/holding).
- **Branch-flip guard** (big joint move over small tool travel = vetoed),
  encoder-drift interlock at every cycle entry, stale/invalid rope state
  blocks motion, grasp verified by settled jaw readback (empty jaws latch a
  halt), 3 consecutive planner-infeasible targets latch a halt.
- **Halt protocol**: any veto or ANY exception idles both arms and latches;
  the only way back is operator clear_and_resync(ack) which re-reads
  encoders, restores gripper position mode, and re-checks frame sanity.
  Soft planner rejections (not vetoes) trigger a gated recovery to the
  overview pose so arms are never abandoned at the rope.

Process: the harness was built from a 3-agent research pass (driver docs,
sim execution contract, 20+-finding adversarial safety audit), then
verified by a 4-lens adversarial review (fail-open, state-tracking,
frames/units, driver API) that found 28 issues — including 3 real criticals
(gripper commands bypassing the latch; arm B's z-interlocks disabled during
drags; drag checked-vs-executed mismatch) — ALL fixed in the rewrite.
Dry run: 3/3 full cycles, 30 legs gated, latch + resync self-tests pass.

Bring-up checklist (in order): bench gripper calibration -> base-frame
measurement -> perception provider -> untethered dry cycles on the rig
(drivers connected, SPEED_SCALE 0.15, no rope) -> supervised rope cycles ->
speed ramp. A human at the power switch for all of it.
