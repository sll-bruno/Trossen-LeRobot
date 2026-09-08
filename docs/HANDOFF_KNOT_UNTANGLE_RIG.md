# Handoff: untangling a loose overhand knot on the rig

For the agent/operator on the Trossen bimanual rig.

> **Status: NOT runnable on hardware yet.** Tomorrow is a **perception-only**
> session — cameras on, arms unpowered. That session decides whether a knot run
> is possible at all, and it costs no arm motion. §1 says exactly what is
> missing. An earlier version of this file gave a `mpc_bimanual.py --arm`
> command as *the rig command*; that was wrong — `--arm` builds a kinematic arm
> inside the Newton **simulator** (`TrossenBimanualEnv` extends
> `gen.CollectorV2`; `link_positions()` reads `state_0.body_q`). It moves no
> hardware. Corrected 2026-09-01.

---

## 1. What is missing, precisely

**(a) No code path from the knot config to the hardware.** The rig is driven by
`mpc_s_run.py` in the separate `rope_rig` repo, which has no `--topo-w`, no
`--grasp-above`, no `--hand-z-max` and no LINE goal. §5 lists exactly what to
port; all of it is small and already written and tested on our side.

**(b) The rig has never demonstrated it can see a knot.** Measured over all
3,929 recorded real states:

| | value |
|---|---|
| states with EXACTLY equal z on all 70 nodes | **92.3%** |
| states containing any crossing | 0.5% |
| most crossings ever observed in one state | **1** (a knot needs 3) |
| of the 21 crossings on record, z-separation < 4 mm | **47.6%** (min 0.03 mm) |

The reconstruction projects onto the table plane, so "which strand is on top"
is usually not measured. That matters because it is the input to the anti-pinch
mask, and the failure was **silent**: with flat z, `graspable_from_above`
returned all-True and the guard that stops the gripper closing on two strands
disappeared with no error. Fixed — see §3.

**(c) Gripper calibration is not confirmed.** `GRIPPER_CAL_CONFIRMED = False`
in `scripts/trossen_real_executor.py:107`; a `--real` run raises `SystemExit`
at :867. It must be re-derived for the **5 mm** real rope, not the 10 mm sim
rope.

**(d) There is no hardware e-stop in the driver API** — the power switch is the
e-stop (`docs/robot/REAL_ROBOT_READINESS.md` item 4). A human stays on it for
any powered session.

## 2. Tomorrow: the perception session (no robot motion)

**Capture**, rig-side, arms unpowered, using the same state provider the rig
controller uses. Contract (`HANDOFF_ROBOT_AGENT.md` §3):
`provider() -> (xyz (70,3) float, metres, env-local, table-centre origin, z=0
at the table surface, ordered link 0..69)`.

```python
import numpy as np, time
xs = []
for _ in range(10):
    x, t = provider()                 # RE-POLL each time: a cached array
    xs.append(np.asarray(x, float))   # cannot reveal sensor noise
    time.sleep(1.5)                   # perception cycle ~1.3 s
np.savez("knotgate_place01.npz", states=np.stack(xs))
```

Do **3 placements minimum**, re-tying between them (§4).

**Score** (anywhere, no robot):

```bash
python scripts/knot_perception_gate.py knotgate_place01.npz
```

It scores with the *exact* functions the planner and the success metric use, so
a pass means the run's own machinery will behave. Three verdicts:

| verdict | meaning | what you may run |
|---|---|---|
| **NO-GO** | the chain itself is unreliable — arc length unstable, nodes sliding, or crossings not stably 3 | nothing; fix perception |
| **GO (XY-ONLY)** | chain good, height unusable | the full task, with the conservative anti-pinch fallback; success scored as crossing **count → 0** |
| **GO (FULL)** | height usable too | as above plus signed over/under topology |

Expect XY-ONLY. That is **not** a blocker: the planner's topology cost is
computed from xy segment intersections and never reads z.

## 3. What was fixed so this is runnable in XY-ONLY mode

`graspable_from_above` now detects a degenerate-z state (z-spread < 3 mm) and
switches to a conservative rule: **if two non-adjacent strands overlap in xy at
all, neither is graspable** — because we cannot tell which is on top. Strictly
safer than the z-aware rule, never less safe. Measured on a gated sim knot:

| state | graspable links |
|---|---|
| real z | 60/70 |
| flat z, old behaviour | 70/70 — mask silently gone |
| flat z, new fallback | **46/70** |

The planner still has 46 links to work with, so the task remains feasible while
the pinch guard stays real.

## 4. Tying and placing the knot

1. **A single overhand knot, left loose** — loop open enough to lay a finger
   through, roughly 6–8 cm.
2. **Flat on the table**, all three crossings visible from above, no strand on
   edge. This is what gives the extractor a chance.
3. **Both tails free** and roughly straight, pointing away from the knot.
4. **Centred in the shared workspace** — only ~3% of the table is reachable by
   both arms.
5. **Note which tail is free** (the one not threaded through the loop). The
   mechanics are asymmetric: pulling the wrong end cinches the knot.
6. **Same chirality every run.**

## 5. To make a robot run possible (rig-side port)

Port into `mpc_s_run.py`, in this order. All of it exists and is tested here:

1. **LINE goal** — straight rope through the table centre along x. Take the
   goal height from the **observed** rope, not a constant: the stored planar
   goals sit at the sim table height (6.0 mm) while the real rope rests at
   10.8 mm (measured over 400 rig states), and that 4.8 mm offset is an
   unreachable 2.8 mm RMSE floor on a task whose threshold is 25 mm. Reference:
   `scripts/mpc_bimanual.py` `_line_goal()` and the `args.arm` correction.
2. **Topology cost** — `BiPlanner._topo_cost` / `_score_xyz`. xy-only, so it
   needs no height. Weight `topo_w = 1e-2`; the measured cinch-vs-untangle flip
   threshold is 1.7e-3–2.6e-3, so 1e-2 is comfortably above it. Weight 0
   reproduces the existing cost exactly.
3. **Anti-pinch mask** — `graspable_from_above` (§3), ANDed into the regrasp
   candidate set **and** the initial grasp.
4. **Drag height cap** — `hand_z_max = 0.008`. Load-bearing: the default
   post-regrasp height (~35 mm) lifts the tip and folds the tail over the loop
   instead of unthreading, a measured failure mode.
5. **Logging** — per-step crossing count, per-step state, executed action,
   grasp indices. Without these the run cannot be analysed the way the sim runs
   were.

Settings from the sim campaign: `H=8`, regrasp every 5 steps, magnitude ladder
30/20/10/5 mm, 60 control steps. Note `H=1` wins on every *other* task we run;
`H=8` here is because the topology term needs lookahead, and it is untested
against `H=1` on the rig — that is the first ablation once anything works.

## 6. What to expect, honestly

**The simulator's biggest caveat does not apply to a real rope.** About half our
simulated runs let rope pass *through* itself, and ~40% of solved sim runs
jumped from ≥2 crossings to 0 in one step. Real rope cannot do either — so if
perception holds up, a rig run is **better evidence than the entire simulation
campaign**.

**Do not expect a model-quality result.** In sim the task did not separate the
models: GA-Net 30/30, ours 28/30, LSTM-GCN 28/30 — and LSTM-GCN is worse than
predicting no movement on real-rope prediction. The topology term plus the
search does the work, not the dynamics model. A successful run demonstrates
**the system**.

**Throughput.** ~25 s of robot time per control step, 60 steps ⇒ ~25 min per
run plus tying and gating. A realistic powered day is **8–12 runs**. Sim solves
in a median of 8 steps, so stop a run once crossings reach 0 and the rope is
extended rather than burning all 60.

## 7. Abort immediately if

* the gate returns NO-GO, or crossing count jumps around with the rope visibly
  still;
* arc length drops below ~660 mm mid-run (the extractor loses rope at
  self-overlaps — exactly where a knot lives);
* the arms load against a snagged rope;
* the rope leaves the table or is dragged over an arm base;
* the run was started without the observed-height goal correction.
