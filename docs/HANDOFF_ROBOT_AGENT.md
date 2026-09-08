# Handoff: running our MPC on the real dual-arm rig

For the agent/operator working on the physical Trossen WidowX AI rig. You
have no conversation context; this file is self-contained. Everything here
is measured unless marked ASSUMPTION.

**READ FIRST: the system cannot run end-to-end today.** Four blockers, two
of them hard. §1 says what they are. Do not skip to §4.

**Safety framing:** the arms have no e-stop in the driver API (verified in
Trossen's docs). The physical power switch is the e-stop. A human stays
within reach of it for every motion, at every stage, no exceptions.

---

## 1. What is missing before a real MPC run (in dependency order)

| # | Blocker | Status | Who/what fixes it |
|---|---|---|---|
| 1 | **No planner→hardware loop.** `scripts/trossen_real_executor.py` is an *executor*: it takes grasp links + Cartesian targets and runs one safe pick-place cycle. It contains no model, no planner, no goal. | HARD | Write the outer loop (§5). ~a day. |
| 2 | **No rope perception.** The planner needs 70 node positions per control step. Nothing produces them on the real rig. | HARD | Separate project. Spec in §3. |
| 3 | **Gripper units uncalibrated.** `GRIPPER_CAL_CONFIRMED = False` in the executor — real mode *refuses to start*. The sim jaw-carriage value and the driver's 0–0.04 m gripper coordinate are different units and the mapping is a guess. | 1 bench hour | §4 step 2. |
| 4 | **Base-frame calibration.** The safety shadow assumes table z=0 and bases 92 cm apart on y. Startup hard-refuses on >8 mm FK-vs-driver disagreement. | 1 rig hour | §4 step 3. |

Also true, and important:

- **`pip install trossen-arm` has not been run anywhere.** Verified: import fails.
- **The deployment-config planner produces flight-leg collisions.** A
  24.5k-chunk audit of three deployment-config episodes found **92 arm-arm
  contact episodes, 100% of them originating in flight legs** (82 approach,
  10 retreat; min penetration −65.5 mm), and **0** in guard-checked drags.
  The twin does not check flight legs; the real executor checks *every*
  leg and **latches a halt**. So the twin's "clean" episodes will halt
  repeatedly on hardware. This is not a harness bug — the harness is
  correctly catching motions the twin let through. Expect it, and see §6.
- **Sim/real execution mismatch:** the twin moves both arms simultaneously
  during a drag; the executor moves them sequentially (A, then B), because
  that is what it can safely verify. The model was trained on simultaneous
  motion.
- **Speed:** at the bring-up speed (`SPEED_SCALE = 0.15`) one cycle costs
  ~14.1 s of commanded motion vs the twin's 4.7–4.9 s/step.

---

## 2a. UPDATE 2026-08-25 — deploy THIS model instead, `deploy/ft1step_rig_s42/`

Everything in the original §2/§3 below was measured on a model trained
before the rig-matched dataset existed. That model's own §3 says its
mag ladder operates it **out-of-distribution above 22mm** because the old
training set's drags capped there. This is now fixed at the source: the
new model trained on `rope_bimanual_rig.npz`, whose action distribution
was built by measuring the real rig's own executed drags (`U(5,30)mm`,
20/40/40 bimanual/A/B mode split, one grasp-lift-drag-release cycle per
step, regrasp every step) instead of a scripted scenario mixture. The
30mm ceiling now matches the rig's own maximum commanded drag, not an
arbitrary sim choice.

**Bundle**, self-contained, no cluster access needed:
```
deploy/ft1step_rig_s42/model_config.json   architecture
deploy/ft1step_rig_s42/norm_stats.pt       state normalisation, REQUIRED
deploy/ft1step_rig_s42/bi_best.pth         weights, 1.49 MB, md5 0bc2ab2491890cee62a339d985dd391c
deploy/ft1step_rig_s42/checkpoints/bi_best.pth   same file, results-dir
                                             layout for scripts/mpc_bimanual.py
deploy/mpc_targets.npz                     goal shapes S / U / Z / amp / knot / J
```
Both checkpoint copies are md5-identical; the duplication exists because
`scripts/eval_real_chain.py` and `scripts/mpc_bimanual.py` expect two
different directory layouts and neither should be edited to match the
other.

**What it is:** field arch, `enc_layers=0`, `act_mode=raw`,
`d_rnn=96 d_z=32 d_hidden=160 d_node=160`, **369,973 params** — 18x
smaller than the old 6.76M model. Trained to convergence on the rig
dataset, then fine-tuned 12 epochs at `pred_weight=5, beta_kl=0.1,
wm_lr=1e-4` (the "1-step" recipe; an inextensibility-penalty variant was
also tried and made no measurable difference, so it was dropped).
`inext_lambda: 0.0` in this checkpoint's config confirms it is the plain
fine-tune, not that variant.

**Verified 2026-08-25, fresh process, bundle path only** (do not take this
on faith — rerun it if you touch the bundle):
```
python scripts/eval_real_rig.py --data-dir <real-rope-branch checkout> \
    --model field --results deploy/ft1step_rig_s42 --out-json /tmp/v.json \
    --warmup 1 --horizon 55 --usable-only
# -> pooled, seg=79: t1=10.67 t2=12.81 t3=14.93 t5=16.21 t10=20.00 t20=21.96 mm
python scripts/mpc_bimanual.py --model-dir deploy/ft1step_rig_s42 \
    --goal-shapes S --envs 3 --control-steps 5 --horizon 1 \
    --mags 0.05 0.025 0.01 0.005 --regrasp --seed 42 --out /tmp/mpc_smoke
# -> ran to completion, no shape/load errors
```

**Sim numbers** (held-out test split, in-distribution, 3 seeds, this is
seed 42 specifically — s43/s44 tie within 0.1mm):
| | t=1 | t=10 | t=50 |
|---|---|---|---|
| open-loop RMSE | 2.7mm | 6.4mm | 6.2mm |
| gauss topology | | | 90.8% |

**Sim closed-loop MPC**, same planner family as §3 below (H=1 greedy,
`--mags 0.05 0.025 0.01 0.005`, regrasp on), 20 independent ropes/shape,
30 control steps:
| shape | best RMSE | final RMSE | hit-30mm | steps→30mm |
|---|---|---|---|---|
| S | 24.4mm | 28.0mm | 14/20 | 12 |
| U | 24.7mm | 27.5mm | 17/20 | 19 |
| J | 25.8mm | 29.4mm | 16/20 | 17 |

Beats every trained baseline tried the same way (GA-Net, IN-BiLSTM 1.16M
and 0.61M, LSTM-GCN 1.38M and 0.53M) on final RMSE, on every shape.

**Real rope**, teacher-forced-off, no warmup, 32 usable runs from
`real-rope-data-2026-08-20` (1758/1760 tuples kept), chains cut at
re-observations:
| | t=1 | t=10 | tail (t≈40-55, n=9-13, thin) |
|---|---|---|---|
| pooled | 10.7mm | 20.0mm | ~19-22mm, best of everything tried |

At short horizon (t≤10) two baselines (GA-Net, IN-BiLSTM) are clearly
**better** than this model on real rope (≈14.5mm vs ≈20mm) — open-loop
one-step accuracy does not carry over from sim to real for this model,
only the long-horizon non-compounding behaviour does. Worth knowing
before assuming this is a strict win.

**The uncomfortable finding, stated plainly:** a zero-parameter kinematic
stub (grasp node follows its commanded target, `exp(-d/5)` falloff, no
rope model at all) scores **23.2 / 26.9 / 25.0mm final RMSE** on S/U/J
under the identical MPC harness — matching or beating every trained model
including this one. H=1 greedy replanning with frequent regrasp is close
to a grasp-following control problem; the learned dynamics model's value
shows up in *how quickly* it converges and its behaviour under harder
conditions, not in a clear final-accuracy win over the free baseline.
Confirm this holds (or doesn't) on hardware before concluding the model
is pulling its weight — it may be worth running the kinstub controller on
the rig too, since it costs nothing to implement and is the honest
baseline to beat.

**Not yet done, flag if it matters before you commit to this ladder:**
the `--mags 0.05 0.025 0.01 0.005` / regrasp-every-N sweep that produced
§3's detailed table for the OLD model has not been repeated for this one.
The numbers above use one planner config chosen for a fair cross-model
comparison, not one tuned specifically for this checkpoint. GA-Net seed
44's MPC run and a same-magnitude size-ablation MPC pass are still
in flight as of this writing — check `mpc_out/ganet_s44_*` before citing
GA-Net as a settled loss.

---

## 2. The model to deploy (SUPERSEDED by §2a above — kept for the record)

**`deploy/model/` — the bimanual field model, enc0 + act-raw, 6.76M params.**
This is the default as of 2026-08-11 (Tim's call). Everything the rollout
needs is in this repo; you do not need cluster access.

```
deploy/model/model_config.json   architecture (loads via BiHybridDreamer)
deploy/model/norm_stats.pt       state normalisation (mean/std) — REQUIRED
deploy/model/bi_best.pth         weights, 27 MB, md5 1af8e62c248cc88f7b463302d88431bc
deploy/mpc_targets.npz           goal shapes S / U / Z
```

| | this model | previous champion (v3_fix) |
|---|---|---|
| params / inference | **6.76M / 1.88 ms** | 14.1M / 2.69 ms |
| open-loop t=50 | **19.1 mm** | 20.3 mm |
| topology (gauss) | **49.9%** | 46.3% |

Do NOT hand-edit `model_config.json`. It still records `d_embed: 1024` and
`d_action: 1024`; the constructor recomputes them to 210 and 14 from
`enc_layers: 0` / `act_mode: raw`. Loading is correct — the file just reads
wrong. (Verified: loads to 6.76M params with a zero-parameter action
encoder.)

**Honest status of this choice.** In the twin, at the ladder tuned for the
OLD model, this model was slower to converge (best-so-far 10.9 vs 9.3 mm;
59 vs 37 steps to 30 mm, two seeds each). Given its own coarser ladder
(80/40/20/10 mm) it improves to 8.9 mm / 39 steps. It is faster and half
the size, and offline it is better at every horizon and better on topology.
Per-magnitude accuracy (scripts/mag_stratified_eval.py) shows why: it is
the better model below ~65 mm commanded motion and the weaker one above.
Ladder cells at 40/60/120 mm were still running when this was written —
if you need the last word, check docs/results/results_full_cycle_timing.md.

**Do not judge runs by final RMSE.** Two identical runs of the SAME model
and config differed by 8.7 mm on final@200 (16.8 vs 25.5) while agreeing to
0.5 mm on best-so-far and 1 step on steps-to-threshold. Episodes peak
40-80% through and then random-walk upward. Use best-so-far and
steps-to-threshold. On 2-seed means the two models' final RMSE is
identical (22.6 vs 22.6) while the stable metrics separate — final RMSE
carries no signal here.

## 3. The planner configuration (do not re-derive; these were measured)

> **How the search actually works — the atoms, the sweep schedule, every cost
> term, the arc guard, the execution rule — is specified in full in
> `docs/robot/HANDOFF_CD_PLANNER.md`. Read that before changing anything here.**
> It is the same code path for every model we compare, which is the only reason
> a rig comparison is a model comparison. Import `BiRegraspPlanner`; do not
> reimplement the search.
>
> Note the values below are the pre-2026-08-25 ones for the `rssm5_v3` model.
> For the deployed `ft1step_rig_s42` bundle (§2a) use the CD-planner doc's §6:
> `--mags 0.05 0.025 0.01 0.005`, `--regrasp-every 1`, `--control-steps 50`,
> `--stop-mm 20`.

```
model      : BiHybridDreamer, bimanual, n_hands=2
planner    : BiRegraspPlanner (scripts/mpc_bimanual.py)
horizon H  : 1                       # H>=5 costs ~5 mm; H=2 ties
mags       : 0.08 0.04 0.02 0.01     # for THIS model (enc0+act-raw).
                                     # 0.05 0.025 0.01 0.005 is the old
                                     # model's optimum; the fine ladder
                                     # 0.02/0.01/0.005/0.0025 is CATASTROPHIC
                                     # here (32 mm) — see below
headings   : 16  (+ explicit hold)   # 65 moves/hand
regrasp    : ON, every 5 steps       # cadence 10 is worse
link-stride: 2,  min-sep 10 links
cd-rounds  : 2   (+ A-only / B-only family sweeps)
penalties  : graded (10 + 100*violation) — NEVER 1e9 (float32 saturates)
arc_guard  : ON
```

Exhaustive joint search (`--exhaustive-joint`) buys ~0.5 mm over CD (16.9 vs
17.4 mm final, 16.0 vs 16.6 best-so-far, 150 matched runs) at **187×** the
planning cost — 11,229 vs 60 ms/env-step, re-measured 2026-08-26 under the
fixed executor. **Use CD on hardware.** Planning is ~132 ms/env-step there,
irrelevant next to robot motion.

**Why the ladder matters more than anything else you could tune.** Measured
1-step prediction RMSE vs a persistence baseline ("nothing moves"), by
commanded magnitude (scripts/mag_stratified_eval.py, 60 trials):

| move | persistence | this model | skill |
|---|---|---|---|
| 2.5 mm | 1.19 | 1.13 | 0.05 |
| 5 mm | 2.03 | 1.65 | 0.18 |
| 10 mm | 3.89 | 2.85 | 0.27 |
| 25 mm | 10.21 | 6.51 | 0.36 |
| 50 mm | 23.86 | 16.12 | 0.32 |
| 80 mm | 44.01 | 33.04 | 0.25 |

Below ~5 mm the model is barely better than predicting no motion at all, so
a ladder with fine rungs makes the planner choose between candidates it
cannot tell apart. Note also that everything above 22 mm is OUTSIDE the
training distribution (the collector capped steps at 22 mm), so the
deployment ladder operates the model out-of-distribution by design — it
works, but it is extrapolation.

**Belief-state hygiene (measured 2026-08-16, do this in the outer loop):**
the sim planner carries the model's recurrent state and tells it the
PLANNED action happened, even when it did not (refused grasp, veto). In the
twin this costs only +8% 1-step error (10.30 vs 9.54 mm, RECOD 82548)
because 94% of actions execute as planned — but on a real rig failures are
frequent, and a sister project measured the same fiction DOUBLING model
error (48.5 vs 28.2 mm) with gain miscalibrated to 0.29. Your outer loop
knows ground truth (the executor reports empty grasps and halts): feed the
model the EXECUTED action with honest held flags, never the plan.

**Perception spec, measured:** injecting i.i.d. Gaussian noise into the
rope state and running the executor's own gated cycle, **5 mm per-node
noise already latches a halt on the first leg** (tool-vs-rope clearance);
10 mm halts on cycle 0. So the estimator needs **≲5 mm per-node accuracy**,
plus timestamps — the executor rejects state older than 1.5 s.

State contract: `provider() -> (xyz, t_monotonic)` with `xyz` shape
`(70, 3)`, float, **metres, env-local frame** (table centre origin, z=0 at
the table surface), ordered from link 0 to 69 along the rope.

## 4. Bring-up sequence — do these in order, do not skip

**Step 0 — no robot.** `pip install trossen-arm`; confirm firmware version
matches the driver. Run `python scripts/trossen_real_executor.py --dry-run
--steps 3`. Expect: 3/3 cycles, ~30 legs gated, latch and resync
self-tests pass. If this fails, stop.

**Step 1 — telemetry only, arms powered, nothing commanded.** Connect both
IPs, read encoders at 10 Hz for a minute. Confirm joint ordering (6 arm
joints in rad, gripper LAST in m) and that both arms report plausible
poses. No motion.

**Step 2 — gripper calibration (fixes blocker 3).** With the arm parked
safely, command a few gripper positions and *measure the physical pad gap*
with calipers. Fit the mapping, update the constants at the top of
`trossen_real_executor.py`, then set `GRIPPER_CAL_CONFIRMED = True`.
Getting this wrong is asymmetric: one plausible reading silently disables
the empty-grasp interlock, another halts on every grasp. Measure, do not
assume.

**Step 3 — base-frame calibration (fixes blocker 4).** Measure both robot
bases into the shadow frame (table z=0, bases 92 cm apart on y, table
92×72 cm). Then run the executor's startup: it FKs the live encoders and
refuses if fingertips compute below the table or if driver-cartesian and
shadow FK disagree by >8 mm. That refusal is the check — treat a pass as
the calibration being right, not as a formality.

**Step 4 — empty table, no rope, no perception, no model.** Drive
`RealHarness.run_cycle()` with hand-picked links/targets and a static rope
provider. `SPEED_SCALE = 0.15`. This is the first real motion. Watch the
arm-arm margin: on nominal twin cycles the tightest point clears by only
**1.1 mm**, so expect halts and treat every one as informative.

**Step 5 — rope, still no model.** Replay a recorded twin trajectory
(`mpc_bimanual.py --save-traj` dumps `traj_env0.npz`). This gives a real
sim-to-real number without any perception stack: commanded-vs-achieved rope
shape, per cycle.

**Step 6 — closed loop.** Only once perception exists and steps 0–5 pass.

## 5. The outer loop you have to write (blocker 1)

The missing glue, roughly 100 lines:

```
load model + norm_stats  ->  BiHybridDreamer (eval, cuda)
build BiRegraspPlanner with the §3 config
goal = 70x3 target shape, env-local metres

h, z = zeros
loop:
    xyz, t = provider()                    # freshness gated by executor
    s_n = (flatten(xyz) - mean) / std
    a_emb = model.action_encoder(a_prev)
    h, z, post, _ = model.rssm.observe(h, z, a_emb, model.encoder(s_n))
    z = post.mean
    tgtA, tgtB, _, linkA, linkB = planner.plan(h, z, s_n, handA, handB,
                                               linkA, linkB, goal,
                                               xyz_obs=xyz,
                                               allow_switch=(step % 5 == 0),
                                               link_ok_a=..., link_ok_b=...)
    harness.run_cycle(linkA, linkB, tgtA + offsets, tgtB + offsets)
    a_prev = [tgtA, linkA, tgtB, linkB, 1, 1, cfg]
```

Three things that will bite:

1. **Frames.** The planner works in env-local metres; the executor wants
   world. The shadow env's `offsets[0]` is the conversion. Get this wrong
   and the arms move to the wrong place with every safety check passing.
2. **`a_prev` held flags are always 1** — the model is told both hands held
   even when a grasp was refused or a leg halted. Known fiction, benign at
   H=1 in sim, unvalidated on hardware. If you can, feed the real outcome.
3. **`run_cycle` returns False on a soft rejection** (infeasible target) and
   parks the arms at overview. That is not an error — replan and continue.
   A `SafetyHalt` exception is different: see §6.

## 6. When it halts (it will)

Any veto **latches**: both arms idle in place, and every further motion
request raises until an operator calls
`harness.clear_and_resync(<exact reason string>)`, which re-reads encoders
and restores gripper mode. This is deliberate — the system fails closed.

Most likely halt, by far: **`fly-*: arm-arm clearance violated`** or
**`tool-vs-rope clearance`** during a flight leg. Cause is known (see §1):
the planner was tuned in a twin that never checked flight legs. Options,
in order of preference:

1. Raise the planner-tier `HAND_MIN_DIST` (currently 0.13 m) so the
   planner stops proposing grasp pairs whose flights collide.
2. Back-port the per-leg collision gate into `trossen_collector.py`
   (`_walk_leg`, `_walk_leg_b`) and re-tune in the twin — the right fix,
   a day of work, and it makes the twin honest.
3. **Do not** loosen the executor's gate. It is the only thing standing
   between a planner artifact and a broken arm.

## 7. Numbers to expect (twin, 15 envs, 200 steps, full-cycle)

- final RMSE ~16.8 mm (CD + coarse ladder), best-so-far ~9.8 mm
- ~4.9 s/step robot time in the twin; **~14.1 s/cycle at bring-up speed**
- 85% of robot motion is ferrying to/from the overview pose; drags are ~7%
- episodes peak around 40–80% of the way through and then drift *worse* —
  best-so-far is the honest metric. A predicted-gain stopping rule is the
  known fix and is **not implemented**; on hardware, consider stopping
  manually when RMSE stops improving.

## 8. Files

| path | what |
|---|---|
| `deploy/model/` | **the model to run** (weights + norm stats + config) |
| `deploy/mpc_targets.npz` | goal shapes S/U/Z |
| `scripts/trossen_real_executor.py` | executor + safety layer (the thing you run) |
| `scripts/mpc_bimanual.py` | planner (`BiRegraspPlanner`) |
| `scripts/bimanual_variant.py` | model (`BiHybridDreamer`) |
| `simulation/trossen_collector.py` | twin + kinematics the shadow reuses |
| `docs/robot/REAL_ROBOT_READINESS.md` | safety design + audit history |
| `docs/handoffs/HANDOFF_ARCH_2026-08-10.md` | architecture and planner verdicts |
| `docs/audits/AUDIT_2026-08-11.md` | open weaknesses, verified |

All of these ARE in the repo as of 2026-08-11 (they were previously
untracked). `.gitignore` excludes the ~43 GB of datasets, training
checkpoints and rendered media, and explicitly whitelists everything a
rollout needs: `deploy/**` and `simulation/assets/**`.

Dependencies not in the repo: `pip install trossen-arm` (driver), plus
`torch`, `warp-lang` and `newton` (the simulator, needed because the safety
shadow reuses its IK/FK and collision geometry). The rollout does NOT need
any dataset.

Loading the model:
```python
import json, torch
from bimanual_variant import BiHybridDreamer
cfg = json.load(open("deploy/model/model_config.json"))
model = BiHybridDreamer(cfg)                       # rebuilds enc0 + act-raw
model.load_state_dict(torch.load("deploy/model/bi_best.pth", map_location="cpu"))
model.eval().cuda()
stats = torch.load("deploy/model/norm_stats.pt")   # {"mean":..., "std":...}
```
