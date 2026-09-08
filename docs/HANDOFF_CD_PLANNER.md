# The CD planner — exact specification

For the agent/operator on the physical Trossen rig. Self-contained; assumes no
conversation context.

This is the planner every one of our MPC numbers comes from — ours, GA-Net,
IN-BiLSTM, LSTM-GCN, the kinematic stub. **It is model-agnostic on purpose.**
Only the one-step rollout primitive (`_rollout_norm` / `_cost_chunk`) differs
between models; the search, the candidate set, the cost, the penalties and the
execution rule are byte-for-byte identical. That is what makes a model
comparison a model comparison and not a planner comparison.

**So: do not reimplement it.** Import it. If you write your own search on the
rig, the rig numbers stop being comparable to any sim number we have, and we
lose the whole closed-loop half of the paper.

```python
import sys; sys.path.insert(0, "scripts")
from mpc_bimanual import BiRegraspPlanner          # ours (field model)
from mpc_bimanual import GANetBiRegraspPlanner     # GA-Net
from mpc_bimanual import KinStubBiRegraspPlanner   # zero-parameter control
```

The rest of this document is the spec, so you can tell when something has
drifted — not an invitation to rebuild it.

---

## 1. Why coordinate descent at all

Each hand picks `(grasp link, heading, magnitude)`. Per hand that is 35 links ×
(16 headings × 4 magnitudes + hold) = 35 × 65 = 2,275 atoms. The **joint**
two-hand space is 2,275² ≈ 5.2M candidates, each of which needs a model
rollout. Not tractable per control step, so we optimise one hand at a time
with the other hand's move held fixed.

CD is an approximation. We measured what it loses, matched at 50 envs × S/U/J
= 150 runs under the **fixed executor** (`mpc_fix/`, 2026-08-26): exhaustive
joint search over the full product is 16.9 mm final / 16.0 mm best-so-far vs
CD's 17.4 / 16.6, and reaches the 20 mm goal in a median 8 steps vs CD's 10 --
for **187× the planning cost** (11,229 vs 60 ms per env-step). Under any
wall-clock budget CD wins.

> The older "15.1 vs 21.8 mm at 19×" figure came from the broken executor
> (lift-only on grasp switches) and is void. The direction survived the fix;
> the magnitude did not -- exhaustive is worth ~0.5 mm, not ~6.7 mm.

## 2. The candidate set (per hand, per sweep)

**Links** — `cand_links = np.arange(0, 70, link_stride)` with `link_stride=2`,
so 35 of the 70 nodes, then filtered by:

- `|link − link_other| >= min_sep` with `min_sep=10`. Both grippers cannot sit
  on nearly the same piece of rope.
- the **reach mask** `link_ok[link]`, if a reach map is supplied: this hand may
  only grasp nodes inside its own arm's workspace. With the bases straddling
  the rope each arm owns roughly half of it, so the hands partition the rope by
  geometry rather than by a rule we invented. **Use this on the rig.**
- If the filters empty the set, it falls back to `[current_link]`.

**Moves** — 16 headings uniformly on `[0, 2π)` × 4 magnitudes
`{50, 25, 10, 5} mm` = 64, **plus the zero move**, = 65.

The zero move is not decoration. It makes "only A moves", "only B moves" and
"both hold" first-class candidates scored on cost like anything else. Holding
is in-distribution, costs no robot time for that arm and carries zero collision
risk for it. Do not drop it.

**Regrasp base point.** If a candidate's link differs from the hand's current
link, the hand's base position becomes `xyz_obs[new_link]`, with
`z ← clip(z + 30 mm, 6 mm, 75 mm)` — i.e. a hop over the rope. Otherwise the
base is the hand's current position. Targets are then the base plus `k+1` copies
of the 2-D move over the horizon, with `z` clipped to `[6, 75] mm` every step.

> The 6 mm floor matters. The original 30 mm floor was copied from the
> unimanual harness, which lifts before control; the bimanual harness never
> lifts, so grasped links sit at the table height and a 30 mm floor silently
> commanded a 24 mm vertical jump on control step 1 — alone beyond the
> dataset's 22 mm/frame cap.

## 3. The sweep schedule — exactly what runs, in order

With `rounds = 2` (`--cd-rounds`):

```
for r in 1..rounds:
    sweep A   with B's move fixed  (B starts at HOLD)
    sweep B   with A's move fixed
# then, unconditionally, both single-arm families:
sweep A   with B holding its CURRENT grasp, B's move = 0
sweep B   with A holding its CURRENT grasp, A's move = 0
# then take the global argmin of the three costs:
#   cJ  = joint  (result of the last B sweep)
#   cA1 = A alone
#   cB1 = B alone
```

That is **6 sweeps** per control step. The two single-arm sweeps are not
redundant with the loop: CD only scores "B moves alone" when A's chosen move
*happens* to be hold, so without them the single-arm families are reachable only
by accident. Both families are evaluated every step and the cheapest of the
three wins.

Tie-break, verbatim: `cA1` wins if `cA1 < cJ and cA1 <= cB1`; else `cB1` wins if
`cB1 < cJ`; else the joint result stands.

**Candidate volume** (model-scored rollouts per control step):

| | per sweep | × 6 sweeps |
|---|---|---|
| no reach mask | ~1,700 (1,625–1,950) | **~10,200** |
| reach mask on (arms split the rope) | ~700 | **~4,200** |

For scale: **CEM at a 1,000-candidate budget matches or beats CD at ~4× fewer
model rollouts** (`mpc_fix/`, 200 envs × S/U/J = 600 runs each, fixed executor):

| CD (~4,200 cand.) | best mm | final mm | reach@20 mm | med. steps | ms/step |
|---|---|---|---|---|---|
| Ours | 16.7 | 17.8 | 600/600 | 10 | 62 |
| GA-Net | 16.4 | 17.5 | 599/600 | 13 | 152 |
| IN-BiLSTM | 16.5 | 17.3 | 600/600 | 13 | 541 |
| LSTM-GCN | 27.4 | 36.2 | 229/600 | 32 | 62 |

| CEM-1000 | best mm | final mm | reach@20 mm | med. steps | ms/step |
|---|---|---|---|---|---|
| Ours | 16.1 | 17.4 | 599/600 | 10 | 52 |
| GA-Net | 16.4 | 17.7 | 599/600 | 12 | 53 |
| IN-BiLSTM | 16.0 | 17.0 | 600/600 | 11 | 73 |
| LSTM-GCN | 21.4 | 26.8 | 356/600 | 30 | 51 |

> The earlier "CEM 52.6 / 45.7 vs CD 18.0 mm" was the executor bug: CEM
> switches grasps more often than CD, and the bug taxed exactly that. Void.

Note what the CEM rows do to the *cost* story: our end-to-end advantage over
IN-BiLSTM is 8.7x under CD and only 1.4x under CEM-1000. The mechanism is NOT
that the models became equally cheap -- CEM spends its budget as 5 iterations
of 200, and at that batch size the fixed per-call work dominates the rollout.
The per-candidate gap is undiminished: at a 1,024-candidate batch, ours 8.7 ms,
LSTM-GCN 8.8, GA-Net 32.8, IN-BiLSTM 88.2. **Our cost advantage is a property of
large candidate sets, not of the model alone** -- say so wherever it is claimed,
and name the search size.

## 4. The cost — every term

For each candidate the model rolls `H` steps forward from the current belief,
then:

1. **Interp/reproject.** The predicted node cloud goes through a window-5
   reflect-padded moving average, then a **grasp-anchored unit-segment
   rebuild**: walk outward from the grasped node in both directions and place
   every node at exactly the rest segment length along the smoothed direction.
   This is `make_interp_torch` in `scripts/hybrid_variant.py`.
2. **Decode to metres**, then `se = mean((xyz − goal)²)` over all 70 nodes × 3
   coords.
3. **Discount** across the horizon with `γ = 0.95`, normalised by the weight
   sum. At `H = 1` this is a no-op.

Then, added to that scalar:

| term | value | why |
|---|---|---|
| **switch cost** | `+2e-5` if this hand changed link | prices regrasping; the training data only changes grasp at a phase boundary, so a freely-regrasping planner is partly out of distribution |
| **hand proximity** | `+10.0` if `d < 130 mm`, **plus** `+100.0 × (130 mm − d)` | dynamic arm-arm safety on the *first* target pair |
| **reach mask on target** | `+10.0` if the first target leaves this arm's reach region | IK-completability, masked at planning time not execution time |

Plus one hard geometric projection applied to the targets *before* scoring:

**Arc guard.** The two commanded targets are never allowed further apart than
the rope arc between their grasped links: `budget = max(|lA − lB| × 10 mm ×
0.95, 20 mm)`. If violated, both targets are pulled symmetrically toward their
midpoint until they sit exactly `budget` apart. Without it the planner wins cost
by *stretching the rope* — 5/15 envs ended with the grasps 1.02–3.15× the arc
apart and segments up to 32.6 mm against a 10 mm rest length. It is applied
inside every sweep, inside exhaustive/CEM, and once more to the final executed
pair.

**The proximity penalty is graded, not a cliff — leave it that way.** It used to
be `1e9`. At 1e9 the float32 ulp is 64, so every penalised candidate rounded to
*exactly* 1e9, `argmin` degenerated to enumeration order, and the planner
emitted a systematic +x 50 mm drag of both hands. A flat 10.0 still dominates
any feasible cost (~1e-3), so feasible candidates always win, but ordering among
penalised candidates survives and you get the least-bad fallback instead of
whatever is first in the array.

## 5. Execution rule

The planner returns the **first** target pair only. The `H`-step sequence it
believed it was committing to is stored in `planner.last_seq` for logging, and
is then thrown away. Every control step: observe → replan from scratch → execute
one step.

The belief (`h`, `z`) is advanced with the **executed** action, never the planned
one. This matters for ours and RopeDreamer (recurrent); GA-Net and IN-BiLSTM are
stateless across control steps so it does not affect their internals — but it
still affects what *you* log.

## 6. Settings — measured, do not re-derive

```
--horizon 1
--mags 0.05 0.025 0.01 0.005
--regrasp --regrasp-every 1
--cd-rounds 2  --link-stride 2  --min-sep 10  --switch-cost 2e-5
--control-steps 50  --stop-mm 20
```

- **`--horizon 1`.** H=1 greedy beat H=8: 14.3 vs 17.5 mm. The model is only
  trustworthy one step ahead. This survived a 45-cell sweep over H ∈ {10,20,50}
  control-step budgets.
- **`--mags 0.05 0.025 0.01 0.005`.** The coarse end is load-bearing. Capping
  the ladder at 30/20/10/5 mm costs every model (`mpc_fix/`, 600 runs each,
  fixed executor): ours 17.8 → 19.3 mm final and a median 10 → 15 steps to
  goal, GA-Net 17.5 → 18.5 and 13 → 18, IN-BiLSTM 17.3 → 18.4 and 13 → 17,
  LSTM-GCN 36.2 → 49.5 with reach@20 mm falling 229/600 → 122/600.
- **`--regrasp-every 1`.** Regrasping every step beats every-10: 188 vs 157 of
  200 sim envs reaching 20 mm. Do not throttle it.
- **`--stop-mm 20`.** An env freezes once it reaches 20 mm; also our success
  threshold.
- **Do NOT pass `--exhaustive-joint`** (187× the cost for ~0.5 mm; §1).
- **`--cem` is no longer forbidden** — the old prohibition was the executor bug
  talking (§3). CEM-1000 matches CD at ~4× fewer rollouts and cuts IN-BiLSTM's
  model time 7.4×. **CD stays the rig default anyway**, because the rig runs
  already collected are CD and the rig's per-step time is ~98% perception/IK
  overhead, so CEM buys almost no wall-clock there. If you do run CEM on the
  rig, report it as its own arm; never pool CEM and CD runs.
- **Do NOT enable `--refine-iters`.** Gradient refinement of the chosen move
  through the differentiable cost is closed negative: CD+refine 22.4 vs CD 21.8,
  exhaustive+refine 19.8 vs exhaustive 15.1. All planner gain here is discrete.

## 7. Running it

Ours:
```bash
python scripts/mpc_bimanual.py \
    --model-dir deploy/ft1step_rig_s42 \
    --goal-shapes S --envs 1 --control-steps 50 --horizon 1 \
    --mags 0.05 0.025 0.01 0.005 --regrasp --regrasp-every 1 \
    --stop-mm 20 --seed 42 --out <run_dir>
```

GA-Net — identical except two flags:
```bash
python scripts/mpc_bimanual.py \
    --model-dir deploy/ganet_rig_s42 --baseline ganet \
    --goal-shapes S --envs 1 --control-steps 50 --horizon 1 \
    --mags 0.05 0.025 0.01 0.005 --regrasp --regrasp-every 1 \
    --stop-mm 20 --seed 42 --out <run_dir>
```

Add `--reach-map <res_m>` on the rig so the link filter reflects the real
workspace.

## 8. Reporting

Report **planning time per control step** alongside every RMSE, from one
controlled process on one GPU — a planner comparison without its cost is not a
comparison. The harness already records `cum_plan_s` and `plan_s_hist` per env
and reports time-to-30 mm / time-to-50-steps / time-to-20 mm.

Report **best-so-far RMSE and steps-to-threshold**, not final-step RMSE. Two
identical twin runs differ by 8.7 mm on final RMSE but 0.5 mm on best-so-far —
final-step RMSE is noise at our effect sizes.

## 9. If you change one thing, change nothing

Every number in §6 is the surviving end of an ablation that cost cluster time.
If you find yourself needing to alter the search — different atoms, different
penalties, a sampler, a longer horizon — tell us before you run, so we can
either re-run sim under the same change or scope the claim. A rig run under a
different planner cannot be compared to a sim run, and a rig run of *our* model
under a different planner than *GA-Net's* is not a model comparison at all.

---

See also: `docs/robot/HANDOFF_ROBOT_AGENT.md` (safety, bring-up, our bundle),
`docs/robot/HANDOFF_GANET_MPC.md` (GA-Net bundle, and the transition data we
need pushed back).
