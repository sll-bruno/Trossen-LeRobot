# Handoff: GA-Net on the real rig, and getting your MPC data back to us

For the agent/operator on the physical Trossen rig. Self-contained; assumes no
conversation context. Two asks:

1. **Run real-robot MPC with GA-Net** (a second model alongside ours).
2. **Push the state-action transitions your MPC runs produce**, so we can reuse
   them for open-loop evaluation instead of throwing them away.

Everything about safety, bring-up order, calibration and the blockers in
`HANDOFF_ROBOT_AGENT.md` still applies unchanged. This file only covers what is
different for GA-Net and what we need back.

---

## 1. The GA-Net bundle

```
deploy/ganet_rig_s42/model_config.json          architecture
deploy/ganet_rig_s42/norm_stats.pt              state normalisation, REQUIRED
deploy/ganet_rig_s42/ganet_bi_best.pth          weights, 2.3 MB,
                                                md5 0ca5f25c68800caa5750e4254fd41de1
deploy/ganet_rig_s42/checkpoints/ganet_bi_best.pth   same file, second layout
deploy/mpc_targets.npz                          goal shapes S / U / Z / amp / knot / J
```

Both checkpoint copies are md5-identical. The duplication exists because
`scripts/eval_real_chain.py` expects a flat layout and
`scripts/mpc_bimanual.py` expects `checkpoints/` — do not "fix" either.

**Why seed 42**: it is the best of the three GA-Net seeds on both splits, and
by a wide margin out of distribution (t=50: 12.22 mm in-distribution vs 12.56
/ 12.43; 59.76 mm on large actions vs 64.57 / 94.06).

**Config**: `d_model 150, d_hidden 150, num_heads 6, num_layers 1,
paper_feats true, dense_action true, delta true, n_hands 2` — 583,327
parameters. This is the published GA-Net configuration (Gu et al. T-ASE 2025,
Table IV), not a variant we tuned.

**Verified 2026-08-25** on this exact bundle path, fresh process:
```
python scripts/eval_curves_all.py --family ganet --out deploy/ganet_rig_s42 \
    --data simulation/rope_bimanual_rig.npz --curves-json /tmp/gn.json \
    --warmup 1 --rollouts 300
# -> 0.58M params, 0.53 ms/step, t=1 2.57 / t=10 7.46 / t=50 12.12 mm
```
Re-run that if you touch the bundle. If the param count is not 0.58M or the
numbers move, something is wrong before you get near the robot.

## 2. Running MPC with GA-Net

Identical to our model except for two flags — the planner search is unchanged,
which is the point (it isolates the dynamics model):

```
python scripts/mpc_bimanual.py \
    --model-dir deploy/ganet_rig_s42 --baseline ganet \
    --goal-shapes S --envs 1 --control-steps 50 --horizon 1 \
    --mags 0.05 0.025 0.01 0.005 --regrasp --regrasp-every 1 \
    --stop-mm 20 --seed 42 --out <run_dir>
```

`--baseline ganet` swaps the dynamics model; `--model-dir` then points at the
GA-Net bundle rather than ours. Everything else — candidate grid, costs, arc
guard, proximity penalty — is the same code path our model uses.

**The planner itself is specified in full in `docs/robot/HANDOFF_CD_PLANNER.md`
— read that before you touch the search.** It is the same code path for every
model, which is the only reason a rig comparison means anything.

**Planner settings, measured, do not re-derive:**
- `--horizon 1`. H=1 greedy; the model is only trustworthy one step ahead.
- `--mags 0.05 0.025 0.01 0.005` (50/25/10/5 mm per hand).
- `--regrasp --regrasp-every 1`. Regrasping **every** step measurably beats
  every-10 (188 vs 157 of 200 sim envs reaching 20 mm), so do not throttle it.
- Coordinate descent (the default). Do NOT pass `--exhaustive-joint`: it costs
  11.2 s/env-step against CD's 60 ms and buys ~0.5 mm.
  **Correction, 2026-08-26:** the old "CD beat CEM-1000 by 3x (18.0 vs
  58.3 mm)" was the executor bug — it taxed grasp switches, and CEM switches
  more than CD. Under the fixed executor CEM-1000 ties CD at ~4x fewer
  rollouts (GA-Net: 17.7 vs 17.5 mm final, 53 vs 152 ms/step, 600 runs each).
  Keep CD for this campaign so the rig runs already collected stay poolable;
  if you run CEM, it is a separate arm.
- `--stop-mm 20`: an env freezes once it reaches 20 mm.

**GA-Net is stateless.** Unlike our RSSM it carries no belief across control
steps — it consumes `(current_state, velocity, action)` each step and nothing
else. So the belief-poisoning warning in the other handoff (feed the EXECUTED
action, never the planned one) does not apply to GA-Net's internals. It still
applies to whatever *you* log.

## 3. What we need back — this is the part that is easy to lose

Every real MPC control step produces a `(state, action, next_state)` triple.
Those are exactly the transitions we use for open-loop model evaluation, and
they are expensive to collect, so **please push them rather than only pushing
summary metrics.** Episodes being different lengths is fine — our evaluator
already handles variable-length chains and cuts them at re-observations.

**Format: one `.npz` per transition, same schema as
`data/real_rope_2026-08-20/collect_*/tuple_*.npz`,** which we already parse:

| key | shape | dtype | meaning |
|---|---|---|---|
| `s_obs` | (70,3) | float64 | rope BEFORE the action, metres, env-local frame |
| `s_next` | (70,3) | float64 | rope AFTER the action, same frame |
| `tgtA`, `tgtB` | (3,) | float64 | commanded target per hand; NaN if that hand is idle |
| `linkA`, `linkB` | () | int64 | grasped node index; **-1 if that hand is idle** |
| `mode` | () | str | `bimanual` / `A` / `B` |
| `rc` | () | int64 | return code, 0 = clean |
| `execA`, `execB` | () | bool | did that hand actually execute |
| `rope` | () | str | `white` / `blue` — set it per tuple, do not infer later |

Please also add, if you can:
- `goal` (70,3) — the target shape that MPC run was driving toward
- `step` () int — control-step index within the run
- `planned_tgtA` / `planned_tgtB` — the target the planner ASKED for, when it
  differs from what was executed (refused grasp, veto, clamp). Keep `tgtA` as
  what actually happened. We need both to tell model error from execution error.

**Directory layout**: one directory per MPC run, `tuple_000.npz` upward in
execution order, same as the existing collections. Push to a branch and tell us
the branch name; do not merge into `main`.

**Two things that will silently ruin the data if they slip:**

- **Do not write a sentinel for an idle hand's target.** The existing dataset
  encodes an unheld hand as a target 40 metres off the table in 244 tuples; it
  produced a 19,342 mm prediction error on a 0.68 m rope and cost us a full
  re-run before we found it. Use `NaN` for `tgtA`/`tgtB` and `-1` for the link,
  and we will handle it.
- **`s_next[i]` and `s_obs[i+1]` should be the same observation** where the run
  is continuous. We use exact equality to detect where the chain breaks (a
  refused grasp forcing a re-observation) and cut the rollout there. If you
  round, re-serialise, or re-observe unnecessarily between tuples, every chain
  looks broken and the data becomes 50 one-step episodes instead of one
  50-step episode. In the last collection only 10 of 146 chains survived to 50
  steps, which is the single biggest limit on what we can measure.

## 4. What we will do with it

- Your MPC metrics go straight into the closed-loop comparison next to the sim
  numbers.
- The transitions get run through `scripts/eval_real_rig.py` for open-loop
  evaluation of every model, exactly as we do with the 2026-08-20 collection —
  no extra work for you beyond pushing them.

One caveat we would state in the paper and you should know: our sim collector
builds its commanded target as *true grasped-node position + drag*, so a
recorded-action rollout is silently re-anchored to ground truth every step. On
the real rig the target comes from **your** perception, not from ground truth,
so real-rope numbers do not have that problem — which is part of why this data
is worth collecting properly.
