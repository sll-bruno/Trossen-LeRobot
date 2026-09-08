# Handoff: IN-BiLSTM and LSTM-GCN on the physical rig

For the agent/operator on the Trossen rig. Self-contained; assumes no
conversation context. Two more baselines to run closed-loop, alongside GA-Net
and our field model.

Everything in `HANDOFF_ROBOT_AGENT.md` (safety, bring-up, halt protocol) and
`HANDOFF_CD_PLANNER.md` (the search — atoms, sweeps, costs, arc guard,
execution rule) applies unchanged. This file only covers what differs for these
two models, plus **one change on our side that matters to you**.

---

## 0. READ THIS FIRST — we fixed an executor bug, and your rig was right

Our **sim** MPC loop had a bug: on a control step where the planner chose a new
grasp link, the executor overwrote the planner's returned target with a **pure
lift**, discarding the drag `plan()` had scored. The rig never had this bug —
you execute a full grasp cycle (grasp, drag, release) every step, which is
exactly what the planner asks for.

That bug is why our sim showed GA-Net collapsing (45.7 mm) while your rig gate
showed GA-Net matching us. **Your data was right and our sim was wrong.** Fixed
2026-08-25, commit `89349e9`; correct execution is now the default in
`scripts/mpc_bimanual.py`.

Consequence for you: **nothing to change.** Your runs remain valid and are now
the reference the sim had to be corrected against. Under the fixed sim, ours,
GA-Net and IN-BiLSTM all converge to 17.3–17.8 mm and differ only in planning
cost — so the interesting rig question is no longer "which model is more
accurate" but **"which model reaches the goal for how much compute"**. Please
log `plan_s` per control step on every run; it is now a headline metric, not a
diagnostic.

## 1. The two bundles

```
deploy/inbilstm_pub/model_config.json     architecture
deploy/inbilstm_pub/norm_stats.pt         REQUIRED, (3,) mean + SCALAR std
deploy/inbilstm_pub/inbilstm_bi_best.pth  weights
deploy/inbilstm_pub/checkpoints/…         same file, second layout

deploy/lgcn_pub/model_config.json
deploy/lgcn_pub/norm_stats.pt
deploy/lgcn_pub/lstmgcn_bi_best.pth
deploy/lgcn_pub/checkpoints/…
```

Both ship two copies of the checkpoint because `eval_real_chain.py` expects a
flat layout and `mpc_bimanual.py` expects `checkpoints/`. Do not "fix" either.

> **Pushed 2026-08-25 in `397501b`** on `latent-planning-pivot`. The earlier
> version of this handoff described these files before they were committed —
> apologies, that was on us. The `checkpoints/` copies were also 0-byte
> symlinks locally and would not have survived a clone; they are now real
> files, md5-identical to the flat copy
> (IN-BiLSTM `24642d7cbd2154ce85b72633669005b8`,
> LSTM-GCN `2b9f6cf8c866ef6d00be4f78c582ed39`).
> `.pth` files are **git-lfs** tracked, so `git lfs pull` after checkout.
>
> Verified on the exact pushed files: strict `load_state_dict` passes,
> IN-BiLSTM **1,156,353** params, LSTM-GCN **1,377,106**, both with a `(3,)`
> mean and a scalar std.

| model | params | config source |
|---|---|---|
| IN-BiLSTM | **1,156,353** | Yang, Stork & Stoyanov, ICRA 2021 — 4-layer `f_e` → 2-layer bi-LSTM → state encoder + effect predictor, all 150 units |
| LSTM-GCN | **1,377,106** | Yue et al., ICMA 2025 — 2×GCN-64 → FC → 1×LSTM-128 |

Both are published configurations, not variants we tuned. If your param count
does not match exactly, stop before you go near the robot.

**Loading** — same as GA-Net, but note each pops different config keys:

```python
import json, torch
cfg = json.load(open("deploy/inbilstm_pub/model_config.json"))
cfg.pop("delta", True)                     # selects bi_step's residual mode
from src.models.in_bilstm import IN_biLSTM
m = IN_biLSTM(**cfg)                       # strict load passes

cfg = json.load(open("deploy/lgcn_pub/model_config.json"))
cfg.pop("delta", True); cfg.pop("window", None)   # LSTM-GCN also has `window`
from src.models.lstm_gcn import LSTMGCNModel
m = LSTMGCNModel(**cfg)
```

`window` is **not** a warmup and must not be used as one — treating it as
observed frames silently feeds the model extra ground truth. Pop it and pass it
to the rollout, exactly as `scripts/eval_lstmgcn_bi.py` does.

**Normalisation**, the usual trap: both carry a `(3,)` per-coordinate mean and a
**scalar** std, unlike our field model's `(210,)`. Tile the mean across the 70
nodes, keep the std scalar, and hand the `(3,)` mean to the planner as `mean3`.
This is the same wiring you already did for GA-Net.

## 2. State handling — they differ from each other, and from GA-Net

| model | carries state across control steps? |
|---|---|
| GA-Net | **No** — stateless, `bi_step(curr, vel, act)` |
| IN-BiLSTM | **No** — its bi-LSTM is *spatial*, run along the 70 nodes and zero-initialised every call. Its only memory of the past is the velocity channel |
| LSTM-GCN | **YES** — a genuine temporal LSTM `(h, c)`, each `(1, 128)` |

For **LSTM-GCN only**, the belief must be advanced once per control step with
the **executed** action, on **observed** states only — never on its own
predictions. `mpc_bimanual.py` does this in the control loop; if you drive it
yourself, mirror that. Zeros are the correct initial `(h, c)` — that is the
trainer's window-start state, so an episode start is in distribution.

Both take the true previous-frame velocity (`curr − prev`), like GA-Net.

## 3. Running

Identical to GA-Net except `--baseline` and `--model-dir`:

```bash
python scripts/mpc_bimanual.py \
    --model-dir deploy/inbilstm_pub --baseline inbilstm \
    --goal-shapes S --envs 1 --control-steps 50 --horizon 1 \
    --mags 0.03 0.02 0.01 0.005 --regrasp --regrasp-every 1 \
    --stop-mm 25 --seed 42 --out <run_dir>

python scripts/mpc_bimanual.py \
    --model-dir deploy/lgcn_pub --baseline lstmgcn \
    ... same flags ...
```

Keep **your** rig deviations exactly as they are for GA-Net and ours — same
ladder (30/20/10/5), same per-shape `stop_mm`, same `link_stride`, same end
margin, same reach mask. The comparison is only meaningful if the four models
see an identical search.

**Budget warning — IN-BiLSTM is expensive.** On an L40S it plans at **541 ms
per env-step** against GA-Net's 152 and ours' 62, i.e. ~3.6× GA-Net and ~8.7×
ours. On your hardware expect it to dominate the run; the exhaustive planner is
out of the question for it. Sizing your campaign, assume IN-BiLSTM runs take
several times longer per control step than GA-Net's did.

> If that budget is what blocks the campaign, `--cem --cem-budget 1000` is now
> allowed (2026-08-26: the old "do NOT use --cem" was the executor bug). It
> ties CD in sim at ~4× fewer rollouts and drops IN-BiLSTM to **73 ms**/env-step
> — 7.4× cheaper. The catch: the rig runs already collected are CD, and a CEM
> run is a separate arm that cannot be pooled with them. Ask before switching.

Expected from the fixed sim (200 envs × S/U/J, CD, our ladder): IN-BiLSTM
converges to **17.3 mm** — statistically level with GA-Net (17.5) and ours
(17.8) — but needs **6.24 s of planning to reach 20 mm** against GA-Net's 1.85
and our 0.60. LSTM-GCN **does not converge** in sim (36.2 mm at step 50, only
229/600 envs reach 20 mm), so treat it as the one that may genuinely fail on
the rig; that is a result, not a bug, and it should still be run.

## 4. Campaign design — what to run next, concretely

Your CD campaign as of `f9bdf7c` gives us **14 GA-Net + 12 ours**, unevenly
spread: ours has 5/3/4 on S/U/J, GA-Net 4/5/5. Two things would help most, in
this order.

**Priority 1 — finish the existing 2x3 grid to 5 runs per cell.** Missing:
ours U (+2), ours J (+1), GA-Net S (+1). Four runs. Uneven n is what currently
stops us stating the ours-vs-GA-Net comparison cleanly, and it is the cheapest
thing on this list.

**Priority 2 — IN-BiLSTM and LSTM-GCN, interleaved with a few fresh anchors.**
Do NOT run them as a separate block: a block confounds model with rope wear,
lighting and calibration drift over a long session, and your existing campaign
was explicitly designed to avoid that. Extend the same construction — with four
models and three shapes, a period-12 sequence visits every (shape, model) cell
with equal spacing. If a full 4x3x5 = 60-run campaign is unaffordable
(IN-BiLSTM alone will dominate it, see the budget warning above), the fallback
is a period-6 interleave of {IN-BiLSTM, LSTM-GCN} across the three shapes with
one fresh GA-Net run inserted every 6 to anchor drift against the runs you
already have.

Target 5 runs per (model, shape). **Tell us what is actually affordable before
you start** and we will scope the claim to it rather than discovering the
imbalance afterwards.

**Stop rule:** keep your per-shape thresholds (S 26.5 / U 25 / J 25). Do not
switch to a flat 20 mm — nothing on this rig has reached it, and it would turn
most runs into stall-outs.

**One thing that would change our analysis a lot, if it is cheap for you:**
a handful of runs at 80-100 control steps instead of ~50. Our sim shows the
slower models are still descending when a 50-step budget cuts them off, and we
currently cannot tell on hardware whether IN-BiLSTM/LSTM-GCN plateau or simply
run out of budget. Three long runs (one per shape, any model that looks slow)
would settle it.

## 5. What we need back

Same as before, and it has not changed:

- **the transitions**, one `.npz` per control step, schema in
  `HANDOFF_GANET_MPC.md` §3 — `s_obs`, `s_next`, `tgtA`/`tgtB` (NaN if idle),
  `linkA`/`linkB` (**−1** if idle), `mode`, `rc`, `execA`/`execB`, `rope`, plus
  `goal`, `step`, and `planned_tgtA`/`planned_tgtB` when they differ from what
  was executed;
- **`plan_s` per control step** — now a headline metric, see §0;
- `log.json` + `outcome.json` per run, as you already produce.

Two things that silently ruin the data, repeated because they cost us a re-run
before: **never write a sentinel target for an idle hand** (use `NaN` + link
`−1`), and **`s_next[i]` must equal `s_obs[i+1]`** where the run is continuous —
we use exact equality to detect where a chain breaks and cut the rollout there.

Push to a branch and tell us the name; do not merge to `main`.

## 6. Verify before the robot

Run your existing offline gate (`scripts/ganet_gate_rig.py`, pointed at each
new bundle) on the real transitions you already have. A mis-wired model does
not land near a correct one — it lands at or above persistence. Both of these
should beat persistence comfortably; if either does not, the wiring is wrong
and nothing downstream is worth collecting.
