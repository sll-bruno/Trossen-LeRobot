# Handoff: switch our model to the BASE checkpoint (`deploy/ours5_rig_s42`)

For the agent/operator on the Trossen rig. Self-contained. One change: which of
our two checkpoints you load. Nothing about the planner, the safety layer, the
ladder or the campaign design changes.

---

## 1. What changed and why

You have been running `deploy/ft1step_rig_s42` — our 0.37M field model after a
**1-step fine-tune on rig-matched simulation**. We just evaluated both of our
checkpoints open-loop on **your** 663 MPC transitions (the 15-run eval set,
403 transitions, free-running with the commanded drag re-anchored to the
model's own belief so it is never told the true rope state).

The fine-tune is worse the further ahead you predict:

| checkpoint | k=1 | k=5 | k=10 | k=20 | k=30 | k=40 |
|---|---|---|---|---|---|---|
| `ours5_rig_s42` (**base**) | 14.8 | 28.8 | 41.5 | **59.4** | **55.9** | **66.3** |
| `ft1step_rig_s42` (deployed) | 14.8 | 29.1 | 43.3 | 68.7 | 86.9 | **163.3** |
| predict no movement | 14.3 | 28.4 | 42.1 | 64.8 | 83.0 | 94.2 |

Identical at 1 step, 2.5x apart by 40. The fine-tune optimised single-step
accuracy and destabilised the rollout — a textbook one-step-objective failure.
The base tracks GA-Net (67.6) and IN-BiLSTM (59.0) at long horizon; the
fine-tuned one is the only model in the comparison worse than doing nothing.

**What this does NOT change:** at the horizon MPC actually uses (`--horizon 1`)
the two are indistinguishable — 14.8 vs 14.8 mm. So if you only ever run H=1
MPC, this swap will not move your closed-loop numbers. It matters for anything
that predicts more than one step ahead, and for what we can claim about the
model in the paper.

## 2. The bundle

```
deploy/ours5_rig_s42/model_config.json        architecture
deploy/ours5_rig_s42/norm_stats.pt            (210,) mean + (210,) std, REQUIRED
deploy/ours5_rig_s42/bi_best.pth              weights, 1.5 MB
deploy/ours5_rig_s42/checkpoints/bi_best.pth  same file, second layout
```

Pushed **2026-08-25** on `latent-planning-pivot`. Both copies are real files
(not symlinks), md5 `3f8ca9d7c3f501b87d2144a0d2496419`. `.pth` is **git-lfs**
tracked — run `git lfs pull` after checkout or you will get a 130-byte pointer
that fails to load.

Verified on the exact pushed files:

```
params = 369,973    0 missing / 0 unexpected keys
d_rnn=96  d_z=32  d_hidden=160  d_node=160  enc_layers=0  act_mode=raw  n_hands=2
```

**Architecture is byte-identical to `ft1step_rig_s42`** — same config, same
370k params, same normalisation shape. Only the weights differ. So nothing in
your loading code needs to change; `--model-dir` is the whole diff.

## 3. Running it

```bash
python scripts/mpc_bimanual.py \
    --model-dir deploy/ours5_rig_s42 \
    --goal-shapes S --envs 1 --control-steps 50 --horizon 1 \
    --mags 0.03 0.02 0.01 0.005 --regrasp --regrasp-every 1 \
    --stop-mm 26.5 --seed 42 --out <run_dir>
```

No `--baseline` flag (it is our own model, not a baseline). Keep every rig
deviation exactly as you have it — ladder 30/20/10/5, per-shape `stop_mm`,
`link_stride`, end margin, reach mask. `strict=False` on the state-dict load is
correct and expected: the reconstruction decoder is training-only. It must
still report **0 missing / 0 unexpected**.

## 4. IMPORTANT — do not swap mid-campaign

Your CD campaign interleaves models across runs specifically so that model is
not confounded with rope wear, lighting and calibration drift. **Changing our
checkpoint partway through that sequence reintroduces exactly the confound the
design exists to remove.**

So:

- **Finish the current 2x3 grid on `ft1step_rig_s42`.** The 4 runs still
  missing (ours U +2, ours J +1, GA-Net S +1) should use the checkpoint the
  other 12 used. Do not switch for those.
- **Start `ours5_rig_s42` as a NEW arm** in the next campaign block, interleaved
  with the others the same way.
- If rig time only allows one of the two, **prefer finishing the existing grid**
  — a balanced 2x3 is worth more to us right now than an unbalanced 3x3.

Tell us which you choose so we scope the claim to what you actually ran.

## 5. What to log

Unchanged from the previous handoffs — the per-step transitions
(`s_obs`, `s_next`, `tgtA`/`tgtB` with **NaN** for an idle hand, `linkA`/`linkB`
with **-1** for idle, `goal`, `step`, `planned_tgt*`, `rope`), plus `plan_s`
per control step.

Two failure modes worth repeating because they silently destroy the data:
never write a sentinel target for an idle hand, and keep `s_next[i]` exactly
equal to `s_obs[i+1]` where the run is continuous — we detect chain breaks by
exact equality and cut the rollout there. In your last collection only 23 of
the runs produced unbroken chains long enough for multi-step evaluation, and
that is the single biggest limit on what we can measure.

Please add `model` to each tuple (you already do) so `ours5` and `ft1step` runs
can never be pooled by accident.
