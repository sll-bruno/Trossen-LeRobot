# Handoff: add the small GRU model (`deploy/gru_small_rig_s42`) to the rig campaign

For the agent/operator on the Trossen rig. Self-contained. This ADDS a model to
test; it does not replace `deploy/ours5_rig_s42` and changes nothing about the
planner, the safety layer, the ladder or the campaign design.

**Read §4 before running anything.** This checkpoint takes a different code
path inside the model than every other model you have run, and there is a
tripwire that will abort the run if that path is not taken.

---

## 1. What this model is, in one paragraph

Our 0.37M field model has two halves: a recurrent core that carries belief
across steps, and a per-node decoder that turns belief into per-node position
deltas. The core has always been an **RSSM** — a GRU state `h` plus a stochastic
latent `z` with a learned prior, posterior and KL term. This checkpoint replaces
that entire RSSM with a **plain GRU cell** (`h = GRU([enc(state), action], h)`,
no `z`, no prior/posterior, no KL). Everything else — encoder, decoder, action
handling, normalisation, self-observation — is identical.

In simulation it is a wash on accuracy and a large win on cost:

| model | active params | k=1 | k=10 | k=50 | ms/step |
|---|---|---|---|---|---|
| `ours5_rig_s42` (RSSM) | 289,769 | 2.93 | 10.26 | 22.26 | 1.89 |
| **`gru_small_rig_s42`** (plain GRU) | **204,265** | 2.81 | 10.15 | **22.16** | **1.05** |

Rig-matched sim test split, 4320 held-out episodes, free-running (the commanded
drag is re-anchored to the model's own believed grasp node, so it is never told
the true rope state). Both timed in one process on one pinned A5000.

Same accuracy, **30% fewer active parameters, 45% less time per rollout step.**

## 2. What we want from you

Run the small GRU through **the same real MPC protocol you already run for
`ours5_rig_s42`** — same shapes, same ladder, same number of runs, same safety
layer. We want a like-for-like closed-loop comparison on the real rig.

Priority order if you are short on rig time:

1. `gru_small_rig_s42` vs `ours5_rig_s42`, same shapes, same run count.
2. Open-loop prediction rows for both against **predict-no-movement** (see §5 —
   this matters more than usual here).
3. Anything else.

## 3. The bundle

```
deploy/gru_small_rig_s42/model_config.json        architecture (fused_cell: true)
deploy/gru_small_rig_s42/norm_stats.pt            (210,) mean + (210,) std, REQUIRED
deploy/gru_small_rig_s42/bi_best.pth              weights, 1.8 MB
deploy/gru_small_rig_s42/checkpoints/bi_best.pth  same file, second layout
```

Both copies are real files (not symlinks), md5
`71dc0ccf965f7ffc2bf3a2456c6f6074`. `.pth` is **git-lfs** tracked — run
`git lfs pull` after checkout or you will get a small pointer file that fails to
load.

Verified on the exact pushed file:

```
total params   = 462,715
  fused_gru    =  92,736   <- the trained recurrent core
  pred_decoder = 111,529
  rssm         = 178,240   <- PRESENT BUT DEAD, see below
  recon_decoder=  80,210   <- train-only, not used at inference
active at inference = 204,265
d_rnn=96  d_z=32  d_hidden=160  d_node=160  enc_layers=0  act_mode=raw
n_hands=2  head_layers=3  self_observe=true  fused_cell=TRUE
```

`--model-dir deploy/gru_small_rig_s42` is the whole diff on the command line.

## 4. The one thing that can go badly wrong

**The checkpoint contains a full RSSM that was never trained.** The model class
builds `self.rssm` unconditionally, so those 178,240 weights are in the file at
random initialisation. The checkpoint therefore loads with **zero missing and
zero unexpected keys** and prints no warning — but any code path that drives
`self.rssm` instead of `self.fused_gru` is running a **random network** and will
produce confident, plausible, meaningless numbers.

This is not hypothetical. It is exactly how we lost a week of ablation results:
the trainer optimised the RSSM while the evaluator ran the untrained GRU, and
the arm looked catastrophically bad for reasons that were entirely an artifact.
Fixed 2026-08-28.

Two protections are in place, and you should confirm both:

**(a) A tripwire.** When `fused_cell` is true, `scripts/mpc_bimanual.py` hooks
every submodule of `model.rssm` and raises on any forward call. At the top of a
correct run you will see:

```
fused_cell=True: RSSM tripwire armed
```

If you see that line and the run completes, the fused path was taken. If you
see

```
RuntimeError: FUSED-CELL VIOLATION: self.rssm was invoked on a --fused-cell
checkpoint, whose RSSM is at random init.
```

then some path still lacks a fused branch. **Do not remove the tripwire and do
not work around it — tell us which command produced it.** A crash here is the
system working; silence would be the failure.

**(b) Patched paths.** These now branch on `fused_cell` in
`scripts/mpc_bimanual.py`: `_cost_chunk` (candidate scoring, the hot path),
`predict_traj`, `_refine_moves.cost_of`, and the main carried-belief update.
`scripts/eval_real_openloop.py` is patched for warmup and rollout. Anything
else is covered by the tripwire.

If you use a script we have not named — `mpc_bimanual_record.py`,
`eval_real_openloop_mpc.py`, or your own — it may not be patched. The tripwire
travels with `mpc_bimanual.py` only. Tell us before you run it.

**Sanity check before spending rig time** (sim, ~2 min, no robot):

```bash
python scripts/mpc_bimanual.py --model-dir deploy/gru_small_rig_s42 \
  --goal-shapes S --envs 6 --control-steps 4 --horizon 1 \
  --mags 0.05 0.025 0.01 0.005 --regrasp --seed 42 --out /tmp/gru_check
```
Expect `fused_cell=True: RSSM tripwire armed`, no violation, final RMSE ~68 mm
on this tiny 6-env/4-step config. (Our RSSM model scores ~70 mm on the identical
command — that difference is noise at this size, it is a smoke test, not a
result.)

## 5. Set your expectations low, and why

On the 3-episode real validation set we already have, **every one of our field
models — including this one and the RSSM base — is worse than predicting no
movement**, by 16–23 mm at t=1. GA-Net is the only learned model in our
comparison that beats the null at longer horizons on real data.

Worse, across nine architecture variants, **sim accuracy anti-correlates with
real-rope accuracy** (Spearman rho = -0.79, p = 0.010; Pearson r = -0.94). The
best sim model was last of nine on real; the best on real was the worst in sim.
Much of that is explained by a boring mechanism — the arms that do well on real
are the ones that barely predict motion at all, and on real data doing nothing
is currently the winning strategy — but the practical consequence stands:

> **Our simulation numbers do not predict real-rig performance, and the
> 22.16-vs-22.26 mm sim tie above is not evidence that this model will behave
> like the RSSM on your rig.**

That is precisely why we want the real closed-loop comparison rather than
shipping the swap on sim evidence. If the GRU is meaningfully worse on the rig,
that is a publishable result and not a failed errand. Report what you measure.

Caveat on our side: that real set is 3 episodes / 33 transitions, one rope, one
session, with ~20 mm median per-frame motion (each frame is a whole
pick-drag-place). It ranks and diagnoses; it does not validate.

## 6. What we are NOT claiming

- **Not** that this is faster to plan with. Sim MPC planning was ~170 ms/env for
  the GRU vs ~167 ms for the RSSM — indistinguishable. The per-candidate decode
  dominates planning cost, so the 45% rollout saving does **not** reach MPC.
  Expect no closed-loop speedup. The win is model size and open-loop cost.
- **Not** that the RSSM is useless in general — only that on this task, at this
  scale, with this decoder, it buys no accuracy for 48% of the parameters.
- **Not** that the sim tie transfers. See §5.

## 7. Provenance

Trained on `simulation/rope_bimanual_rig.npz`, 60 epochs, seed 42, identical
recipe and hyperparameters to `ours5_rig_s42` — the ONLY difference is
`--fused-cell`. Checkpoint selected on a de-anchored (leak-free) rollout metric,
so no ground truth re-enters the rollout during selection.

Job `139590_1` on IAS, 2026-08-28. A second seed (43) exists and agrees
(21.7 vs 21.9 mm on the selection metric); `fusedgru160`, a wider variant with
307k active params, is marginally more accurate (21.96 mm) at the same 1.03
ms/step and can be shipped too if you want a third arm.

Grad-flow assertion added to `scripts/bimanual_variant.py`: training now fails
on the first batch if the recurrent core the rollout will drive receives no
gradient. That is the regression guard for the bug in §4.
