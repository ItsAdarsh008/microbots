# Emergent-physics journal

Newest last. Every number names its seeds; "1 seed" is not a finding.

## 2026-11-04 tick 1: setup

- **Direction:** train the swarm end to end inside physics instead of writing structural behaviours by hand (`PLAN.md`).
- **Physics (`sim.py`)**, checked by hand tests: a lone bot in the air falls and cannot push itself (no flying); a bot
  on the floor crawls; a single-row arm folds under its own weight even with unbreakable bonds (no bending
  stiffness exists); a two-row (triangulated) arm stays up except its one dangling tip bot; a lopsided T on a
  narrow column tips over as a whole (bots cannot anchor to the floor, so balance matters).
- **Shapes (`world.py`)**: towers, T/L/Gamma cantilevers, arches, stairs, stacked blocks, trees (training);
  mushrooms and Y shapes held out. 128 bots, area 1 per bot, standing on the floor.
- **Smoke run (tiny data, 30 iterations, f_bond 8):** first baseline already informative: **the even sample of the
  target, placed directly with all grips on, does not stand**: footprint IoU 0.92 at placement falls to 0.68
  (val) / 0.58 (held-out) after a 10-time-unit passive hold, 44% / 65% of bots drop by more than 1.5 diameters.
  So simply filling the shape evenly is not enough; there is room for the swarm to find something structurally better.
- Speed: `torch.compile` brings one training iteration (64 control steps x 4 physics steps + 100 hold steps,
  forward and backward) from 3.5 s to 0.9 s on an A100.

## 2026-11-04 tick 2: wave 1 started; backprop through the physics explodes

- Data: 3000 training shapes (500 per family), 192 validation, 192 held-out (mushroom, Y). Both boxes run 6 jobs
  at once (about 2 s per iteration each, GPU at 99%).
- Wave 1 (12 runs, plain backprop through all 356 physics steps) started. Within 150 iterations the gradient norm
  reached 10^2 to 10^4 in most runs (clipped to 1 before the update), the training loss on the same batch sequence
  was the same for six different configurations (no visible learning yet), and the drone-actuation control hit
  non-finite gradients at iteration 120 and failed. Several runs saturate the force output (mean a^2 near the
  maximum 9), where tanh passes almost no gradient.
- **Fix under test (box 1, 1000 iterations each):** gradient decay backward in time (0.90 / 0.95 / 0.98 per control
  step; forward pass unchanged), a per-step clamp on the gradient norm, and a shape loss at every control step
  weighted towards the end. Box 2 keeps the six original wave-1 runs as the no-fix comparison.

## 2026-11-04 tick 3: why nothing moved, and what the physics allows

- **Wave 1 and the gradient-fix tests learned nothing** (validation IoU 0.32 to 0.33 in all 12 runs, the pile's own
  level; bond counts identical across runs). Cause: grip started at 0.88, so the starting pile is one bonded solid
  (each bond holds about 7 units, a bot pushes at most 3); training then drove grip to 1.0 and force outputs into
  saturation. All those runs were stopped. **Fix:** bots start mostly free (grip 0.12), and the raw force output is
  penalised so it cannot saturate.
- **Open-loop diagnostic** (`diag_openloop.py`: optimise a free force and grip sequence per bot, no controller,
  8 validation shapes, 200 to 300 Adam steps):

  | actuation | gravity | loss start -> end | IoU end | mean bot travel |
  |---|---|---|---|---|
  | drone (unphysical) | 1 | 38 -> 8.4 | (not logged) | 2.7 |
  | push only against what it touches (r_touch 1.15) | 1 | 39 -> 36 | 0.29 | 1.5 |
  | same, reach 2.0 | 1 | 39 -> 25 | 0.29 | 1.7 |
  | same, reach 2.35, friction 0.2 | 1 | 46 -> 19 | 0.29 | 2.1 |
  | touch only | 0 | 26 -> 17 | 0.29 | 2.8 |
  | reach 2.0 | 0 | 25 -> 17 | 0.27 | 3.4 |

  Gradients through the physics can form shapes when force is free (drone); with momentum-conserving actuation,
  gradient descent finds no way out of the pile, **even without gravity**. So the obstacle is optimisation through
  contact-limited actuation, not gravity alone.
- **Is climbing physically possible?** A scripted bot next to a 6-bot column: with touch-only reach (1.15) a
  constant force never gets it above 3.9 (it loses contact and falls back); with a 2.0 reach (about the v1 arm
  reach) it climbs past the top of the column. So climbing exists in the physics, but needs reach or feedback.
- **Next (running):** closed-loop controllers with low starting grip (drone, touch, reach 2.0, no gravity), and a
  **continuation curriculum**: train with drone forces first and blend to fully physical actuation by iteration
  1800 of 3000 (`mix`). The shape is learned first, physics is tightened afterwards; no behaviour is scripted.
