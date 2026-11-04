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
