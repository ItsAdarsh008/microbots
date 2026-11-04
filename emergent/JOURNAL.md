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

## 2026-11-04 tick 4: learning is unstable; a better-conditioned motor model

**Wave 2 and the curriculum runs (validation IoU of the EMA controller, 64 shapes, pile = 0.32):**

| run | it 250 | it 500 | note |
|---|---|---|---|
| drone forces | **0.475** | 0.090 | learned to fill the lower part of shapes, then diverged (bots fly off) |
| drone, gradient decay 0.95 | 0.31 | 0.32 | no learning |
| drone, no gravity | 0.30 | 0.33 | |
| drone, no per-step loss | 0.32 | 0.29 | 0.19 at 1000 |
| physical (touch) | 0.29 | 0.32 | grip back to 0.96: the pile trap |
| physical, reach 2.0 | 0.29 | 0.32 | grip 0.97, pile trap |
| physical, no gravity | 0.28 | 0.30 | |
| curriculum drone->physical (reach 1.15 / 2.0) | | 0.28 / 0.28 | |

- Gradient decay and dropping the per-step shape loss both hurt; neither is used from here on.
- The one run that clearly learned (drone, 0.475) diverged within the next 250 iterations: raw force outputs
  through stiff contact dynamics are badly conditioned (learning rate 1e-3, gradient clipped to norm 1 every step).
- Looked at the drone run at iteration 250 (`render.py`): bots fill the bottom of each shape and hover; grip ~0.09,
  so no bonds and 68% fall during the hold.
- **The even sample of the target, placed and frozen, slumps into a puddle for most shapes**, towers included
  (bond strength 8). Hand test: a densely packed (hexagonal) tower 5 bots wide stands (top 10.0 -> 9.7, mostly contact
  compression), 2 to 3 wide buckle unless bonds are strong (16 holds noticeably better than 8). An even sample at
  1 unit of area per bot is looser than dense packing, so many neighbours are out of bond range and the bond network
  is floppy. Standing requires compaction; that is something the swarm would have to find.
- **Change (wave 4):** the controller can output a *commanded velocity*; a force-limited motor tracks it
  (force = f_act tanh(k (v_cmd - v) / f_act), still momentum-conserving in the physical mode). Sweep: velocity vs
  force control, learning rate 1e-3 / 3e-4 / 1e-4, air damping, batch 32, physical reach 1.15 / 2.0, no gravity,
  curriculum, no hold loss. 12 runs, 1000 iterations each.

## 2026-11-04 tick 5: drone forces learn at a lower learning rate; physical forces under gravity do not (yet)

Validation IoU of the averaged (EMA) controller at iteration 500 (64 shapes; pile 0.32; 1 seed each):

| run | build | after hold |
|---|---|---|
| drone, force output, lr 3e-4, air damping 0.5 | **0.59** | 0.34 |
| drone, force output, lr 3e-4 | 0.56 | 0.26 |
| drone, force output, lr 1e-4 | 0.47 | 0.27 |
| drone, velocity command, lr 3e-4 | 0.46 | 0.26 |
| drone, velocity command, lr 1e-3 | 0.25 | 0.24 (diverged; stopped) |
| physical (reach 2 or touch), velocity, lr 3e-4 / 1e-4, with or without hold loss | 0.31 to 0.32 | 0.29 to 0.32 (grip -> 0.97 to 1.00; three stopped) |
| physical, reach 2, no gravity | 0.37 | 0.30 |
| curriculum drone -> physical (78% physical at it 500) | 0.30 | 0.26 |

- Lower learning rate (3e-4) fixes the divergence of the drone runs; force output learns faster than velocity commands.
- Drone controllers never learn to grip (grip about 0.1): they hover the shape and fall in the hold. The hold loss
  alone does not teach latching yet.
- Every gradient run with momentum-conserving actuation under gravity sits at the pile. Grip goes to 1 **even with
  no hold loss**: a fully bonded pile moves as one body that the bottom row can shove, which is a local optimum for
  the gradient, not a structural insight.
- **Next:** PPO (`rl.py`) on the same controller (stochastic actions, reward = decrease of shape loss per step plus
  the hold result), on drone (sanity check) and physical forces; replaces the four dead gradient runs.
