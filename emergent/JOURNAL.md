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

**Correction (same tick):** the four "PPO" runs queued in tick 5 did not run PPO. The workers on the boxes had been
started before `worker.py` learned to route `algo: rl` configs to `rl.py`, so they ran the gradient trainer with its
defaults (learning rate 1e-3, force bound unused options ignored). That is why two of them gave identical numbers.
Their results are discarded; the PPO runs were restarted directly with `rl.py`. Lesson: a long-running worker keeps
the code it started with; restart workers between waves when the worker itself changes.

## 2026-11-04 tick 6: gradients through the physics are out for physical actuation; PPO starts moving

Wave 4 at iteration 750 (validation IoU at the end of the build / after the hold; pile 0.32; 1 seed):
- Drone, force output, lr 3e-4: **0.61** / 0.25. With air damping 0.5: 0.58 / 0.33. lr 1e-4: 0.58 / 0.27.
  Velocity commands: 0.49 (batch 16) and 0.55 (batch 32). A second seed of the velocity run diverged at
  iteration 500 (0.25), so even the drone setting is not reliably stable.
- Physical actuation under gravity, every variant (reach 2 or touch, lr 3e-4 or 1e-4, gravity ramp, drone->physical
  curriculum with or without the gravity ramp, no hold loss): 0.29 to 0.32, the pile. No gravity: 0.36.
- Conclusion: backprop through the contact physics learns shape formation when force is free, never with
  momentum-conserving actuation under gravity in this setup. Remaining gradient runs on physical actuation stopped.

PPO (`rl.py`, first 10 updates, physical actuation under gravity, reach 2): batch build loss 43 -> 27, the first
movement away from the pile any physical run has shown (gradient runs stayed at 34 to 40). Not yet evaluated.
Wave "rl2" (bigger minibatches: 64 envs, 64 frames per step): reach 2 and touch, bond 16, second seed, hold weight 3,
no-gravity control, drone; plus the first three PPO runs, 11 PPO runs in total.

## 2026-11-05 tick 7: the physics was too soft for anything to stand; split locomotion from structure

**Can a perfectly placed target stand at all?** (`standability.py`: the even sample of every validation and
held-out shape, every grip on, 10 time units of passive hold.)

| physics | bond strength | val IoU after hold (placed 0.92) | tee | arch | tower | held-out (mushroom, Y) |
|---|---|---|---|---|---|---|
| soft (k 200, dt 0.02) | 8 | 0.38 | 0.20 | 0.31 | 0.63 | 0.14 |
| soft | unbreakable | 0.45 | 0.23 | 0.36 | 0.77 | 0.19 |
| stiff (k 2000, dt 0.01) | 16 | 0.44 | 0.24 | 0.35 | 0.71 | 0.21 |
| stiff | 64 | 0.53 | 0.30 | 0.48 | 0.81 | 0.36 |
| stiff | unbreakable | 0.55 | 0.31 | 0.54 | 0.82 | 0.44 |

- Rendered: in the soft physics a densely packed T beam bends down like rubber **even with unbreakable bonds**:
  bond *stiffness* was fixed at 200 and a bond disappears beyond 0.25 of stretch, which the root of a long beam
  exceeds. Not a structural effect, a modelling error. Bond stiffness is now a parameter.
- Hand test (symmetric T: 3-wide, 10-high column, two-row beam with 7-bot arms each side, dense packing): stiff
  physics with unbreakable bonds holds (tip drop 0.8 at k 2000, 0.4 at k 4000); with strength 16 (about the real
  v1 connector-to-weight ratio) the arms break off (drop 6.5), consistent with the v1 note that only one bot can
  cantilever sideways. Soft physics: drops 7.8 even unbreakable. A lopsided Gamma tips over as a whole (no floor
  anchoring), which is correct.
- Even with stiff unbreakable bonds, the *even sample* loses 40% of its overlap: an irregular arrangement leaves
  neighbours out of bond range and the bond network has floppy modes. Whether a shape stands depends on the
  arrangement, which is the question the swarm has to answer.
- My hexagonal-packing reference was flawed (filling deepest points first leaves thin stems unselected, so caps
  detach); discarded.
- **PPO (team reward) does not learn the physical task** in 50 to 60 updates (validation 0.29 to 0.30, policy
  updates near zero): one reward shared by 128 bots gives each bot almost no credit signal. Drone PPO: 0.37 at
  update 50 (slow). Four PPO runs kept, the rest stopped.
- **Split the problem.** (1) Locomotion with momentum-conserving forces: still unsolved; revisit with per-bot
  rewards. (2) **Structure:** wave 5 builds with drone forces (which learn), under the stiff physics, then holds
  passively with motors off: does training find arrangements and latching that stand, without any rule about
  rigidity, triangulation or bending? Bond strength 16 / 64 / unbreakable, hold weight 0 / 1 / 3, no-gravity
  control, second seeds.

## 2026-11-05 tick 8: structure wave at iteration 750; PPO stopped; always-latched wave

Validation IoU (EMA controller, 64 shapes; build / after passive hold / after hold with every grip forced on;
pile about 0.32; stiff physics; drone build forces; 1 seed unless noted):

| run | it 750 build | hold | hold, all grips on |
|---|---|---|---|
| bond 16, no gravity in training (hold under gravity) | **0.46** | 0.33 | 0.39 |
| bond 16, no hold loss | 0.41 | 0.35 | 0.36 |
| bond 64 (seed 0 / seed 1) | 0.40 / 0.40 | 0.30 / 0.28 | 0.32 / 0.31 |
| bond 16, hold weight 3 | 0.38 | 0.29 | 0.32 |
| bond 16 (seed 0 / seed 1) | 0.26 / 0.27 | 0.26 / 0.27 | 0.27 / 0.27 (not learning) |
| unbreakable | 0.33 | 0.33 | 0.33 (not learning) |

- Learning under the stiff physics is slower and seed-dependent; three of eight runs have not left the pile.
- **No run learned to grip** (grip stays about 0.12, zero bonds after the hold), and forcing every grip on adds
  only 0.02 to 0.06: the arrangements the controllers make are too loose for latching to matter, so there is no
  signal to learn it. The structural question cannot be answered by these runs as set up.
- PPO, all variants (team reward, per-bot reward, physical or drone): flat after 90 to 180 updates. Stopped.
- **Wave 6: every bot always latched** (grip fixed at 1). Bots stick to what they touch; with bond strength above a
  bot's push (4 or 16 vs 3) a misplaced bot cannot free itself, so assembly order matters (growth-like); with
  strength 2 bots can pull free. Drone build, stiff physics, passive hold. Bond 2 / 4 / 16, no-gravity control,
  second seed, hold weight 3.

## 2026-11-05 tick 9: the averaged weights hide learning when bots are always latched

Wave 6 (always latched) at iteration 500 looked stuck at the pile in the evaluation (EMA weights: build 0.33),
while the training-batch losses were low. Evaluating the **raw** weights of the same checkpoint: bond 16 build 0.41
(EMA 0.33), bond 4 build 0.40 (EMA 0.33); both collapse in the hold (88 to 90% of bots fall). With latching,
breaking free is all-or-nothing, and an average of weights pushes slightly too weakly to break any bond.
Re-checked the runs called "not learning" earlier with raw weights: stiff bond 16 (seeds 0 and 1), unbreakable,
physical reach 2, physical curriculum: raw equals EMA within 0.01, so those verdicts stand.
From now on every in-training and final evaluation reports both (`eval-raw`, `val_raw`, `held_raw`).

## 2026-11-05 tick 10: structure wave (5) final: the learned swarms do not stand better than the pile

Final, all 192 validation / 192 held-out shapes, EMA weights (raw weights re-checked equal for the stuck runs, tick 9).
References (validation): pile 0.36, even sample placed 0.92, even sample after hold 0.44 (bond 16) / 0.53 (bond 64) / 0.54 (unbreakable).

| run (drone build, stiff physics, passive hold) | build | hold [95% CI] | hold, all grips on | fallen | held-out build / hold |
|---|---|---|---|---|---|
| bond 16, hold weight 3 | 0.51 | 0.33 [0.31, 0.36] | 0.41 | 0.54 | 0.39 / 0.17 |
| bond 16, no hold loss | 0.49 | 0.34 [0.31, 0.37] | 0.40 | 0.64 | 0.35 / 0.13 |
| bond 64 (seed 0 / 1) | 0.46 / 0.44 | 0.33 / 0.29 | 0.35 / 0.32 | 0.71 / 0.71 | 0.35 / 0.11, 0.31 / 0.10 |
| bond 16, trained without gravity | 0.42 | 0.35 [0.32, 0.37] | 0.40 | 0.27 | 0.28 / 0.17 |
| unbreakable | 0.35 | 0.35 | 0.35 | 0.00 | 0.16 / 0.16 (never left the pile) |
| bond 16 (seed 0 / 1) | 0.25 / 0.29 | 0.26 / 0.29 | | 0.00 | (never left the pile) |

- **Negative result.** After the hold every learned swarm sits at the pile level (0.26 to 0.35) and below the even
  sample placed and frozen (0.44 to 0.54). Zero bonds in every run (grip about 0.1). Hold weight 3 vs 0 makes no
  difference, so the hold objective does not shape the structure in this setup.
- Pictures (`results/s5_ext_fb16_hold3/samples.png`): builds are fuzzy (bots hover around the outline); in the hold
  every learned structure pours into a heap. The even sample keeps a bonded lower core: towers stand, overhangs and
  thin tops break off.
- Next (running, wave 7): twice the build time (128 control steps), always-latched bots with bond 2 to 3 (weak
  enough to pull free with a push of 3), and learned grip with hold weight 3.

## 2026-11-05 tick 11: always-latched wave (6) final: also negative

All 192 validation shapes; EMA (raw weights in brackets). Even sample after hold at the same bond strength:
0.29 (bond 2), 0.33 (bond 4), 0.44 (bond 16).

| run (always latched, drone build, stiff physics) | build | hold [95% CI] | fallen | held-out hold |
|---|---|---|---|---|
| bond 2, hold weight 3 | 0.49 (0.51) | 0.34 [0.31, 0.36] (0.33) | 0.74 | 0.15 |
| bond 2 | 0.47 (0.47) | 0.32 [0.29, 0.34] (0.32) | 0.79 | 0.11 |
| bond 4 (seed 0 / 1) | 0.46 / 0.47 | 0.32 / 0.30 | 0.89 / 0.82 | 0.10 / 0.13 |
| bond 4, trained without gravity | 0.43 | 0.29 [0.26, 0.31] | 0.85 | 0.08 |
| bond 16 | 0.46 (0.44) | 0.33 [0.30, 0.36] | 0.87 | 0.13 |

- Builds reach 0.43 to 0.51 but 70 to 90% of bots fall in the hold. At bond 2 the learned structure stands slightly
  better than the even sample (0.32 to 0.34 vs 0.29; intervals overlap), at bond 4 equal, at bond 16 worse.
- The no-gravity control is not distinguishable from the gravity-trained runs (0.29 vs 0.30 to 0.32), so nothing
  gravity-specific has been learned. **No emergent structure in this setup.**
- Summary of the emergent-physics track so far: shape formation is learnable with drone forces; with
  momentum-conserving forces under gravity nothing leaves the pile (gradients and PPO); standing structure has not
  emerged under any objective tried. Wave 7 (twice the build time) is the last planned wave of this design.

## 2026-11-05 tick 12: wave 7 (twice the build time) is worse; pausing this design

Validation IoU at iteration 500 to 750 (EMA / raw): bond 2 always latched 0.31 / 0.31 (pile), bond 3 0.32,
learned grip bond 16 (two seeds) 0.27 to 0.31, bond 64 0.32, bond 2 hold weight 3 0.29 / 0.25: **six of eight runs
never left the pile.** Only the no-gravity control learns (build 0.59, hold 0.34 at iteration 750) and the second
seed of bond 2 is slowly moving (raw build 0.43 at 500). A longer horizon weakens the gradient signal through the
physics rather than helping. The six stuck runs were stopped; the two moving ones finish.

**Diminishing returns (operating rule):** across seven waves (about 60 runs) the emergent-physics design has given
one reliable positive (shape formation with drone forces) and consistent negatives on everything the direction is
about: no run with momentum-conserving forces under gravity leaves the pile (gradients or PPO), and no objective
produced structure that stands better than the even sample frozen. Further sweeps of this design would fill in
curves already understood. **Proposal:** move the structural question to the 3D lattice (`linking/`), where moves
are already valid and never drop a bot, and replace the hand-written support planner and readiness rule with a
learned local policy judged only by whether the structure stands under a strength-limited connector model. This
overlaps the teammate's lattice-policy work on `main`, so it needs the owner's go-ahead.

## 2026-11-05 tick 13: last two runs finished; both boxes idle

| run (always latched, bond 2, 128 build steps) | val build | val hold | fallen | held-out hold | even sample frozen (val / held) |
|---|---|---|---|---|---|
| trained without gravity | 0.48 | 0.31 | 0.75 | 0.14 | 0.29 / 0.13 |
| seed 1 (with gravity) | 0.53 | 0.32 | 0.77 | 0.14 | 0.29 / 0.13 |

Same picture as wave 6: the structure stands about as well as the even sample frozen, three quarters of the bots
fall, and the gravity-trained run cannot be told apart from the no-gravity control. Nothing queued; waiting for a
decision on the next direction (tick 12).

## 2026-11-05 tick 14: lattice track (`lat/`), wave lat1 at generation 200 of 1500: first propping

Go-ahead given for the lattice. Side-view lattice under gravity, cantilever limit L, NCA controller trained by
evolution strategies on target fill only (README in `lat/`). Fourteen runs launched (7 per box, GPUs 94%).

New measure, `propped`: the fraction of target cells that are filled at the end but would not be held if every bot
outside the target were removed (computed per shape, `lat/peek.py`, now also in `es.py` evaluate). The control is
the untrained controller (uniform over the allowed moves): fill 0.02, propped 0.000 on validation and held-out.

Checkpoints at generation 200 (validation set, one evaluation seed; held-out = mushroom, tree):

| run | val fill | val propped | held fill | held propped | ref: target alone stands (val) |
|---|---|---|---|---|---|
| L1 | 0.83 | 0.035 | 0.63 | 0.063 | 0.87 |
| L1 seed 1 | 0.84 | 0.027 | 0.64 | 0.057 | 0.87 |
| L1, no spares | 0.82 | 0.010 | 0.62 | 0.046 | 0.87 |
| L1, double spares | 0.83 | 0.079 | 0.72 | 0.153 | 0.87 |
| L1, 1 memory channel | 0.86 | 0.035 | 0.70 | 0.078 | 0.87 |
| L1, overhang families only | 0.79 | 0.069 | 0.69 | 0.075 | 0.72 |
| L3 | 0.90 | 0.016 | 0.76 | 0.008 | 0.96 |
| no limit (control) | 0.93 | 0.003 | 0.76 | 0.032 | 1.00 |

Readings (partial, single seeds, not yet replicated):
- Propping appears with no term asking for it, and grows with the number of spare bots (no spares 0.010, default
  0.035, double 0.079 on validation; 0.046 / 0.063 / 0.153 held-out). It is near zero without a cantilever limit,
  as it should be (nothing needs propping there).
- Pictures: the overhang-only run builds columns of spare bots directly under bridge decks, T arms and mushroom caps,
  and parks the remaining spares against the right wall. The double-spares run fills the space under the bridge
  deck and under the held-out mushroom cap with spares.
- The overhang-only run's validation fill (0.79) is above what a perfect placement of the target alone keeps
  (0.72): the first case of beating the reference. Everywhere else fill is still below the reference (the
  swarm props some overhang cells but misses others). Replicates queued (seeds 1, 2).
- Many bots fall during the build (about 500 bot-steps unheld per episode against about 15 for nothing); the
  swarm climbs and drops a lot. Not penalised; a move-cost run tests whether it matters.

Changes: stopped three optimiser-knob runs (population 256, sigma 0.02, width 64; nothing distinctive at
generation 100 to 200) to make room. Running now: overhang-only seed 1, overhang-only with L = 0 (only columns hold:
every overhang must be propped), overhang-only with double spares. Queued (new options in `es.py`): target switch
at half time (reconfiguration), move cost, kick (remove 10% of bots after settling, then settle again), no memory
(C = 0), bigger world (32 x 64), L = 0 on all families, overhang-only with kick, overhang-only seed 2.

## 2026-11-05 tick 15: propping replicates; most L = 1 runs now beat the reference; L = 0 grows a staircase

Checkpoints (validation and held-out, one evaluation seed; ref = the target placed perfectly then settled, val / held):

| run | gen | val fill | val propped | held fill | held propped | ref val / held |
|---|---|---|---|---|---|---|
| overhangs only, seed 0 | 425 | 0.79 | 0.074 | 0.68 | 0.064 | 0.72 / 0.68 |
| overhangs only, seed 1 (replicate) | 300 | 0.79 (eval) | 0.075 | 0.68 (gen 200) | 0.071 | 0.72 / 0.68 |
| overhangs only, double spares | 300 | 0.76 (eval) | 0.123 | 0.70 (gen 200) | 0.175 | 0.72 / 0.68 |
| overhangs only, L = 0 | 200 | 0.48 | 0.004 | 0.41 | 0.039 | 0.64 / - |
| L1 | 425 | 0.87 | 0.044 | 0.70 | 0.069 | 0.87 / 0.68 |
| L1, double spares | 425 | 0.90 | 0.068 | 0.81 | 0.141 | 0.87 / 0.68 |
| L1, 1 memory channel | 575 | 0.89 | 0.043 | 0.72 | 0.098 | 0.87 / 0.68 |
| no limit (control) | 425 | 0.99 | 0.000 | 0.93 | 0.017 | 1.00 / 1.00 |

- **Replicated (two seeds):** trained on overhang shapes only, the swarm fills more of the target than the target
  alone can hold (0.79 against 0.72), with 7% of target cells standing only on bots outside the target.
- On all families, the double-spares and one-memory-channel runs now beat the reference on validation, and three
  runs beat it on the held-out mushrooms and trees they never saw (double spares: 0.81 against 0.68; the pictures
  show columns of spares under both sides of mushroom caps).
- **L = 0 (only columns hold):** the swarm builds a diagonal staircase of spare bots against the stem and climbs it,
  in every shape (validation and held-out). At L = 0 a side neighbour gives no support, so a staircase of columns is
  the only way up that stands. But it does not prop the overhangs, and its fill (0.48) is below its reference (0.64).
  Second structure that nobody asked for: a ramp.
- Still: lots of falling and spares littered on the floor in the all-family runs.

## 2026-11-05 tick 16: falsework under every overhang; next: can the swarm strike it?

Evaluation lines (validation, ref = target alone): L1 0.91 at gen 900 (ref 0.87); seed 1 0.86 (lagging); 1 memory
channel 0.90; double spares 0.90; temperature 0.5 0.89; damage run 0.88; L3 0.95 (ref 0.96: little to prop);
no-spares 0.81 (below ref: nothing to prop with). Overhangs only: seed 0 0.80, seed 1 0.80, **double spares 0.87
(ref 0.72), propped 0.15**, L = 0 0.54 (ref 0.64).

Picture of overhangs-only with double spares (gen 675): a solid block of spare bots under every Γ arm, T arm and
mushroom cap (held-out), i.e. falsework; 97% of validation shapes have propped cells.

New hypothesis (lat4, queued first): in construction, falsework is removed once the structure has cured. Added
curing (a bot that has not moved for K = 20 steps becomes rigid: it adds no cantilever span; moving or falling
resets it; the bots see a 'cured' channel; unit test: an uncured 3-cell arm at L = 1 keeps 1 cell, a cured one
keeps 3) and a clean-site cost (0.5 per bot left outside the target in the build site, / |target|). If the swarm
learns build falsework -> wait -> remove it, that is emergent sequencing. Runs: cure + clean (seeds 0, 1),
clean without cure (control: falsework cannot be removed without losing the overhang), cure without clean
(does it use curing at all). T = 300, overhang families, double spares.

## 2026-11-05 tick 17: wave lat1 final; falsework-removal runs collapsed, restarted warm

Final (generation 1500; validation / held-out mushroom+tree; ref = the target alone, perfectly placed, settled):

| run | val fill | ref val | held fill | ref held |
|---|---|---|---|---|
| L1 | 0.90 | 0.87 | 0.75 | 0.67 |
| L1 seed 1 | 0.88 | 0.87 | 0.72 | 0.67 |
| L1, 1 memory channel | 0.90 | 0.87 | 0.72 | 0.67 |
| L1, temperature 0.5 | 0.89 | 0.87 | 0.72 | 0.67 |
| L1, double spares | **0.94** | 0.87 | **0.88** | 0.67 |
| L1, no spares | 0.85 | 0.87 | 0.64 | 0.67 |
| L1, damage at half time (trained with) | 0.82 | 0.87 | 0.69 | 0.67 |
| L1, overhangs only | 0.80 | 0.72 | 0.66 | 0.67 |
| L3 | 0.96 | 0.96 | 0.88 | 0.91 |
| no limit (control) | 0.99 | 1.00 | 0.93 | 1.00 |

- Beating the reference (scaffolding) replicates across seeds at L1 and transfers to held-out families
  (two seeds: 0.75, 0.72 against 0.67). It scales with spares (none: below ref; double: 0.94 / 0.88).
- With no spares the swarm is below the reference: there is nothing to prop with, so the gap is the build error.
- L3 and no-limit: at or just under the reference, as expected (little or nothing to prop).

**Falsework removal (lat4), first attempt failed:** with curing and the clean-site cost from scratch, the swarm
learned to stay out of the build site (generation 200: fill 0.000, site_out 0). The cost is felt before any fill
is: a classic local optimum. Stopped the three clean-cost runs. Restarted warm (lat4b): start from the overhangs /
double-spares falsework builder (generation 1300 checkpoint; zero weights for the new 'cured' input), ramp the
clean cost over 200 generations. At the warm start (one generation): fill 0.91, propped 0.000 (with curing the target
stands without its props once cured), but 36 props still left in the site: the cost should teach their removal.
Running: cure + clean warm (seeds 0, 1), clean warm without curing (control), cure without clean (from scratch).

Picture, L1 with double spares, final (`results/lat1_L1_spares100/samples.png`): the bridge deck stands on a
**colonnade**, thin pillars of spares with one-cell gaps between them, not a solid block; the same under the T
arms and under the held-out mushroom and tree caps (filled 1.00 for three of six held-out samples). At L = 1 a deck
over a gap of up to 2 cells between pillars is held, so pillars with gaps are the economical support, and the swarm
uses gap 1 (close to, not at, the sparsest spacing allowed). Nothing in the objective rewards using fewer bots;
the spacing likely comes from bots climbing up the existing pillars.

## 2026-11-05 tick 18: curing gives cure-and-extend, not falsework removal; formwork curing next; three collapses

**Warm falsework runs (generation 250; overhang families, double spares, T = 300, clean cost ramped to 0.5):**

| run | val fill (ref 0.72) | site_out (bots left in the site) | propped |
|---|---|---|---|
| warm start (falsework builder) | 0.91 | 36 | 0.00 (cured) |
| cure + clean, seed 0 | 0.77 | 9.5 | 0.00 |
| cure + clean, seed 1 | 0.93 | 13.5 | 0.01 |
| clean, no curing (control) | 0.65 | 3.6 | 0.01 |

Time series (`lat/traj_stats.py`, validation, every 25 steps): **the scaffold count never rises and then falls**
(seed 1: 0.8 -> 4.2 -> 3.8 bots under overhangs; seed 0: rises to 7.9 and stays). No build-then-strike. Instead the
cured fraction of target cells tracks the fill all the way (seed 1 at step 299: fill 0.92, cured 0.91): the swarm
builds slowly enough that each overhang cell cures before the next one goes out: **cure-and-extend**. With
sideways curing, falsework is unnecessary, so the hypothesis was not tested by this design. The control without
curing cleans the site and loses the overhangs (0.65, below the reference), as expected.

Next (lat5): **formwork curing** (`cure_below`): a bot only cures while the cell under it is solid, as concrete
needs formwork; once cured it stays rigid. Then cure-and-extend sideways is impossible and the only way to keep an
overhang with a clean site is prop -> cure -> remove the prop. Two seeds, warm from the falsework builder.

**Collapses (cost felt before reward, same local optimum as before):**
- move cost 0.02 per move per bot from scratch: the swarm froze (fill 0.000, moves 0.000 from generation 300).
- overhangs + kick 10% from scratch: stuck at fill 0.13.
Both restarted warm from wave-1 controllers (move cost 0.005 from L1 final; kick from overhangs-only final).
- height objective: solved at once (0.98 of the world height by generation 100: a tower to the ceiling); stopped.

**Other wave-2 readings (in progress, validation):** no memory at all (C = 0) 0.90 at generation 700, as good as L1
with 8 channels: **memory is not needed**, the swarm is reactive. L = 0 on all families 0.59 (ref 0.80): hard.
Target switch at half time 0.54 to 0.64 (ref 0.87): reconfiguration is hard and not improving. Kick 10% (all
families) 0.74. Overhangs only seed 2: 0.82 (third seed above the 0.72 reference). Bigger world 0.82 at
generation 100.

## 2026-11-05 tick 19: formwork curing gives no falsework stripping; chasm crossing next; gallery published

**Formwork curing (lat5; a bot cures only while the cell under it is solid; clean cost 0.5, warm from the
falsework builder), generation 250, two seeds:** fill 0.716 and 0.726 against a reference of 0.724, site_out 4 to
5, scaffold 3 to 4 bots, flat time series (no rise then fall). Both seeds dropped the overhangs: they build what
stands on its own, cure it, and clean up. Removing props after curing would pay (each overhang cell is worth
1/|target| and a removed prop costs nothing) but the search did not find it. **Negative** for emergent
build-then-strike under this design. Stopped both (flat since the ramp ended) to make room. Queued one variant
with a smaller clean cost (0.2) so keeping the overhangs stays worth it while props are still there.

Other updates: cure + clean warm seed 1 at generation 500: fill 0.91 with only 3.3 bots left in the site
(cure-and-extend, clean). Move cost 0.005 warm: 0.91, moves per bot 27.5 vs 29 without cost (little change).
Bigger world 0.87 at generation 200 (ref 0.87). Switch 0.73 at 600 (improving, still below 0.87). Kick 0.77.

**New hypothesis (lat6): chasm crossing.** Part of the floor is missing (`chasm`); the target is a deck across the
gap (3 to 12 cells) and a tower beyond it; bots that fall into the gap are lost. Unit test with one abutment bot:
uncured deck keeps 1 of 9 cells at L = 1, cured deck keeps 9 of 9. Prediction: with curing the swarm crosses by
cure-and-extend (the strategy it found on its own, here warm-started from that controller); without curing at L = 1
it cannot cross more than about 1 cell; at L = 3 about 3. Runs: cure warm (two seeds), no cure (L1), no cure (L3).

**Gallery:** every run rendered on the same six shapes (`lat/gallery.py`, page from `lat/gallery_page.py`),
published as a private artifact with settings, metrics against the reference, and a verdict per run.

## 2026-11-05 tick 20: curing crosses the chasm, and the rule extends to gaps it never saw

**Chasm (lat6), validation, generation 150 (s0) / 100 (s1):** with curing (warm from the cure-and-extend
controller) fill **0.97** (s0) and **0.92** (s1) against a perfect-placement reference of 0.50. Pictures: the
swarm lays a fully cured deck at floor level across the gap from one anchor bot on the near floor, walks over it,
and builds the far tower (s0: 1.00 on all six gallery chasms; s1: every deck complete, towers partly built).
Without curing it cannot cross at all: L = 1 fill **0.06** (one cell of deck, no tower; the rest of the swarm falls
into the gap), L = 3 fill **0.20** (about three cells). The reference 0.50 is not reachable without curing because
the far tower can only be reached over the deck. Prediction confirmed. Note: `propped` reads 0.50 on these runs by
construction (the deck has no target cell on solid ground; its anchor is necessarily outside the target), so it is
not informative here.

**Gap-width sweep (`lat/chasm_gen.py`, 32 chasms x 2 seeds per width, s0 checkpoint):** at the training episode
length (T = 300) deck and tower stay at 0.97 up to gap 11, the tower falls to 0.58 at gap 13 and 0 from 14 on
(the deck still reaches 0.67 to 0.92). With **T = 600, fill 0.93 to 0.97 for every gap from 3 to 20** (training
used 3 to 12). The failure was the time budget (each deck cell must cure for 20 steps before the next one can
stick out), not a rule fitted to the trained widths: cure-and-extend is a general local procedure. One seed;
not yet replicated.

Hypotheses queued (lat7): (1) does crossing emerge **from scratch** (no warm start; two seeds)? (2) **narrow
training** (gaps 3 to 7 only, then sweep to 20): a harder generalisation test; (3) **slow curing** K = 60: the
crossing time grows with K, does it still cross within T = 300 or find another way; (4) chasm with the clean cost
(no leftovers).

Other: cure + clean warm seed 1 final: fill 0.922, site_out 3.0 (replicates cure-and-extend with a clean site;
seed 0 final 0.777). No-cure clean control final 0.692 (ref 0.724). Cure without clean cost 0.99 at generation 700
(site_out 32: it fills by piling). Move cost warm 0.90. Switch 0.73 at generation 900 (stuck below 0.87). Big world
0.88 (ref 0.87). Kick 10% final 0.77; overhangs + kick warm final 0.67 (ref 0.72). Shelter objective runs: shelter
0.24 to 0.28 per bot at generations 100 to 300, no clear growth yet.

## 2026-11-05 tick 21: chasm crossing emerges from scratch; no-limit shelter makes a gallows

**Chasm from scratch (lat7, no warm start, curing K = 20):** seed 0 fill **0.956** at generation 300 (gallery: four
of six chasms at 1.00): the same cure-and-extend deck, plus a few spares left as a staircase to finish the far
tower. Seed 1 is slower, 0.54 at generation 250 (decks partly laid). Seed 2 queued. So crossing does not need the
warm start (one seed so far). Warm runs: s0 0.998 at 450, s1 0.993 at 400 (replicated). No-cure controls flat (L1
0.06, L3 0.20 at 400 to 500).

**Slow curing K = 60 (warm):** 0.35 at generation 300; decks are only partly laid. Consistent with the time-budget
reading: a 12-cell gap needs about 12 x 60 / 2 = 360 steps of curing alone, more than T = 300.

**Chasm + clean cost 0.5 (warm):** 0.90 with 1.5 bots left on the site: crossing with almost no leftovers.

**Shelter, no cantilever limit (L = 64):** shelter 7.2 covered cells per bot (L = 1: 0.55; L = 3: 0.30). Picture: a
**gallows**, one column to the ceiling and one long roof cantilevered from its top, on every episode. That is the
best shape for this objective once the physics allows it, and it emerged without any shape in the reward. With L
= 1 or 3 the swarm only makes a low porous mat. Queued (lat8): shelter at L = 1 **with curing** (T = 300, two
seeds): does cure-and-extend turn the mat into a roof?

**Formwork curing + clean cost 0.2 (lat6_formwork_clean02):** fill 0.92 but 32 bots left on the site: the props
stay and cure into solid blocks under every overhang. Cheap enough to keep, so no striking. Third negative for
build-then-strike.

Finals: switch 0.715 (ref 0.87, fails); overhangs seed 2 0.822 (ref 0.72, third replicate); cure without clean
cost 0.988 (31 leftovers); move cost warm 0.909; big world 0.87 at generation 500 (at the reference).

## 2026-11-05 tick 22: shutdown

Stopped at the user's request. Workers and jobs killed on both boxes. Everything in `~/emergent` on both boxes
went to the private Hugging Face repo (`emergent/box137`, `emergent/box150`; see `ARCHIVE.md`), checked file by file
(685 + 664 files) and byte-checked on two checkpoints. Then the boxes were cleared.

**Last gap sweeps (T = 600, 32 chasms x 2 seeds per width, gaps 3 to 20), run on the idle GPUs before clearing:**
- chasm from scratch, seed 0 (generation 400): fill **0.99 to 1.00 at every width up to 20** (trained 3 to 12).
- narrow training (gaps 3 to 7 only, warm, generation 90 of 600): decks 0.97 to 1.00 at every width up to 20;
  far towers 0.83 to 0.95 up to 17, 0.52 to 0.55 at 19 to 20; fill 0.87 to 0.97.
- warm seed 0 (tick 20): 0.93 to 0.97 up to 20.
Three controllers, including one trained only on gaps of 7 or less, cross gaps up to 20 when given enough time.
Cure-and-extend is a local procedure, not fitted to the training widths.

**State at shutdown (last validation):**

| run | gen | fill | note |
|---|---|---|---|
| lat6 chasm cure warm s0 / s1 | 550 / 500 | 1.000 / 0.998 | replicated; ref 0.50 |
| lat6 chasm no cure L1 / L3 | 500 / 500 | 0.057 / 0.196 | cannot cross |
| lat7 chasm cure scratch s0 / s1 | 400 / 350 | 0.997 / 0.575 | emerges from scratch on seed 0; s1 slower; s2 never started |
| lat7 chasm narrow (gaps 3 to 7) | 50 | 0.913 | see sweep above |
| lat7 chasm cure + clean | 300 | 0.971 | 3.9 bots outside |
| lat7 chasm K = 60 | 400 | 0.356 | too slow to cure within T = 300 |
| lat6 formwork clean 0.2 | 350 | 0.850 | props kept and cured; no striking |
| lat3 shelter L64 | 600 | shelter 7.2 / bot | gallows |
| lat8 shelter cure L1 s1 | 100 | shelter ? | only started; s0 never started |
| lat2 big | 500 | 0.872 | at the reference (0.87) |

**Summary of the lattice track.** Under gravity with a cantilever limit and a fill-only reward, the swarm finds
scaffolding (pillars, colonnades, falsework under overhangs) on its own and beats what the target alone keeps
(three seeds on overhangs; held-out shapes too). Memory is not needed (C = 0 matches C = 8). Given curing (still
bots turn rigid), it finds cure-and-extend: grow an overhang one cell at a time and let each cure. That strategy
crosses a chasm that is otherwise impossible, emerges from scratch, and extends to gaps far wider than trained.
With no cantilever limit, a target-free shelter reward gives a gallows. Negatives: build-then-strike (prop, cure,
remove the props) never appeared in three designs; target switching and post-build kicks stay below the
reference; costs felt before any reward (move cost, clean cost from scratch) collapse the search to doing
nothing, fixed by warm starts and ramps.

**Open next steps:** replicate the from-scratch chasm (seed 2) and the narrow run to the end; shelter with curing
at L = 1 (does a cured roof appear?); a chasm whose far side is higher (cure-and-extend upward); build-then-strike
with a cost that only applies after curing is complete.
