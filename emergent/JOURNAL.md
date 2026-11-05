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
