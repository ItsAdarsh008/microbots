# Journal (newest last)

## 2026-10-29 first two Stage 1 runs (single seed each, NOT replicated)

Setup: 10,000 train shapes, N = 256 particles, 30k iterations, flow matching with in-set OT pairing, map read at own position. Eval: 128 val shapes, 128 held-out-family shapes (stars, rings), 12 emoji x 8 noise draws. Units: particle spacings (sp).

| run | val w2 | held-out family w2 | emoji w2 | collisions (val) | notes |
| --- | --- | --- | --- | --- | --- |
| noise (chance) | 9.74 | 7.08 | 8.77 | 0.06 | |
| classical (no learning, has the map) | 1.45 | 1.31 | 1.60 | 0.094 | slide to shape + local repulsion |
| **global attention**, 6 layers | **1.24** | **1.14** | **1.40** | **0.027** | 22 min |
| **local r=0.25, 2 rounds** | 1.29 | 1.64 | 1.68 | 0.024 | 14 min |
| floor: second independent sample of the same shape | ~0.81 | ~0.68 | ~0.69 | 0 | approximate (64-grid resample) |

Controls: wrong-map test passes (global: error 1.14 against the map it was given, 3.23 against the other shape; local: 1.29 vs 3.17). Reference data scores inside 1.00 / cover 1.00 / collisions 0. Chance and projection rows behave as expected.

What it shows:
- With a map in hand, flow matching forms the shapes: skill 0.84 to 0.88 against chance, and 4 Euler steps already match 64 (w2 1.23 vs 1.24). Beats the no-learning baseline by about 15% in w2 and cuts collisions from 9% to 3%; measured against the floor (about 0.8) the model sits 0.4 above it and the baseline 0.65 above it.
- It is not a clean win: 9% of particles end more than half a spacing outside the shape and 6% of the shape pixels have no particle within one spacing, where the classical baseline is exactly 1.00 on both by construction (it projects onto the shape).
- Held-out families (star, ring) are no worse than in-distribution for the global model (1.14 vs 1.24; the family difference itself is confounded with shape area, not tested).
- The local model (r = 0.25, only 2 message rounds) matches global in-distribution (1.29) but degrades on held-out families (1.64) and emoji (1.68, worse than the classical baseline), and makes more holes (0.34 per shape vs 0.17 for real shapes) and fewer connected shapes (0.92 vs 1.00). This is the pattern Kamb and Ganguli's theory predicts for local denoisers (right patches, wrong global topology). One seed, one radius, one depth: it is a hypothesis, not a finding.
- Training loss is about 2x lower for global (0.0044) than local (0.0100). Not a result by itself, since it is an in-training number.

Hypotheses raised:
- H1: the local gap on unseen shapes closes with more message rounds or a larger radius (queued: local r=0.25 with 4 rounds, r=0.15 with 4 rounds).
- H2: the model needs no clock (queued: time-free run). Not yet known.
- H3: the generative approach earns its keep on the *unconditional* task and on the hard-to-engineer cases, not on conditional shape filling where a 20-line classical rule is already close. Queued: unconditional global and local.
- H4: held-out shape error falls with training-set size and saturates before 10^4 (queued: 100 and 1,000 shapes).
- Replicate: seed 1 of the global run is queued before anything is claimed from the 1.24 vs 1.45 gap.

## 2026-10-29 tick 1: local-denoiser ladder (single seed each, NOT replicated)

Held-out-family w2 in spacings (classical baseline 1.31, global model 1.14, floor ~0.7):

| run | val w2 | held-out family w2 | emoji w2 | connected shapes |
| --- | --- | --- | --- | --- |
| global, 6 layers | 1.24 | 1.14 | 1.40 | 1.00 |
| local r=0.25, 2 rounds | 1.29 | 1.64 | 1.68 | 0.92 |
| local r=0.25, 4 rounds | 1.27 | 1.63 | 1.65 | 0.95 |
| local r=0.15, 4 rounds | 1.33 | 1.70 | 1.71 | 0.75 |
| local r=0.25, 4 rounds, **no clock** | 1.65 | 1.86 | 1.90 | 0.07 |

All wrong-map controls pass (error stays low against the map fed, about 3.2 against the other shape).

What it shows:
- Locality costs nothing on shapes like the training ones (1.27 to 1.33 vs 1.24) and costs about 0.5 spacings on unseen families and emoji, where it falls behind the no-learning baseline (1.63 vs 1.31).
- More message rounds did not close the gap (2 rounds 1.64, 4 rounds 1.63). A smaller radius hurt (0.15: 1.70, connected shapes down to 0.75). So the limit looks like radius, not depth. Single seed; the 2-vs-4 difference is far inside what a seed could move.
- Removing the clock hurts a lot: val w2 1.27 -> 1.65 and only 7% of shapes come out as one piece. The samples show the bodies form but stray particles stay scattered around them.

Hypotheses:
- H5: the strays are isolated particles that cannot tell how far along they are without `t` and with few neighbours. A larger radius should help (queued: r=0.4 with and without clock).
- H6: the local gap on unseen shapes is a radius effect (queued: r=0.4 with clock, and a second seed of r=0.25).
- H7 (design): if H5 holds, a broadcast clock is a real hardware requirement, and the cheap fix is a coarse global beacon rather than a clock.

## 2026-10-29 tick 2: replicate, shuffled-label control, first unconditional run

**Replicate (global, conditional, seed 1):** val w2 1.33 (seed 0: 1.24), held-out family 1.12 (1.14), emoji 1.41 (1.40). The global model is therefore about 1.2 to 1.35 on familiar shapes and 1.1 on held-out families against 1.31 for the no-learning baseline: the held-out-family and emoji advantage over the baseline repeats across two seeds; on familiar shapes (1.24 and 1.33 vs 1.45) it is smaller and I would not call it settled. Collisions 2 to 3% vs 9%, both seeds.

**Control, shuffled labels (map paired with the wrong shape in training):** val w2 3.07, the same as the unconditional model (3.10), as it should be: the improvement comes from the map, not from the model memorising shapes. The wrong-map control passes for every conditional run.

**Unconditional global (the "any shape from noise" case, one seed):**
- w2 against a reference is meaningless here (there is no target); the 3.1 equals the shuffled-label control, i.e. "a plausible but unrelated shape".
- Against real validation shapes: 99.6% of samples are one connected piece (real 100%); radius of gyration, elongation, extent and centring distributions match real ones (Wasserstein distances 0.006 to 0.01); best overlap with a training shape 0.848 vs 0.852 for real validation shapes, so no sign of copying.
- Mismatches: 3.5x more holes than real shapes (0.49 vs 0.14 per shape), and nearest-neighbour spacing is tighter (0.043 vs 0.056).
- By eye (`results/s1_uncond_global/grid.png`, 48 distinct samples): compact blobs, wedges, bars, a few with holes; few branching or L-shaped forms that the real set has. Visual impression only, no measure behind it yet.

Hypotheses:
- H8: the unconditional model averages towards compact shapes (mode-averaging) and under-produces branching ones. Needs a measure (e.g. convexity, number of arms) before it is a claim. Queued: a 60k-iteration run to see if it is under-training.
- H9: the excess holes are interior gaps from imperfect packing rather than real holes; check by looking at hole size.

## 2026-10-29 tick 3: radius, clock and training-set size

Errors in particle spacings (no-learning baseline 1.45 familiar / 1.31 held-out family / 1.60 emoji; floor ~0.7 to 0.8).

| run | familiar | held-out family | emoji | connected | collisions | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| global, 10k shapes | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 | 1.00 | 0.03 | 2 |
| local r=0.25, 4 rounds | 1.27, 1.29 | 1.63, 1.65 | 1.65, 1.68 | 0.95, 0.94 | 0.02 | 2 |
| local r=0.40, 4 rounds | 1.30 | 1.29 | 1.51 | 0.97 | 0.04 | 1 |
| local r=0.25, no clock | 1.65 | 1.86 | 1.90 | 0.07 | 0.05 | 1 |
| local r=0.40, no clock | 1.44 | 1.55 | 1.65 | 0.61 | 0.09 | 1 |
| global, only 100 shapes | 1.35 | 1.01 | 1.33 | 1.00 | 0.04 | 1 |

All wrong-map controls pass.

What it shows:
- **Replicated:** the local denoiser at r=0.25 is about 0.5 spacings worse than global on unseen shape families (1.63, 1.65 vs 1.14, 1.12). Rounds don't fix it (2 vs 4 rounds equal).
- **Radius matters (one seed):** at r=0.4 the unseen-family error drops to 1.29, close to the baseline (1.31) and global, and connectivity returns to 0.97. The cost is a slight rise in collisions (0.04 vs 0.02). Replicate queued.
- **Clock (one seed each):** without a clock the larger radius helps a lot (connected 0.07 -> 0.61, familiar 1.65 -> 1.44) but does not fully recover and collisions triple (0.09). Consistent with H5 (isolated particles lose track of progress), not yet replicated.
- **Surprise, training-set size (one seed):** a map-conditioned global model trained on only 100 shapes does as well as the 10k one on unseen families (1.01) and emoji (1.33), and slightly worse on familiar shapes (1.35). This does not support "10,000 shapes are needed" for the conditional task. The 1,000-shape run is finishing; 10-shape and a second 100-shape seed are queued.

Hypotheses:
- H10: with the map given at every particle, the task is mostly "go where the signed distance is negative and spread out", which a handful of shapes can teach; the dataset size matters for the *unconditional* task, where the model has to learn the shape distribution itself. Queued: unconditional with 100 shapes (expect copying: high overlap with training shapes against real validation shapes).
- H11: a coarse global signal (bigger radius, or a beacon) is what a clock-free local swarm needs.

## 2026-10-29 tick 4: unconditional with local-only communication (one seed)

`s1_uncond_local_r25_L4`: no map, each particle sees only neighbours within 0.25, 4 message rounds, no global attention. Compare with `s1_uncond_global` (same data, global attention).

| | global | local r=0.25 | real val shapes |
| --- | --- | --- | --- |
| samples that are one connected piece | 0.996 | **0.676** | 1.00 |
| mean connected pieces | 1.00 | 1.41 | 1.00 |
| holes per shape | 0.49 | 0.56 | 0.14 |
| elongation (aniso; real 0.675) | 0.67 | **0.80** (W1 0.12) | 0.675 |
| best overlap with a training shape | 0.848 | 0.799 | 0.852 |

By eye (`results/s1_uncond_local_r25_L4/grid.png`): ragged blobs with detached bits and stray particles; the long thin bars and L-shapes of the real set are mostly missing.

What it shows (one seed, so a hypothesis): when there is no map to tell particles where the shape is, a local-only denoiser cannot settle on one coherent global shape. A third of samples fragment and the shapes are rounder than real. This is the failure that Kamb and Ganguli's theory predicts for local denoisers (locally plausible pieces, wrong global structure), and it is what the Stage 1 ladder expected: a shape that nobody communicates cannot be agreed on by neighbours only.

Hypotheses:
- H12: connectivity of unconditional local samples rises with radius (queued: r=0.4).
- H13: for a real swarm, the target must be broadcast in some form (map, code or coarse density); "noise to any shape" with purely local communication is not what a swarm can do. The cheap thing to test is how little needs broadcasting: a low-resolution map (queued next if H12 fails).

## 2026-10-29 tick 5: three more GPUs, radius replicated, training-set curve so far, robustness

**Compute.** Five A100-40GB boxes now. Allocation: boxes 1 and 2 finish the earlier queue (replicates, unconditional runs, clock-jitter training); box 3 = how little must be broadcast (`coarse` maps of 4x4, 8x8, 16x16 numbers; local and global); box 4 = training-set-size curve (conditional 30/300/3000 shapes, unconditional 1000/3000); box 5 = radius curve (0.6, 0.8), global model without a clock, replicates.

**Replicated: radius closes the local gap (2 seeds).** Held-out-family error: r=0.25 gives 1.63 and 1.65, r=0.40 gives 1.29 and 1.25, global gives 1.14 and 1.12, no-learning baseline 1.31. Familiar shapes: r=0.40 1.30 and 1.34. Connected shapes at r=0.40: 0.97 and 0.98. A local denoiser with a larger sensing radius is about as good as the baseline on unseen families and about 0.15 worse than global; rounds (2 vs 4) made no difference.

**Training-set size, conditional global model (1 seed each unless stated):**

| shapes | familiar | held-out family | emoji |
| --- | --- | --- | --- |
| 100 | 1.35 | 1.01 | 1.33 |
| 1,000 | 1.26 | 1.11 | 1.32 |
| 10,000 (2 seeds) | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 |

The curve is flat from 100 to 10,000 shapes for the map-conditioned task (differences are within what seeds do). Second seed of the 100-shape run and runs at 10, 30, 300 and 3,000 are in progress. H10 (shape count matters for the unconditional task, not the conditional one) is still untested; unconditional 100/1,000/3,000 runs are queued.

**Robustness (S1d-lite, `robust.py`, no retraining, one model per row, 96 val shapes, 16 steps).** Collisions (fraction of particles closer than 0.35 spacings to a neighbour); base value first:

| model | base | clock jitter 0.1 | clock jitter 0.25 | half the particles update each round | motion noise 0.3 sp | 128 / 512 particles |
| --- | --- | --- | --- | --- | --- | --- |
| global | 0.02 | 0.13 | 0.22 | 0.11 | 0.04 | 0.01 / 0.06 |
| local r=0.25 | 0.02 | 0.04 | 0.04 | 0.10 | 0.04 | 0.02 / 0.05 |
| local r=0.40 | 0.04 | 0.07 | 0.11 | 0.12 | 0.06 | 0.01 / 0.08 |
| local r=0.40, no clock | 0.08 | 0.08 | 0.08 | 0.14 | 0.11 | 0.04 / 0.25 |

- The headline error barely moves under these perturbations (w2 1.17 to 1.23 for global under jitter) while collisions rise 6 to 10 times: error alone hides it. Keep collisions in every table.
- Models handle half or double the particle count they were trained on (coverage and collisions stay near base), apart from the time-free one at 512 particles (coverage 0.81, collisions 0.25).
- Asynchronous updates cost every model about 0.15 to 0.2 spacings of error and raise collisions to 10 to 14%.

**CORRECTION.** The first version of the asynchronous test had a bug: each update used a step 1/p too large, so particles overshot by 2x. It showed catastrophic failures (cover 0.14, collisions 0.96 for the time-free model) and led me to the wrong hypothesis (H14: "async fails because the clock disagrees with each particle's progress"). With the step fixed, the failure is mild and the time-free models are no better than clocked ones, so H14 is rejected. The numbers above are from the fixed script.

Hypotheses:
- H15: models trained with every particle on one shared clock are fragile to clock errors. Train each particle with its own noisy time (`tjit`; the target velocity is unchanged). Queued for global, local r=0.25 and local r=0.40; evaluated with `robust.py` afterwards.
- H16: the global model's jitter fragility (collisions 0.22) is an artefact of attention across all particles reading many inconsistent clocks; the local models, which see fewer particles, are less affected (consistent with the table; one seed each).
- H13 stays open: coarse-map runs on box 3 test how few numbers need broadcasting.

## 2026-10-29 tick 6: no-clock global model, unconditional runs, training-set curve, rounds at r=0.4

**Global model without a clock (1 seed):** val 1.35, held-out family 1.20, emoji 1.45, 99% connected shapes, collisions 0.05 (with clock: 1.24 and 1.33 / 1.14 and 1.12 / 1.40 and 1.41, collisions 0.03). Global attention does not need `t`. Local models without a clock broke apart (connected 0.07 at r=0.25, 0.61 at r=0.4). Reading (hypothesis, consistent with H5): the clock is only needed when a particle cannot see enough of the configuration to tell how far along it is; global attention sees all of it.

**Rounds at r=0.4 (1 seed):** 2 rounds give val 1.27, held-out family 1.63, emoji 1.63, collisions 0.14; 4 rounds give 1.30 / 1.29 / 1.51 with collisions 0.04. At r=0.25 rounds made no difference (2 vs 4: 1.64 vs 1.63). So at a large radius two rounds are not enough; replicate queued.

**Training-set size, conditional global model.** Held-out-family error, one entry per seed:

| shapes | familiar | held-out family | emoji |
| --- | --- | --- | --- |
| 10 | 1.22 | 1.21 | 1.55 |
| 100 | 1.35, 1.35 | 1.01, 1.17 | 1.33, 1.34 |
| 1,000 | 1.26 | 1.11 | 1.32 |
| 10,000 | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 |

The two 100-shape seeds differ by 0.16 on held-out families, so my earlier remark that 100 shapes "beat" 10,000 was seed noise. What the table supports: between 100 and 10,000 shapes nothing visible changes for the map-conditioned task; 10 shapes are as good on familiar and held-out families but worse on emoji (1.55). Runs at 30, 300 and 3,000 are in progress to locate the knee.

**Unconditional (the "noise to any shape" task), one seed each:**

| run | one piece | holes per shape (real 0.14) | elongation (real 0.675) | collisions | solidity mean (real 0.956) | best overlap with a training shape (real 0.852) |
| --- | --- | --- | --- | --- | --- | --- |
| global, 30k iterations | 0.996 | 0.49 | 0.671 | 0.14 | 0.945 | 0.848 |
| global, 60k iterations | 0.992 | 0.22 | 0.689 | 0.06 | 0.958 | 0.856 |
| local r=0.25 | 0.676 | 0.55 | 0.797 | 0.05 | 0.885 | 0.799 |
| local r=0.40 | 0.949 | 0.30 | 0.758 | 0.06 | 0.940 | 0.841 |

- H9 (excess holes are under-training) is supported for the global model: doubling iterations roughly halves the holes (0.49 to 0.22) and collisions (0.14 to 0.06). One seed; replicate queued.
- H12 (radius fixes local unconditional fragmentation) is supported by one seed: 68% to 95% of samples in one piece. Local samples remain rounder than real ones (elongation 0.76 vs 0.675). Replicate and a 60k run queued.
- H8 (the unconditional model collapses towards compact blobs) is **not supported** by the one measure I have: real validation shapes are themselves compact (solidity 0.956, only 3% below 0.8) and the 60k global model matches that (0.958, 1%). The measure is blunt (it dilates the particles by twice the median spacing, which fills thin concavities), so thin arms could still be under-produced. No memorisation signal: best overlap with training shapes equals that of real validation shapes (0.856 vs 0.852).

Hypotheses:
- H17: with 10 training shapes the conditional model only fails on the shapes most unlike its training set (emoji), so the knee is around 30 to 100 shapes. Tested by the 30 and 300-shape runs in progress.
- H18: a generator with more shapes that are long and thin or branched (low solidity) would be needed before the unconditional task can be called generalising; the current set is mostly compact by construction. Next: add a "skinny and branched" family and a held-out version of it.

## 2026-10-29 tick 7: radius curve, how little to broadcast, clock replicate, clock-jitter training

**Radius curve (conditional, 4 rounds). Held-out-family error, one entry per seed; no-learning baseline 1.31, global 1.12 and 1.14:**

| radius | held-out family | familiar | collisions (val) | connected |
| --- | --- | --- | --- | --- |
| 0.15 | 1.70 | 1.33 | 0.02 | 0.75 |
| 0.25 | 1.63, 1.65 | 1.27, 1.29 | 0.02 | 0.95 |
| 0.40 | 1.29, 1.25 | 1.30, 1.34 | 0.04 | 0.97 |
| 0.60 | 1.16 | 1.31 | 0.06 | 0.98 |
| global | 1.14, 1.12 | 1.24, 1.33 | 0.03 | 1.00 |

Error keeps falling with radius up to 0.6, where it matches global (one seed), but collisions rise steadily (0.02 to 0.06). Radius 0.8 is running. For a real swarm the radius is a hardware cost (sensing range, number of neighbours to talk to), so the useful reading is a trade-off curve, not a "best" radius.

**Clock, replicated.** Local r=0.4 without a clock: connected shapes 0.59 (seed 1) and 0.61 (seed 0); error 1.49 and 1.44 familiar. So local models need a clock (2 seeds); the global model does not (1 seed: 1.35 / 1.20 / 1.45).

**How little must be broadcast (global attention, the swarm is given only a g x g occupancy grid instead of the full map, 1 seed each):**

| broadcast | numbers | familiar | held-out family | emoji | collisions |
| --- | --- | --- | --- | --- | --- |
| full map (64x64 plus distance field) | 4,096+ | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 | 0.03 |
| 8 x 8 | 64 | 1.38 | 1.23 | 1.41 | 0.03 |
| 4 x 4 | 16 | 1.49 | 1.58 | 1.71 | 0.04 |

Caveat: the no-learning baseline cannot use these coarse maps, so it stays at its full-map numbers (1.45 / 1.31 / 1.60). A swarm told only 64 numbers (an 8x8 occupancy grid) still forms unseen held-out shapes about as well as with the full map (1.23 vs 1.13). 16 numbers is too few. This supports H13 for the global model; the local versions (r=0.25 at 4, 8, 16 and r=0.4 at 8) are running and are the ones that matter for a swarm.

**Training-set curve completing (1 seed each unless noted), held-out family:** 10: 1.21, 30: 0.99, 100: 1.01 and 1.17, 300: 1.18, 1,000: 1.11, 10,000: 1.14 and 1.12. Flat across three orders of magnitude within seed noise (about 0.15); emoji only worse at 10 shapes (1.55). 3,000 is running. So for a map-conditioned task a few dozen shapes suffice; the open question is the unconditional task (runs at 100, 1,000 and 3,000 are in progress).

**Clock-jitter training did not help (and I tested the wrong thing).** I trained with every particle given its own noisy time in the *input and the position* (`tjit`: particles at different stages of progress, each told its true stage). Result: nominal collisions went up (r=0.25: 0.02 to 0.08, r=0.40: 0.04 to 0.12) and under clock jitter the models got worse than the ordinary ones (r=0.25 at jitter 0.25: error 2.08, collisions 0.14, vs 1.51 and 0.04 before). That training did not match the test: the test gives particles a *wrong* clock reading at the right position. H15 as designed is rejected. The matching training is noise on the clock reading only (`tnoise`); queued for global and local r=0.4. Local r=0.25 was robust to clock errors already (collisions 0.04 at jitter 0.25), which may make the fix unnecessary there.

**New data (v2).** Added thin branching families: trees (training) and spiders (held out), all with arms 0.04 to 0.08 wide. Jobs on this set (global, local r=0.25, local r=0.40, unconditional) are queued on box 5. They test whether the conclusions so far hold on shapes whose parts are narrower than the sensing radius (H18).

Hypotheses:
- H19: local denoisers fall furthest behind global on v2 (thin arms hold fewer neighbours), so the radius needed will be larger there.
- H20: if H13 holds for the local models too, the swarm needs one 64-number broadcast and nothing else; that is the cleanest statement of what the generative approach could give a physical swarm.

## 2026-10-29 tick 8: radius plateau, unconditional local replicated, broadcast with local models, memorisation check fixed

**Radius curve, last point.** Held-out family: r=0.6 1.16, r=0.8 1.22 (both 1 seed; global 1.12 and 1.14; baseline 1.31); r=0.8 gives 100% connected shapes and collisions 0.05. Error stops improving beyond about r=0.6, so about 6 particle spacings of sensing radius is where local ties global in this setup.

**Replicated: unconditional local r=0.4.** Second seed: 95% of samples are one piece (first seed 95%), holes 0.27 (0.30), elongation 0.755 (0.758). Radius 0.25 gave 68%. Two seeds now support "radius lifts local unconditional connectivity from 68% to 95%". Samples remain rounder than real (elongation 0.755 vs 0.675), so local unconditional is not equal to global unconditional (elongation 0.67, 99% one piece).

**Training-set size, conditional global, held-out family, 3,000 shapes:** 1.16 (val 1.24, emoji 1.40). The curve is flat from 10 to 10,000 shapes; seed spread about 0.15.

**Unconditional, training-set size (1 seed each, global attention):**

| shapes | one piece | holes (real 0.14) | elongation (real 0.675) | best overlap with a training shape: generated / real unseen |
| --- | --- | --- | --- | --- |
| 100 | 0.988 | 0.27 | 0.651 | 0.774 / 0.763 |
| 1,000 | 0.992 | 0.26 | 0.671 | 0.829 / 0.823 |
| 10,000 (60k iterations) | 0.992 | 0.22 | 0.689 | 0.865 / 0.856 |

No sign of copying at any size, and the descriptor statistics are nearly the same at 100 and 10,000 shapes. H10 (the shape count matters for the unconditional task) is therefore **not supported**, but the descriptors I use may be too coarse to see a difference; I would not conclude that the shape distribution is learned equally well. (Also: after the 100-shape run showed 0.85 overlap I found my memorisation check compared against all 10,000 training shapes, not the ones the model was trained on. The numbers in the table come from the fixed check, `memo.py`; earlier `iou` columns for the 100- and 1,000-shape runs in `collect.py` are the old, wrong comparison and should be ignored. `evalm.py` is fixed for new runs.)

**How little must be broadcast, local models (1 seed each; global with the same maps and full-map references in the earlier table):**

| model | broadcast | familiar | held-out family | emoji | one piece |
| --- | --- | --- | --- | --- | --- |
| local r=0.25 | full map | 1.27, 1.29 | 1.63, 1.65 | 1.65, 1.68 | 0.95 |
| local r=0.25 | 16 x 16 | 1.38 | 1.74 | 1.68 | 0.77 |
| local r=0.25 | 4 x 4 | 1.50 | 1.83 | 1.83 | 0.84 |

At r=0.25 a coarse map costs 0.1 to 0.2 in error and 10 to 20 points of connected shapes. The r=0.25 model was already the weak local setting; the r=0.4 (8x8) and r=0.6 (8x8) runs decide whether H20 holds for a swarm.

Hypotheses:
- H21: the unconditional task is not hard in the way the plan assumed: even a model trained on 100 shapes samples plausible compact shapes without copying. The difficulty will appear when the generator contains shapes that cannot be produced by a smooth average (the v2 thin-branch data) or when the target is specified by something other than a map.
- H22: a swarm that starts as a pile (where real bots start) is a different problem from one that starts as scattered noise: paths from a compact pile to a shape are shorter, but particles start overlapping. Pile-start runs (global and local r=0.4) are queued. They also use the same evaluation (the chance row becomes the pile itself).
- H23 (design): flow-matching sampling takes 4 steps as well as 64 (earlier tick); combined with pile starts this is the first thing that looks like a feasible bot controller: 4 to 8 rounds of communication-and-move.

## 2026-10-29 tick 9: rounds at r=0.4 replicated, coarse broadcast with a local model, first thin-shape (v2) results

**Rounds at r=0.4 (2 seeds):** 2 rounds give held-out-family error 1.63 and 1.60, 4 rounds give 1.29 and 1.25. So at a large radius the number of message rounds matters (it did not at r=0.25). The collision spike I saw in the first 2-round run (0.14) did not replicate (second seed 0.04); it was seed noise.

**Local model, 64-number broadcast (1 seed each; full-map references in the radius table):**

| model | broadcast | familiar | held-out family | emoji | one piece | collisions |
| --- | --- | --- | --- | --- | --- | --- |
| global | 8 x 8 | 1.38 | 1.23 | 1.41 | 1.00 | 0.03 |
| local r=0.40 | full map (2 seeds) | 1.30, 1.34 | 1.29, 1.25 | 1.51, 1.52 | 0.97 | 0.04 |
| local r=0.40 | 8 x 8 | 1.36 | 1.38 | 1.59 | 0.91 | 0.05 |
| local r=0.25 | 8 x 8 | 1.38 | 1.70 | 1.66 | 0.81 | 0.02 |

A swarm of bots that each sense 0.4 around them and are told only an 8x8 occupancy grid (64 numbers) forms unseen held-out families at 1.38 against 1.31 for a no-learning baseline that gets the full-resolution map. That is one seed; a replicate is queued. It supports H20 for r=0.4 (broadcast cost drops from thousands of numbers to 64 at a price of about 0.1 in error and a few points of connected shapes); not for r=0.25.

**Thin branching shapes (v2: tree in training, spider held out; arms 0.04 to 0.08 wide). One seed each; baselines recomputed on v2:**

| | familiar | held-out family (stars, rings, spiders) | emoji | collisions (val) | one piece |
| --- | --- | --- | --- | --- | --- |
| no-learning baseline (full map) | 1.71 | 1.96 | 1.60 | 0.105 | n/a |
| global | 1.41 | 1.47 | 1.42 | 0.047 | 0.99 |
| local r=0.25 | 1.51 | 1.97 | 1.65 | 0.023 | 0.91 |

- The global model's advantage over the baseline grows on thin shapes (held-out family 1.47 vs 1.96, 25% lower; on v1 data the margin was 1.13 vs 1.31, 14%). The baseline struggles with thin shapes; the global model does not.
- Local r=0.25 only ties the baseline on held-out families (1.97 vs 1.96) and puts 22% of particles outside the shape there (inside 0.78). H19 (thin arms need a larger radius) is supported in direction by this one seed; r=0.4 and r=0.6 runs are queued/running.
- Wrong-map control passes (1.38 against the map fed, 3.75 against the other shape).

Hypotheses:
- H24: on thin shapes the local models fail because arms are narrower than the sensing radius so few neighbours are on the shape, not because of the held-out family; test: report error by family (spider vs star vs ring) for the v2 held-out set (needs a per-family split in the evaluator; not yet built).
- H25: a swarm needs the radius to scale with the thinnest feature it must form; for hardware, report required radius per shape class.

## 2026-10-29 tick 10: broadcast knee, noisy-clock training works, physical sampler, per-family errors

(Compute: 5 A100 GPUs, two jobs at a time on each = 10 job slots. Earlier entries that say "runs going" count jobs.)

**Broadcast, global model (held-out family; full-map references 1.14, 1.12):** 8x8 (64 numbers) 1.23 and 1.22 (2 seeds); 6x6 (36 numbers) 1.27 (1 seed); 4x4 (16 numbers) 1.58 (1 seed). The knee is between 16 and 36 numbers. Local r=0.60 with 8x8: 1.29 (full map 1.16), 98% one piece.

**Noisy-clock training (`tnoise` 0.15; clock reading wrong, position right; 1 seed each).** Collisions under clock jitter 0.25 are 0.02 for global (the ordinary global model: 0.22) and 0.05 for local r=0.4 (ordinary: 0.11); nominal error is unchanged for global (1.28 / 1.17 / 1.38) and a little worse for local r=0.4 (1.32 / 1.33 / 1.53, 86% one piece vs 97%). H15 in its corrected form is supported: train with the failure you expect. Asynchronous updates still raise collisions to 0.09 (global) and 0.14 (local).

**Physical sampler (`physical.py`, 1 model each, Gaussian start, 96 val shapes).**

| variant | global w2 | collisions | local r=0.4 w2 | collisions |
| --- | --- | --- | --- | --- |
| 8 steps | 1.16 | 0.024 | 1.19 | 0.032 |
| 32 steps | 1.18 | 0.025 | 1.23 | 0.038 |
| 32 steps + hard-core spacing 0.7 | 1.17 | 0.000 | 1.23 | 0.000 |
| 32 steps, speed cap 1 spacing per step | 1.38 | 0.025 | 1.40 | 0.037 |
| 64 steps, speed cap 0.5 spacings per step | 1.39 | 0.026 | 1.41 | 0.038 |

- A hard-core spacing rule applied after every step removes all collisions at no cost in error.
- Mean path is 7.7 spacings per particle, but a cap that allows 32 spacings in total costs 0.2 in error: the cost comes from the tail of particles that start far away (a Gaussian start reaches about 30 spacings out). A pile start (queued results) should shorten that tail. Eight steps are as good as 64.

**Per-family held-out error (32 steps, 1 seed per model; spacings):**

| model | star | annulus | spider (thin, v2 only) |
| --- | --- | --- | --- |
| global (v1) | 0.90 | 1.27 | n/a |
| local r=0.25 (v1) | 1.25 | 1.91 | n/a |
| local r=0.40 (v1) | 1.07 | 1.43 | n/a |
| global (v2) | 1.01 | 1.26 | 2.00 |
| local r=0.25 (v2) | 1.25 | 1.96 | 2.77 |
| local r=0.40 (v2) | 1.07 | 1.46 | 2.52 |

- H24 supported in direction: thin spiders are the hardest family for every model (global 2.00, with 23% of particles outside the shape), and local models fall furthest behind there (2.52 to 2.77). Rings (annulus) are the next hardest and are where radius matters most (1.91 at r=0.25 vs 1.43 at r=0.4): a ring is a thin curved band, so a particle sees few neighbours on the shape.
- The v2 headline number for local r=0.4 (1.69 held-out) averages these; stars are fine, spiders are not. The thin-arm limit is real even for the global model, and it is not a broadcast limit, since these models were given the full map.

Hypotheses:
- H26: thin arms narrower than a few spacings fail because the 256 particles cannot resolve them (spacing vs arm width), not because of the architecture; test by giving the spiders more particles (N = 512) at test time with the same model (it handled 512 in the robustness test) (training at N = 512 would need the dataset rebuilt with 512 points per shape; not done).
- H27: a pile start shortens the long-travel tail and removes the speed-cap cost; pile runs are in progress.

## 2026-10-29 tick 11: thin shapes replicated, pile start, a naming mistake

**Naming mistake.** `s2p_cond_global_pile`, `s2p_cond_local_r40_L4_pile` and `s2p_cond_global_pile_seed1` were queued without `"data":"shapes2"`, so they trained and were evaluated on the **original (v1) shapes**, despite the `s2` in their names; compare them with the v1 rows. `s2p_cond_local_r40_g8_pile` and `s2p_cond_local_r60_g8_pile` do use the thin-shape set (v2). Nothing was wrong with the numbers, only with my reading of the names. Also: a job failed because box 4 had no copy of the v2 data; copied and requeued.

**Thin shapes (v2), replicated.** Error in spacings, baseline (full map) 1.71 familiar / 1.96 held-out family / 1.60 emoji:

| model | familiar | held-out family | emoji | seeds |
| --- | --- | --- | --- | --- |
| global | 1.41, 1.47 | 1.47, 1.41 | 1.42, 1.42 | 2 |
| local r=0.40 | 1.46, 1.50 | 1.69, 1.68 | 1.48, 1.53 | 2 |
| local r=0.25 | 1.51 | 1.97 | 1.65 | 1 |

On thin and branched shapes both global and local r=0.4 beat the baseline on held-out families (2 seeds each; global by 25 to 28%, local r=0.4 by 14%); local r=0.25 does not (1 seed).

**Local r=0.4, 64-number broadcast, replicated (v1 data):** held-out family 1.38 and 1.43, familiar 1.36 and 1.39, 91 to 93% one piece. Versus the no-learning baseline with the *full* map: better on familiar shapes (1.37 vs 1.45), about 0.1 worse on held-out families (1.40 vs 1.31).

**Pile start (a random pile of radius 0.7; 1 seed each; v1 data unless noted). Gaussian-start numbers from earlier ticks in brackets:**

| model | familiar | held-out family | emoji | collisions |
| --- | --- | --- | --- | --- |
| global | 1.26 (1.24, 1.33) | 1.42 (1.14, 1.12) | 1.60 (1.40, 1.41) | 0.20 (0.03) |
| local r=0.40 | 1.46 (1.30, 1.34) | 1.82 (1.29, 1.25) | 1.78 (1.51, 1.52) | 0.27 (0.04) |
| v2 data: local r=0.40, 8x8 broadcast | 1.83 | 2.28 | 1.84 | 0.26 |
| v2 data: local r=0.60, 8x8 broadcast | 1.77 | 2.17 | 1.81 | 0.27 |

- The random pile is itself 23% collisions (Poisson clumps; bots cannot overlap like that), and the learned flow keeps them: collisions stay at 0.19 to 0.27 at every step count. The pile start is worse than Gaussian noise by 0.3 to 0.5 on held-out shapes, most for local models, and a coarse broadcast on top compounds it.
- Physical sampler on the pile model (`physical.py`): the speed cap now costs nothing (error 1.32 with and without a cap of 1 spacing per step; mean path 6.1 spacings, vs 7.7 from Gaussian noise), which supports H27. Hard-core spacing (0.7) again removes all collisions for 0.01 of error. So pile + speed cap 0.5 spacings per step + 64 steps + hard-core = error 1.34, no collisions. This is the first complete set of physical constraints that works for the global model (1 seed).

Hypotheses:
- H28: the poor pile result is mostly the overlapping random pile, not the pile idea: a well-spaced pile (what a real pile of non-overlapping bots would look like) should close most of the gap. Queued: global, local r=0.4, and local r=0.4 with 64-number broadcast, all on the thin-shape data (so their baselines are the v2 rows above).
- H29: with a physically realisable start the local models need larger radius than from Gaussian noise (start is dense and nobody can see the shape's outline); to check once H28 results are in.

## 2026-10-29 tick 12: unconditional replicated, 64-number broadcast on thin shapes

Compute: 5 A100 GPUs, 9 of 10 job slots in use; boxes 1 and 2 had run dry and were refilled (see below).

**Unconditional global, 60k iterations, replicated (2 seeds, v1 data):** one piece 0.992 and 0.992, holes 0.22 and 0.19 per shape (real 0.14), elongation 0.689 and 0.681 (real 0.675), collisions 0.06 and 0.05. H9 (the excess holes in the 30k run were under-training) now holds on two seeds.

**Unconditional global on the thin-shape data (v2, 60k, 1 seed):** 99.6% in one piece, size and elongation match (0.676 vs 0.679), but 0.56 holes per shape against 0.30 for real shapes; real v2 shapes contain loops more often (trees), and generated ones show about twice as many gaps. Not examined shape by shape yet.

**64-number broadcast on thin shapes, global model (v2, 1 seed):** familiar 1.49, held-out family 1.59, emoji 1.46 (full map: 1.41 / 1.47 / 1.42, two seeds; baseline with full map 1.71 / 1.96 / 1.60). Cost of cutting the broadcast from the full map to 64 numbers is about 0.1 on this harder set, and it still beats the no-learning baseline's held-out error by 19%.

**New runs queued to fill idle slots:**
- `s1_cond_local_r25_L4_60k` and `s1_cond_local_r25_L4_d384`: is the r=0.25 local gap on unseen shapes a training-length or capacity problem rather than a radius problem? (H30: no. Prior: 2 vs 4 rounds made no difference, and the r=0.4 model at the same size and length is fine.)
- `s2pf_cond_local_r60_L4_pf` and `s2pf_cond_global_g8_pf`: well-spaced pile start with a larger radius and with a 64-number broadcast (H28/H29).

## 2026-10-29 tick 13: self-repair is not free; repair has to be trained in

Compute: 5 A100 GPUs; this tick I loaded the one idle GPU (box 5) with repair-trained models (below).

**New results.** Thin shapes (v2), local r=0.60: familiar 1.41, held-out family 1.54, emoji 1.56, 97% one piece, collisions 0.07 (1 seed). The radius trend on thin shapes: r=0.25 1.97, r=0.40 1.69 and 1.68 (2 seeds), r=0.60 1.54, global 1.47 and 1.41; baseline 1.96. Pile start (random pile, v1 data), global model, seed 1: 1.49 / 1.67 / 1.60 with collisions 0.20; seed 0 was 1.26 / 1.42 / 1.60 with 0.20. The pile result is noisy (0.25 between seeds) but both seeds are worse than Gaussian start on held-out shapes (1.12 and 1.14) and both keep 20% collisions.

**Self-repair test (`repair.py`, no retraining; 64 val shapes; form the shape, then damage it; 1 seed per model). Cover = fraction of shape pixels with a particle within one spacing (spacing for the surviving particle count). Formed shapes: cover 0.90 to 0.93.**

| model | damage | cover after damage | cover after rewinding the clock to 0.5 | collisions after: before / after |
| --- | --- | --- | --- | --- |
| global | cut a chunk (disc r=0.3), 204 bots left | 0.77 | 0.81 | 0.04 / 0.06 |
| local r=0.40 | chunk | 0.78 | 0.75 | 0.06 / 0.09 |
| local r=0.60 | chunk | 0.77 | 0.76 | 0.08 / 0.10 |
| global | random 20% removed | 0.92 | 0.88 | 0.03 / 0.03 |
| local r=0.40 | random 20% | 0.92 | 0.85 | 0.04 / 0.06 |
| global, no clock | chunk, kept stepping | 0.75 | 0.14 | 0.07 / 0.97 |

- Rewinding the clock does not refill a cut chunk (cover 0.75 to 0.81, no gain) and after a random 20% loss it makes things worse (0.92 to 0.85 to 0.88). The flow sends particles from noise to the shape; particles that are already on the shape get squeezed together instead of spread out.
- The no-clock model is not stationary: stepping it past completion collapses the swarm into a clump (collisions 0.97, cover 0.14). So the "no clock" results above are results about forming a shape, not about holding one.
- So repair is not a free property of this sampler (unlike what the NPA work was trained for). H31: it has to be trained in.

**What I built:** `damage_p` in the trainer: with that probability a batch starts from the shape with a contiguous chunk of bots removed (192 to 240 bots left, fewer than the 256 normally used), and the target is the same shape at that bot count, evenly spread. Three models queued on box 5 (global, local r=0.6, local r=0.4, all with damage_p = 0.5, v1 data). Evaluation: `repair.py` restarts the clock at 0 for these and also reports the old rewind variants. Prediction (to be tested, not a result): cover after repair rises above 0.9 while formation from scratch stays at 1.2 to 1.3 in error.

## 2026-10-29 tick 14: CORRECTION to tick 13, well-spaced pile, local-gap controls

**CORRECTION (tick 13).** That entry says the three repair-trained models were "queued on box 5". They were not: the connection to box 5 dropped ("Network is unreachable") during that step and I wrote the entry without checking. This tick found all five boxes idle for that reason, and the repair runs (`s3_*`) were queued and verified running only now.

**Is the local gap at r=0.25 a training-length or size problem? (conditional, v1, held-out family; 1 seed each)** r=0.25 baseline run 1.63 and 1.65 (2 seeds); 60,000 iterations 1.52; hidden width 384 (4x the parameters) 1.54. Both controls help a little (about 0.1) and neither closes the gap to r=0.40 (1.29 and 1.25) or global (1.12 and 1.14). H30 is mostly supported: the gap is mainly a sensing-radius effect; training length and model size account for at most about a quarter of it.

**Well-spaced pile start (bots begin evenly spread in a disc of radius 0.7, no overlaps; thin-shape data v2; 1 seed each).** Compared with the same models from Gaussian noise (v2: global 1.41 and 1.47 held-out; r=0.4 1.69; r=0.6 1.54):

| model (pile start) | familiar | held-out family | emoji | collisions | one piece |
| --- | --- | --- | --- | --- | --- |
| global | 1.57 | 1.93 | 1.82 | 0.01 | 1.00 |
| global, 64-number broadcast | 1.67 | 2.02 | 1.83 | 0.01 | 1.00 |
| local r=0.40 | 1.73 | 2.13 | 1.82 | 0.00 | 1.00 |
| local r=0.40, 64-number broadcast | 1.75 | 2.22 | 1.83 | 0.00 | 1.00 |
| local r=0.60 | 1.71 | 2.11 | 1.86 | 0.00 | 1.00 |

- H28 (the overlapping random pile was the problem) is **half right**: collisions drop from 0.20-0.27 to 0.00-0.01 and every shape stays in one piece, but error from a pile is still 0.4 to 0.6 worse on held-out shapes than from scattered noise, and a larger radius no longer helps (r=0.4 2.13, r=0.6 2.11). Going from a compact pile to a shape is a bigger global rearrangement (for a ring the bots must move outwards and leave the middle empty), not just local attraction.
- **Baseline from the same start.** The no-learning rule (project onto the shape, then local repulsion) run for 200 steps from this pile gets 4.57 / 5.40 (familiar / held-out family) with 19 to 23% collisions: it has not finished spreading. I reran it with more steps (`classical_pile.py`): 600 steps 3.13 / 3.82, 1500 steps 2.03 / 2.42 with about 9% collisions. The learned global model reaches 1.51 / 1.79 in 8 steps and 1.57 / 1.93 in 64, with 1% collisions, so from a pile it is better than the baseline's 1500 steps by 0.4 to 0.5 in error and uses 190x fewer steps. Caveats: the baseline rule is untuned, and it scores cover 0.99 and inside 1.00 (it projects onto the shape) where the learned model gets cover 0.92 and inside 0.8 (about 20% of particles end more than half a spacing outside the shape). This is a 1-seed result.
- My reading in the earlier tick that the pile results "tie the baseline" used the wrong reference (the Gaussian-start baseline). Corrected here.

Hypotheses:
- H32: pile-start difficulty is coordination, not sensing: a bigger model or longer training should help the global model if so (queued: global 60k iterations; seed replicates of global and local r=0.6).
- H33: all the physical pieces together (pile start, 64-number broadcast, noisy clock, local radius 0.6) are within a factor 1.2 of the best Gaussian-start local model (queued: r=0.6, 8x8 broadcast, noisy clock training from a pile).
- H31 (repair has to be trained in) is being tested: `s3_cond_global_dmg`, `s3_cond_local_r60_L4_dmg`, `s3_cond_local_r40_L4_dmg`, plus a pile-start and repair model.

## 2026-10-29 tick 15: repair works when trained in; CORRECTION of the tick-14 correction

**CORRECTION (of tick 14).** Tick 14 says the repair-trained runs "were not queued" in tick 13 and that all boxes were idle because of it. That was wrong. In tick 13 the jobs were submitted and ran on box 5 (global and local r=0.6 took 20 to 25 minutes each); the "Network is unreachable" error came from my final status check after they were already submitted. The boxes were idle at tick 14 because those runs (and the others) had finished. I then re-queued the same configs, same seed, on box 4, which would only have duplicated work; I stopped them and replaced them with seed-1 replicates (`s3_cond_global_dmg_seed1`, `s3_cond_local_r60_L4_dmg_seed1`). Lesson: check the queue and run directories before writing "not queued".

**Repair-trained models (`damage_p` = 0.5: half of the training batches start from a shape with a chunk of bots removed; Gaussian start otherwise; v1 data; 1 seed each).**

Formation from scratch (held-out-family error, spacings; earlier models: global 1.14 and 1.12, r=0.4 1.29 and 1.25): global 1.18, local r=0.40 1.34, local r=0.60 1.15; familiar 1.35 to 1.37. So training with repair batches costs nothing on formation.

Repair test (`repair.py`, 64 val shapes, 16 steps, cover = fraction of the shape's pixels with a bot within one spacing for the surviving bot count; clock restarted at 0):

| model | damage | cover before | cover after repair | with clock rewound to 0.5 instead |
| --- | --- | --- | --- | --- |
| global, repair-trained | cut chunk, 205 bots left | 0.75 | **0.98** | 0.89 |
| local r=0.60, repair-trained | cut chunk | 0.76 | **0.97** | 0.89 |
| global, repair-trained | random 20% removed | 0.89 | **0.99** | 0.95 |
| local r=0.60, repair-trained | random 20% | 0.91 | **0.98** | 0.96 |

H31 is supported (1 seed each; seed 1 replicates are running): once repair is part of training, a swarm with a cut-out chunk, or a fifth of its bots gone, refills the shape to cover 0.97 to 0.99 with the surviving bots, 0.07 above freshly formed shapes (0.90 to 0.91) because the same bots now spread evenly. Without it (earlier models) cover stayed at 0.75 to 0.81. The sampler needs the clock restarted at 0, not rewound.

**Cost: collisions.** The repair-trained models make more close pairs than ordinary ones: 0.15 (global), 0.19 (local r=0.6) and 0.21 (r=0.4) against about 0.03 to 0.06 before, in ordinary formation too. The damage-batch targets are not the cause (1 to 2% close pairs; stored reference 0%). The hard-core spacing rule fixes it again at no cost: global 32 steps 1.33 error and 0.15 collisions without it, 1.31 and 0.00 with it. A speed cap of one spacing per step again costs about 0.15 from Gaussian noise.

Hypotheses:
- H34: the collision rise comes from the model having to tell a nearly-formed start from noise at t=0 and compromising; a clock reading that distinguishes the two starts (e.g. starting repair at t=0.3 and noise at t=0) would test that.
- H35: repair should work from a pile-start model too (queued: `s3pf_cond_local_r60_L4_dmg_pf`, running).

## 2026-10-29 tick 16: repair replicated (2 seeds), pile start with repair, thin-shape radius trend replicated

Compute: 5 A100 GPUs, 10 job slots, all in use after this tick's refill.

**Thin shapes (v2), radius trend replicated.** Held-out-family error, one entry per seed (no-learning baseline with full map 1.96): global 1.47 and 1.41; local r=0.60 1.54 and 1.59; r=0.40 1.69 and 1.68; r=0.25 1.97 (1 seed). Familiar-shape error r=0.60: 1.41 and 1.53. The radius ordering holds on two seeds for r=0.4 and 0.6; r=0.6 is within about 0.15 of global and beats the baseline by 19%.

**Repair-trained models, replicated (Gaussian start, v1 data; 2 seeds each).** Formation from scratch, held-out family: global 1.18 and 1.18; local r=0.60 1.15 and 1.17 (earlier ordinary models: 1.12 to 1.14 and 1.16). Repair test (cover after cutting a chunk, clock restarted at 0; before repair was 0.75 to 0.76):

| model | seed | cover after repair, chunk cut | cover after repair, random 20% |
| --- | --- | --- | --- |
| global | 0 | 0.98 | 0.99 |
| global | 1 | 0.98 | 0.99 |
| local r=0.60 | 0 | 0.97 | 0.98 |
| local r=0.60 | 1 | 0.97 | 0.99 |

**H31 is replicated (2 seeds, two model types): training with damaged starts gives a swarm that refills a cut-out chunk to cover about 0.97 to 0.98 with 20% fewer bots, at no cost to ordinary formation.** Also replicated: these models have 0.15 to 0.21 collisions in ordinary formation (global seeds 0.15 and 0.17), removed completely by the hard-core spacing rule.

**Pile start plus repair (thin shapes, local r=0.60, well-spaced pile start, 1 seed).** Formation from the pile: 1.63 familiar / 2.01 held-out family / 1.67 emoji with collisions 0.00 (without repair training: 1.71 / 2.11 / 1.86). Repair test on it: cover after cutting a chunk 0.79 -> 0.97, after random 20% 0.92 -> 0.99, collisions 0.02 to 0.03. So repair from a physically realisable start works, and repair training did not hurt formation from the pile (slightly better, 1 seed; replicate queued).

**Queued to separate "coordination" from "capacity or length" for the pile start (H32):** global model with hidden width 384 (4x the parameters), local r=0.60 at 60,000 iterations, local r=0.40 second seed, and a second seed of the pile-plus-repair model. Still running: global 60k from a pile (first look at the end of this tick), pile seed 1 replicates, v1-data pile runs, and the everything-physical run.

## 2026-10-29 tick 17: pile-start cost replicated, localised to thin shapes, and the all-physical bundle

Compute: 5 A100 GPUs, 10 job slots; three boxes had run dry and were refilled.

**Pile start (well-spaced pile of radius 0.7 at a random spot), thin-shape data, replicated.** Held-out-family error, one entry per seed (Gaussian start in brackets): global 1.93 and 1.96 (1.47 and 1.41); local r=0.60 2.11 and 2.10 (1.54 and 1.59). The extra cost of starting from a pile is about 0.5 (2 seeds, both models). Collisions 0.00 to 0.01.

**Where it is lost (`perfam.py`, global model from a pile, 1 seed):** star 1.32 (from Gaussian start 1.01), ring/annulus 1.63 (1.26), spider 3.07 (2.00) with 43% of the spider's bots ending outside the shape (inside 0.57). Spiders have thin legs reaching 0.5 to 0.9 out; from a compact pile the bots must be pushed out along each leg, from Gaussian noise they only have to be pulled in. H36: the pile cost is mostly an extrusion problem on thin, far-reaching parts, not a general slowdown.

**Original-shape (v1) data, pile start (1 seed each; seed 1 queued).** Global 1.38 / 1.47 / 1.78 and local r=0.60 1.39 / 1.74 / 1.82 (familiar / held-out family / emoji), collisions 0.00 to 0.01, all shapes one piece. Gaussian start: global 1.24, 1.33 / 1.14, 1.12 / 1.40, 1.41; local r=0.60 1.31 / 1.16 / 1.45. So on blobby v1 shapes the pile costs about 0.3 to 0.6 on held-out ones, and radius matters again (r=0.60 is 0.27 worse than global from a pile).

**Everything physical at once (local r=0.60, 64-number broadcast, noisy clock training, well-spaced pile, thin shapes; 1 seed):** 1.76 / 2.21 / 1.84, collisions 0.00. Compared with the same local model without broadcast and noisy clock (2.11, 2.10 held-out) the two extra physical pieces cost about 0.1. So H33 in its original form (within 1.2x of the best Gaussian-start local model: 2.21 vs 1.54 is 1.44x) is **not met**; the shortfall is almost all the pile start. Against the no-learning rule from the same start (2.0 on familiar and 2.4 on held-out shapes after 1500 steps, 9% collisions) the learned bundle (1.76 and 2.21 in at most 64 steps, no collisions) is roughly level on error, with far fewer steps and no collisions.

Hypotheses and queued tests:
- H37: a pile centred on the arena (not at a random offset) and a larger, sparser pile (radius 0.9) are easier because less transport is needed (queued: `s2pfc_*` and `s2pfr_*`, global and local r=0.60, thin shapes).
- H32 (capacity or length) controls still running: wider network, 60k iterations, second seeds.

## 2026-10-29 tick 18: pile plus repair training replicated

Thin shapes, local r=0.60, well-spaced pile start, with repair batches (`damage_p` 0.5), held-out-family error: 2.01 and 2.00 (2 seeds); the same model without repair batches: 2.11 and 2.10 (2 seeds); familiar 1.63 and 1.61 vs 1.71 and 1.68; emoji 1.67 and 1.70 vs 1.86 and 1.86. Collisions 0.00. Training with damaged starts in the mix therefore does more than add repair: from a pile it improves formation by about 0.1 on held-out shapes and 0.17 on emoji (2 seeds).
H38: the damaged-start batches teach the "refine a nearly formed shape" half of the flow, which the pile start needs at the end. Queued to test the dose and the model type: damage fraction 0.8 (local r=0.60) and a global model with repair batches from a pile.

Still running: pile variants (centred pile, larger pile), wider network (d=384), 60k iterations (global and local), second seeds on v1 data, and r=0.40 second seed from a pile.

## 2026-10-29 tick 19: the pile cost is mostly distance, not density; a two-phase swarm closes it

Compute: 5 A100 GPUs, 10 job slots, all in use.

**Pile variants (thin shapes, v2; 1 seed each unless noted). Held-out-family error:**

| start | global | local r=0.60 |
| --- | --- | --- |
| scattered noise (Gaussian) | 1.47, 1.41 | 1.54, 1.59 |
| pile at a random offset (radius 0.7) | 1.93, 1.96 | 2.11, 2.10 |
| pile centred on the arena | **1.40** | **1.53** |

(Familiar-shape error from a centred pile: global 1.36, local 1.33.) The pile's density and compactness are not what costs error: a compact pile centred on the shape forms held-out shapes as well as scattered noise (1 seed; centred seed 1 queued). What costs 0.5 is that the random-offset pile starts up to 0.7 away from where the shape sits. H36 (extrusion of thin legs) is mostly **not** the explanation; the thin spiders were hard because they were far away.

**Two-phase swarm (`travel_eval.py`, no retraining, thin shapes, 32 steps, 96 shapes per set, 1 seed per model).** Phase 1: the offset pile travels as a rigid body to the shape's centre (read from the map; mean travel 6.8 spacings, longest 17.6). Phase 2: the model trained on a *centred* pile forms the shape.

| model | set | offset pile, no travel | offset pile after travel | centred pile (as trained) | trained directly on offset piles (earlier) |
| --- | --- | --- | --- | --- | --- |
| global | familiar | 9.49 | 1.60 | 1.38 | 1.57, 1.62 |
| global | held-out family | 9.04 | **1.53** | 1.32 | 1.93, 1.96 |
| local r=0.60 | familiar | 11.24 | 1.84 | 1.39 | 1.71, 1.68 |
| local r=0.60 | held-out family | 10.86 | **1.80** | 1.43 | 2.11, 2.10 |

Travel then form beats learning the whole move from an offset pile on held-out shapes by 0.3 to 0.4 (about the same on familiar shapes for global, 0.1 worse for local). Without the travel phase it fails completely, as expected from a model trained on centred piles. Collisions stay at 0.001 to 0.01. This is the cleanest controller for a real swarm so far: a trivial common travel (move towards the target site) followed by local shape formation.

**Capacity and length controls for the pile start (global, offset pile, 1 seed each; held-out family; baseline runs 1.93, 1.96):** 4x wider network 1.66 (familiar 1.49); 60,000 iterations 1.73 (1.42). So about half of the offset-pile penalty also goes away with capacity or training length. Wider plus 60k and a wider local model are running.

Hypotheses:
- H39: for a real swarm the right decomposition is rigid travel plus local formation, not one network doing both; the travel distance is small (about 7 spacings on average here) and needs no learning.
- H40: forming from a pile placed anywhere would be solved by translation equivariance in the model (give the model positions relative to the pile centre instead of absolute positions); to test after the capacity results.

## 2026-10-29 tick 20: v1 pile replicated, larger pile, repair batches with a global model

**v1 shapes, well-spaced pile at a random offset (2 seeds):** global 1.38 / 1.47 / 1.78 and 1.37 / 1.42 / 1.76 (familiar / held-out family / emoji); local r=0.60 1.39 / 1.74 / 1.82 (1 seed, second queued earlier). Gaussian start: global 1.24, 1.33 / 1.14, 1.12 / 1.40, 1.41. The pile cost on blobby shapes is about 0.3 on held-out ones and 0.37 on emoji, replicated for global.

**Larger, sparser pile (radius 0.9, thin shapes, offset as before; 1 seed each; radius-0.7 pile in brackets):** global 1.53 / 1.66 / 1.58 (1.57, 1.62 / 1.93, 1.96 / 1.82, 1.84); local r=0.60 1.65 / 2.04 / 1.75 (1.71, 1.68 / 2.11, 2.10 / 1.86). A bigger pile starts closer to every part of the shape and costs less (global 0.3, local 0.07 on held-out). Consistent with the distance reading (H37); second seeds queued.

**Repair batches with a global model from a pile (thin shapes, 1 seed):** 1.55 / 1.82 / 1.73 against 1.57, 1.62 / 1.93, 1.96 / 1.82, 1.84 without. Again about 0.1 better on held-out shapes, as for the local model (2 seeds); replicate queued.

**Capacity, length, dose still running:** wider global model for 60k, wider local model, local 60k, repair share 0.8; centred-pile seed 1.

## 2026-10-29 tick 21: centred pile and the two-phase swarm replicated; the reference swarm

**Centred pile (thin shapes, 2 seeds). Familiar / held-out family / emoji; Gaussian-start numbers in brackets:**

| model | centred pile | scattered noise (earlier) | offset pile (earlier) |
| --- | --- | --- | --- |
| global | 1.36, 1.31 / 1.40, 1.52 / 1.51, 1.35 | 1.41, 1.47 / 1.47, 1.41 / 1.42, 1.42 | 1.57, 1.62 / 1.93, 1.96 / 1.82, 1.84 |
| local r=0.60 | 1.33, 1.34 / 1.53, 1.47 / 1.44, 1.45 | 1.41, 1.53 / 1.54, 1.59 / 1.56, 1.57 | 1.71, 1.68 / 2.11, 2.10 / 1.86, 1.86 |

**H37 replicated (2 seeds, both models): a compact pile centred on the shape is as good as scattered noise; the cost of an offset pile is distance.**

**Two-phase swarm (rigid travel to the shape's centre, then form with the centred-pile model), replicated.** Held-out-family error after travel: global 1.53 and 1.51; local r=0.60 1.80 and 1.79; versus the models trained on offset piles 1.93, 1.96 and 2.11, 2.10. Familiar shapes after travel: global 1.60, 1.55; local 1.84, 1.90 (offset-trained: 1.57, 1.62 and 1.71, 1.68). So travel-then-form is better on unseen shapes (0.4 for global, 0.3 for local) and roughly level on familiar ones. H39 is supported on 2 seeds.

**Other replicates this tick:** v1 data, local r=0.60 from an offset pile 1.74 (2 seeds, both 1.74); local r=0.60 from an offset pile trained for 60k iterations 2.01 (1 seed; 2.10 to 2.11 at 30k).

**Reference swarm for the hand-off to Stage 2 (queued).** One model with everything that worked: local radius 0.60, only a 64-number (8x8) broadcast, noisy-clock training (0.15), repair batches (50%), trained from a centred pile, thin-shape data. Two seeds local, one global as the upper bound. To be evaluated as a whole: formation after rigid travel, repair after cuts, clock jitter, speed cap and hard-core spacing, particle count change; against the no-learning rule from the same start.

**Where Stage 1 stands.** The questions the ladder asked are answered at the level of 1 to 2 seeds: a flow-matching swarm forms unseen shapes from noise or a pile, local sensing needs a radius of about 0.6 (6 particle spacings), a 64-number broadcast is enough, repair needs to be trained in, a clock and a hard-core spacing rule are cheap fixes, and distance, not pile density, is what hurts from a pile. What remains is building the reference swarm and re-checking it end to end. After that, new sweeps would only fill in curves I understand; I propose to pause Stage 1 there and move to Stage 2, which needs the real bot numbers (diameter in particle spacings, top speed, sensing and communication radius, link strength).

## 2026-10-29 tick 22: pile controls replicated

Thin shapes, held-out-family error, one entry per seed. Reference: scattered noise global 1.47, 1.41; local r=0.60 1.54, 1.59. Offset pile (radius 0.7): global 1.93, 1.96; local r=0.60 2.11, 2.10.

| change from the offset-pile baseline | global | local r=0.60 | seeds |
| --- | --- | --- | --- |
| larger, sparser pile (radius 0.9) | 1.66, 1.74 | 2.04, 1.94 | 2 |
| repair batches (50% damaged starts) | 1.82, 1.81 | 2.01, 2.00 | 2 |
| repair batches 80% | | 1.97 | 1 |
| 4x wider network | 1.66 | 2.04 | 1 |
| 60,000 iterations | 1.73 | 2.01 | 1 |

- A larger pile helps by about 0.25 for the global model and 0.1 for the local one (2 seeds each): the pile has to travel less far. Repair batches help by about 0.12 for global and 0.1 for local (2 seeds each). The dose saturates: 80% is no better than 50% (1 seed).
- Capacity and length help the global model (0.2 to 0.27, 1 seed each) but the local model hardly (0.03 to 0.07), so the local model's pile gap is not a size problem; consistent with the sensing-limit reading. A second seed of the wide global run is queued.
- None of these reach the centred-pile or two-phase numbers (global 1.40, 1.52 and 1.51, 1.53; local 1.53, 1.47 and 1.79, 1.80), which stay the best handling of the pile start.

Still running: the three reference-swarm models (stage 1 hand-off), the wide-and-long global pile run, and the wide global second seed.
