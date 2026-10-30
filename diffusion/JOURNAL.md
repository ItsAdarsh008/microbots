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

## 2026-10-29 tick 23: the Stage 1 reference swarm, tested end to end (2 seeds)

(The first wave of shared-global-state runs, `s5a_*`, is still training; nothing from it is reported here. Plan and predictions: `GLOBAL_STATE_PLAN.md`.)

**The reference swarm** = local radius 0.60, 4 message rounds, only an 8x8 occupancy grid (64 numbers) as the target description, noisy-clock training (0.15), repair batches (50%), trained from a pile centred on the arena, thin-shape data. Trained with two seeds (local) plus one global version as an upper bound.

**Formation from the centred pile (familiar / held-out family / emoji; no-learning baseline with the full map from scattered noise 1.71 / 1.96 / 1.60):**

| model | familiar | held-out family | emoji | collisions |
| --- | --- | --- | --- | --- |
| reference swarm, local r=0.60, seed 0 | 1.37 | 1.67 | 1.46 | 0.00 |
| reference swarm, local r=0.60, seed 1 | 1.39 | 1.61 | 1.47 | 0.00 |
| reference swarm, global (upper bound), 1 seed | 1.36 | 1.57 | 1.36 | 0.00 |
| same local model without the extras (full map, no noisy clock, no repair), 2 seeds | 1.33, 1.34 | 1.53, 1.47 | 1.44, 1.45 | 0.00 |

The three extras (64-number broadcast, noisy-clock training, repair batches) together cost about 0.1 on held-out shapes.

**End-to-end checks on the reference swarm (seed 0 / seed 1; 96 val shapes; 16 to 32 steps).**

| check | seed 0 | seed 1 |
| --- | --- | --- |
| travel (offset pile moves rigidly to the shape's centre), then form: familiar / held-out error | 1.42 / 1.64 | 1.49 / 1.61 |
| forming from the centred pile as trained | 1.45 / 1.58 | 1.46 / 1.55 |
| repair: cover after a cut chunk, before / after | 0.80 / 0.97 | 0.80 / 0.97 |
| repair: cover after 20% random loss, before / after | 0.93 / 0.97 | 0.93 / 0.97 |
| clock jitter 0.25: error / collisions | 1.24 / 0.00 | 1.23 / 0.00 |
| half the bots update per round: error / collisions | 1.27 / 0.05 | 1.28 / 0.05 |
| motion noise (0.3 spacings per step): error / collisions | 1.23 / 0.10 | 1.25 / 0.11 |
| 128 and 512 bots instead of 256: collisions | 0.00, 0.00 | 0.00, 0.00 |
| speed cap 1 spacing per step: error vs uncapped (32 steps) | 1.41 vs 1.41 | 1.42 vs 1.42 |
| hard-core spacing 0.7: collisions | 0.000 | 0.000 |

**What this shows (2 seeds):** one local network, given a 64-number target description and only neighbours within 6 spacings, forms unseen thin shapes after a trivial rigid travel, refills a cut-out chunk or lost bots to cover 0.97, tolerates clock errors, a changed bot count and a speed cap, with no collisions in ordinary formation. The moves are short (mean path 2.2 spacings from the centred pile), so the speed cap is free.
**Weak point:** random motion noise (0.3 spacings per step) gives 10 to 11% collisions; the hard-core rule has not yet been tested together with motion noise. Half-asynchronous updates also give 5%.
**Not tested:** the actual bot dynamics (finite size and links, arm reach, strength), real aggregation latency, anything in 3D. The thin-shape data is my own generator.

Hypothesis: H41, hard-core spacing removes the motion-noise collisions as it removed the others (to test in Stage 2 with the real body model).

## 2026-10-29 tick 24: interactive demo; first hand-drawn shapes expose a gap (1 seed per model, 1 run per cell)

**Demo.** `diffusion/demo/` serves 12 trained models behind a small web page (pick a model, pick or draw a shape, choose start, steps, bot count, clock error, motion noise, hard-core rule; cut a chunk and repair). It runs on one of the GPU boxes (about 0.1 to 0.2 s per run on the GPU; 6 to 18 s on that box's CPU because training saturates it), bound to localhost and reached through an SSH tunnel. No GPU rental is needed: the models are 2.5M to 3.6M parameters.

**What hand-drawn shapes show** (`demo/test_drawn.py`; fraction of bots inside the shape / fraction of the shape covered; one seed per model; shapes drawn on the 64x64 grid, then centred and scaled like training shapes; `s2_*` models start from scattered noise, `s4_*` from a centred pile):

| shape | reference swarm (local r=0.6, pile) | reference swarm, global | global, scattered start | local r=0.6, scattered start |
| --- | --- | --- | --- | --- |
| disc | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| fat T | 0.55 / 0.70 | 0.46 / 0.56 | 1.00 / 1.00 | 0.96 / 0.98 |
| thin T | 0.33 / 0.50 | 0.70 / 0.98 | 0.99 / 0.99 | 0.33 / 0.47 |
| L | 0.78 / 0.95 | 0.89 / 0.98 | 0.99 / 0.99 | 0.98 / 0.99 |
| ring | 0.92 / 1.00 | 0.95 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 |
| preset star via the drawing path | 0.86 / 0.95 | 0.95 / 0.99 | 1.00 / 1.00 | 0.78 / 0.88 |
| preset ring via the drawing path | 0.70 / 0.94 | 0.75 / 0.97 | 0.98 / 0.98 | 0.62 / 0.76 |

The drawing path is not the cause (discs, rings and presets work through it, and the global scattered-start model forms every drawn shape). The models differ: the pile-start reference swarms fail on wide T-shapes (inside 0.46 to 0.55) and the local r=0.6 model with scattered start fails on a thin T (inside 0.33). None of my held-out families (stars, rings, spiders, emoji) exposed this. H42: the pile start (radius 0.7) cannot supply bots to the extremities of wide bars, and thin parts need either a larger sensing radius or all-to-all attention.
**Next:** a hand-drawn/letter test suite evaluated on 2 seeds of every model (letters A to Z drawn from a font, plus T, L, plus, arrow), so this becomes a measured result rather than a handful of anecdotes.

## 2026-10-29 tick 25: first wave of shared global state, scored against the pre-registered rules (partial; NOT a finding yet)

Base: thin shapes, scattered-noise start, full map, local r=0.25, 4 rounds, clock given. Error in spacings, held-out family (familiar shapes in brackets); lower is better. Baseline r=0.25 (no shared state): **1.97, 2.03** (seed 0, 1; familiar 1.51, 1.56). Seed spread of the baseline: 0.06 (a third seed is running). Reference: local r=0.40 1.69, 1.68; global attention 1.47, 1.41.

| variant (floats shared per round) | seed 0 | seed 1 | wrong-state swap at test (seed 0): familiar-shape error | verdict so far |
| --- | --- | --- | --- | --- |
| mean64 (64) | 1.75 (1.38) | 1.85 (1.47) | 4.12 (from 1.38) | gain about 0.20, **but see the control** |
| mean64_shuf, **control**: broadcast from another swarm | 1.83 (1.44) | queued | n/a | **gain 0.17 with no information about this swarm** |
| dens8 (64, raw density grid) | 1.90 (1.43) | 1.94 (1.48) | 2.24 (from 1.43) | gain 0.08 (below the 0.15 threshold); used, but barely helps |
| grid8 (64, learned grid) | 1.62 (1.49) | running | 3.09 (from 1.49) | gain 0.38, one seed, **no matched control yet** |
| isab4x16 (64, four tokens) | 1.59 (1.55) | queued | 4.88 (from 1.55) | gain 0.41 on held-out, but familiar shapes and emoji not better (1.55 vs 1.51; 1.71 vs 1.65), one seed, **no matched control yet** |

Pre-registered rules (`GLOBAL_STATE_PLAN.md`): "helps" needs a gain of at least 0.15 on both seeds, a rise of at least 0.10 when the shared state is swapped, and a control without a comparable gain.
- **mean64: not established.** The swap rises a lot (the model uses the broadcast), but the control that receives another swarm's broadcast also gains 0.17, nearly the same as mean64's 0.20. So the gain is capacity or regularisation of an extra path, or seed luck, not coordination.
- **dens8: fails** the size threshold (0.08) on both seeds, although it is used.
- **grid8, isab4x16: promising and untested**: bigger gains (0.38 and 0.41) but one seed each and no matched shuffled control. Queued now: shuffled controls for both (`s5a_r25_grid8_shuf`, `s5a_r25_isab4x16_shuf`), second seeds, a second shuffled mean64, and a third baseline seed.

**Predictions scored.** 1 (spatial beats pooled vector): wrong for the raw density grid (0.08), unresolved for the learned grid. 2 (mean16 about mean64): not scored (mean16 running). 4 (shuffled control shows no gain): **wrong** (0.17), which is exactly the case the control exists for. 5: not applicable yet.

Hypotheses: H43, the benefit of an extra global path may mostly come from an added normalising or averaging channel (a mean over all nodes), not from swarm-specific information; the shuffled controls for grid8 and isab4x16 decide this. H44, isab's lower familiar-shape performance (1.55) with a better held-out score (1.59) suggests a different trade-off than the grid; per-family breakdown next.

## 2026-10-29 tick 26: CORRECTION. I did not look at the outputs; when I did, the "beats the no-learning baseline" claim fell apart

**What happened.** The user ran the demo and said the results were not good. I then rendered the models' outputs (`demo/render_grid.py`, `demo/render_vs_classical.py`; pictures in the scratchpad, not committed) and looked. Until then I had judged the models almost entirely by numbers (`w2`, inside, cover), looking at pictures only for a few runs.

**What the pictures show.** On the real emoji shapes the trained models make fuzzy blobs of the right size and place. Thin features are lost (anchor, the key's ring, snowflake arms, the fish tail), parts are missing (butterfly wings, wrench ends), and bots spill over the boundary. The reference swarm (pile start, 64-number map) is no better and the snowflake becomes a disc. The no-learning rule (project onto the shape, spread by local repulsion, 200 steps) draws all eight shapes crisply, evenly spaced and in the right outline.

**Why the earlier claim was wrong.** `w2` measures distance to one random evenly-spaced sample of the shape (floor about 0.8). The rule fills shapes evenly and densely along the outline, which a different sample does not match well, so `w2` ranked it behind the learned models; that ranking does not match how the shapes look. Every statement in this journal and README of the form "beats the no-learning baseline" (from tick 1 on) rests on `w2` and is **withdrawn as a statement about shape quality**. What stays: comparisons *between learned variants* by `w2` (radius, clock, broadcast size, pile start, repair, etc.) are internally consistent but have not been checked visually.

**A metric that matches the eye** (`demo/iou_eval.py`): footprint IoU of the bots (pixels within 0.7 spacings of a bot) against the target mask. The ceiling (the data's own independent evenly spaced sample of the same shape): **0.79 emoji / 0.78 held-out family / 0.83 familiar**. Thin shapes, scattered start, 256 bots, 1 model per row, one seed:

| method | emoji | held-out family | familiar |
| --- | --- | --- | --- |
| ceiling (perfect even sample) | 0.79 | 0.78 | 0.83 |
| no-learning rule, 200 steps | 0.63 | 0.61 | 0.67 |
| global attention, 16 steps | 0.64 | 0.59 | 0.70 |
| global + 60 steps of the rule as polish | **0.71** | **0.69** | **0.76** |
| local r=0.6 | 0.58 | 0.61 | 0.67 |
| local r=0.6 + polish | **0.70** | **0.70** | **0.75** |
| reference swarm (pile start, 64-number map) | 0.62 | 0.60 | 0.67 |

- By overlap the learned models are **level with the rule, not better** (0.64 vs 0.63; 0.59 vs 0.61 held-out), and well below the ceiling (recall of the target shape is about 0.8 against 0.96 for the ceiling: parts of the shape are not covered).
- **Learned transport followed by a short run of the rule is best** (+0.07 to +0.10 IoU over either alone, about 90% of the ceiling). Both pieces are local and swarm-realisable (the rule uses the map at the bot's own position and repulsion from near neighbours). One seed per row; not yet replicated.
- The reference swarm with a 64-number map matches what the rule gets with the full-resolution map. That is a fair, narrower claim; it is not "better than the baseline".
- The IoU metric is not perfect either (it punishes boundary bots), which is why the ceiling matters.

**Changes made.** `iou`, `prec`, `rec` are now in every evaluation (`evalm.metrics`, shown in `collect.py`); older results lack them. New training option: geometric auxiliary losses on where each bot would end up if it kept going (penalise ending outside the shape, penalise crowding; `aux_in`, `aux_rep`), aimed directly at what the pictures show. Five runs queued (`s6_*`; global and local r=0.6; weights 1 and 5), to be judged by IoU **and by looking at the pictures**.

Hypotheses: H45, the learned flow's errors are mostly near the end (small displacement errors count little in the velocity loss but matter for the outline), so endpoint-aware losses or polishing will help; H46, uneven spacing and bots outside the shape, not wrong placement of the bulk, are what makes the models look bad.

**Process rule from now on:** every claim about shape quality is checked against rendered pictures of the same runs, side by side with the no-learning rule and the ceiling, before it goes in the journal.

## 2026-10-29 tick 27: the shared-global-state wave re-scored by IoU and by eye: no visible gain

(Asked by the user: do I want more GPUs? No: runs take about 25 minutes, 5 GPUs hold 10 at a time, and the constraint has been judging results correctly, not compute.)

**Re-scoring `s5a_*` with footprint IoU (ceiling 0.79 emoji / 0.78 held-out / 0.83 familiar; thin shapes; scattered start; local r=0.25, 16 steps; one model per row, 4 starts for emoji):**

| model | emoji | held-out family | familiar | `w2` held-out (earlier) |
| --- | --- | --- | --- | --- |
| no shared state (`s2_cond_local_r25_L4`) | 0.61 | 0.58 | 0.67 | 1.97 |
| no shared state, seed 1 | 0.60 | 0.57 | 0.67 | 2.03 |
| mean64 (pooled vector) | 0.57 | 0.54 | 0.63 | 1.75 |
| mean64_shuf (control) | 0.59 | 0.56 | 0.64 | 1.83 |
| grid8 (learned grid) | 0.62 | 0.60 | 0.67 | 1.62 |
| grid8, seed 1 | 0.61 | 0.60 | 0.66 | 1.64 |
| no-learning rule, 200 steps | 0.63 | 0.61 | 0.67 | |

- **Pooled vector: worse than no shared state by IoU (0.54 vs 0.57 to 0.58 held-out), although `w2` said 0.2 better.** Learned grid: +0.02 to +0.03 held-out on both seeds, i.e. small and about the size of seed spread. Rendered pictures (rows of the same eight emoji; scratchpad, not committed) show no visible improvement from any shared-state variant: the ring fills in, the anchor and key stay clouds; at best the grid keeps a little more of the ring's hole.
- **After 60 steps of the rule as polish, all six models land within 0.70 to 0.72 (emoji) and 0.69 to 0.70 (held-out)**: the shared state adds nothing once the polish is applied.
- **The `w2` tables of this wave (tick 25) are withdrawn as evidence for shared state.** The pre-registered decision rules were written in `w2`; they are superseded and, applied to IoU, the answer is: no variant shows a gain beyond seed spread. Waves B to F of `GLOBAL_STATE_PLAN.md` (clock replacement, pile offset, repair, matched r=0.4, noisy aggregates) are **paused**, not run: testing more elaborate uses of a shared state on a base model whose output looks poor is not informative. The runs still in flight (`isab4x16` seed 1 and shuffled controls, `grid8_shuf`, `mean64_shuf` seed 1, a third baseline seed) will be scored by IoU and pictures for completeness.

**Is the rule just under-run? (`demo/rule_steps.py`, IoU of the no-learning rule from scattered noise).** 200 steps 0.63 / 0.60; 600: 0.65 / 0.63; 1500: 0.66 / 0.65; 3000: 0.67 / 0.65 (emoji / held-out). It plateaus. **The learned model plus 60 polish steps (0.71 to 0.72 / 0.70) is better than the rule at any step count and gets there with about 50 times fewer steps** (16 learned plus 60 rule steps against 1500 or more). That is a fair, replicated-across-models finding (six models, polish gives 0.69 to 0.72 for all). Caveat: the polish uses the full-resolution distance map at the bot's own position, so it needs more than the 64-number broadcast that the reference swarm uses.

**What the generative approach contributes, as measured so far:** fast coarse transport to the right region (16 steps instead of 1500), conditioning on a coarse 64-number map (the rule cannot use one), repair when trained for it, an unconditional mode, and a better start for the rule's refinement. It does **not** yet draw crisp shapes on its own.

**Next (in order):** the five geometric-loss runs (`s6_*`) aimed at crisp edges and even spacing, judged by IoU and by looking at pictures; a letters and hand-drawn test set for all models; then, if a model reaches crisp outputs on its own, revisit shared state on that base.

## 2026-10-29 tick 28: first geometric-loss model; shared-state controls finished

**Geometric auxiliary losses (`aux_in` penalises where a bot would end up if it kept going outside the shape, `aux_rep` penalises crowding; weight 5 and 5), local r=0.6, thin shapes, scattered start, 1 seed.** Same protocol as the earlier IoU table (16 steps, footprint IoU, ceiling 0.79 / 0.78 / 0.83); emoji / held-out / familiar:

| model | emoji | held-out | familiar | after 60 polish steps |
| --- | --- | --- | --- | --- |
| no-learning rule | 0.63 | 0.61 | 0.67 | |
| local r=0.6 (plain) | 0.58 | 0.61 | 0.67 | 0.70 / 0.70 / 0.75 |
| local r=0.6 + geometric losses | **0.63** | **0.65** | **0.70** | 0.71 / 0.70 / 0.76 |

Looked at it (`aux_grid.png` in the scratchpad, next to the plain model and the rule): the ring now keeps its hole and reads as a band, the star is star-shaped, bots stay closer to the target; the outline is still fuzzy and thin parts (anchor loops, key ring, snowflake arms) are still lost. So: **a real but modest improvement, +0.03 to +0.05 IoU, up to the level of the no-learning rule and not past it; not crisp on its own.** The polished result is unchanged (0.70 to 0.71). Replicate and larger weights queued (`s6_r60_aux_in5_rep5_seed1`, `s6_r60_aux_in20_rep20`, `s6_global_aux_in20_rep20`, plus the reference-swarm setting `s7_ref_local_r60_aux5`). Other `s6` runs (global, weight 1 and 5) are still running.

**Shared-state controls finished (held-out `w2`, one entry per seed; no-learning baseline here 1.97, 2.03, 2.01 on 3 seeds):** grid8 1.62, 1.64 vs its shuffled control 2.09; isab4x16 1.59, 1.59 vs control 1.87; mean64 1.75, 1.85 vs controls 1.83, 2.02; mean16 1.85. Precise reading: in `w2` the learned grid and four-token variants **do** beat their matched controls and the baseline on 2 seeds (so these `w2` gains are real and information-driven, not seed luck); the pooled vector does not beat its control. But `w2` is not shape quality: by IoU and by eye (tick 27) these variants show no visible improvement. So the finding is "shared state changes `w2`, not what the shapes look like", not "shared state does nothing in any sense".

Hypotheses: H47, a stronger geometric weight (20) will tighten further but may trade off with fine structure; the wider test decides. H48, since polish equalises all models at 0.70 to 0.72, the useful split of labour is learned transport plus a local refinement rule, and the learned part should be trained for what the refinement cannot do (reaching far parts, topology).

## 2026-10-30 tick 29: geometric losses replicated, a letters test set, two evaluation mistakes of mine

**Geometric auxiliary losses, same protocol (16 steps, scattered start, footprint IoU; thin-shape models; ceiling 0.79 / 0.78 / 0.83; emoji / held-out / familiar).**

| model | weight | emoji | held-out | familiar | seeds |
| --- | --- | --- | --- | --- | --- |
| global, plain | 0 | 0.64 | 0.59 | 0.70 | 1 |
| global + geometric losses | 1 + 1 | 0.64 | 0.59 | 0.69 | 1 |
| global + geometric losses | 5 + 5 | 0.66 | 0.64 | 0.70 | 1 |
| local r=0.6, plain | 0 | 0.58 | 0.61 | 0.67 | 1 |
| local r=0.6 + geometric losses | 1 + 1 | 0.59 | 0.62 | 0.68 | 1 |
| local r=0.6 + geometric losses | 5 + 5 | 0.63, 0.64 | 0.65, 0.65 | 0.70, 0.70 | **2** |
| no-learning rule | | 0.63 | 0.61 | 0.67 | |

- **Weight 5 helps and replicates for the local model (2 seeds: +0.05 to +0.06 emoji, +0.04 held-out, +0.03 familiar); weight 1 does nothing.** For the global model weight 5 gives +0.05 held-out on one seed. After 60 polish steps all of them tie (0.70 to 0.76), so the gain is in what the learned part does on its own.
- Pictures (8 emoji, plain vs weight 1 vs weight 5, both model types): the weight-5 models are tighter and keep the ring's hole; **the key stays a diagonal blob in every row, the anchor loses its loops, the wrench jaws only half appear**. Better, not crisp. Still running: weight 20 (local, global), 60k iterations, late-time emphasis (`tpow`), ablations of the two penalty terms (`aux_in` only, `aux_rep` only; so I can say which one does the work), second seed for global, and the reference-swarm recipe with the new losses.

**Letters test set (`demo/make_letters.py`: 26 uppercase letters in a thick font (DejaVu Bold) and a thin one (Lato); nobody trained on letters; ceiling IoU 0.76).** Upright letters, each model started the way it was trained, 16 steps, 26 per font, 1 seed per model:

| model | thick letters IoU | thin letters IoU |
| --- | --- | --- |
| no-learning rule | 0.59 | 0.56 |
| global | 0.60 | 0.50 |
| global + geometric losses | 0.62 | 0.52 |
| local r=0.6 | 0.62 | 0.46 |
| local r=0.6 + geometric losses | **0.67** | 0.55 |
| reference swarm (pile start, 64-number map) | 0.61 | 0.44 |
| any of these + 60 polish steps | 0.68 to 0.70 | 0.61 to 0.63 |

By eye the learned models make **readable letters** (A, K, S, E, H, Z, B are clearly legible from global and local r=0.6; the reference swarm's bold A collapses to a ring); the rule's outlines are crisper. Thick letters: the best learned model is above the rule (0.67 vs 0.59). Thin letters: most learned models are **below the rule** (0.44 to 0.55 vs 0.56): thin strokes remain the weak spot, as the earlier T result suggested. Polish again equalises everything.

**Two mistakes of mine, both found by looking at the pictures.**
1. The first letters run used upside-down targets: array rows run bottom-to-top in this code and I did not flip the font images, so "A" was a "V". The emoji images are stored upside-down for the same reason (anchor with its ring at the bottom); that does not change any score (models were trained on randomly oriented shapes) but the demo showed upside-down emoji. Fixed (letters flipped when made; emoji flipped for display in the demo). The first, wrong-orientation letter numbers (e.g. global 0.67 / 0.57) are discarded; the upright numbers are different (0.60 / 0.50): **these models are not orientation-invariant** (a flipped shape is a different test).
2. The first letters run also started every model from scattered noise, including the reference swarm, which was trained from a centred pile. Its 0.31 was a broken test, not a model result. Fixed: each model now starts as it was trained.

Demo: now has the letters (52 presets, thick and thin), upright emoji, 14 described models including the two geometric-loss models; restarted. Unlisted models stay hidden.

## 2026-10-30 tick 30: the geometric-loss weight is the lever; first learned models that clearly beat the rule

**Weight sweep, thin shapes, scattered start, same protocol (16 steps, footprint IoU; ceiling 0.79 / 0.78 / 0.83); emoji / held-out / familiar, 1 seed each unless noted:**

| model | weight | emoji | held-out | familiar |
| --- | --- | --- | --- | --- |
| no-learning rule | | 0.63 | 0.61 | 0.67 |
| local r=0.6 | 0 | 0.58 | 0.61 | 0.67 |
| local r=0.6 | 5 (2 seeds: 0.63, 0.64 / 0.65, 0.65 / 0.70, 0.70) | 0.63 | 0.65 | 0.70 |
| local r=0.6 | **20** | **0.70** | **0.71** | **0.75** |
| global | 0 | 0.64 | 0.59 | 0.70 |
| global | 5 | 0.66 | 0.64 | 0.70 |
| global | **20** | **0.68** | **0.71** | **0.73** |

- Error and IoU improve monotonically with the weight so far (0 -> 5 -> 20). **Looked at the pictures** (rows: rule, local plain / 5 / 20, global plain / 5 / 20, eight emoji, upright): the weight-20 models draw a recognisable anchor (ring, shank, arms), a ring with its stem, the wrench jaws and the six snowflake arms; the plain ones give blobs. Still fuzzier than the rule's outlines and the key's ring end is only partly formed; but this is the first time learned models are clearly better than the rule by overlap (+0.07 to +0.10 on emoji and held-out shapes) **and by eye**. Weight 20 is a single seed per model; the replicate (`s6_r60_aux_in20_rep20_seed1`) and higher weights (50) are running.
- In the evaluator's own protocol (64 steps) weight 20 gives local IoU 0.74 / 0.70 / 0.69 and global 0.72 / 0.68 / 0.69 (val / held-out / emoji), against 0.69 / 0.63 / 0.64 at weight 5. Late-time emphasis alone (`tpow` 2) did not help: IoU 0.64 / 0.58 / 0.58 (plain 64-step reference unavailable; the weight-5 model is 0.69 / 0.63 / 0.64) and adding it to weight 5 changed nothing (0.69 / 0.64 / 0.64). H45 (the error is mostly at the end of the flow) is not supported as an explanation; the geometric penalty works by directly punishing bots outside and crowding.
- Reference-swarm recipe (pile start, 64-number map, noisy clock, repair) with weight 5: IoU 0.71 / 0.61 / 0.66 (val / held-out / emoji, 64 steps), collisions 0.00 (1 seed; weight 20 version queued). Not directly comparable with the numbers above (different start and map).

Still to do: replicate weight 20 (1 more seed), higher weights, which of the two penalty terms matters (ablations `in only`, `rep only` at weight 5 and 20 are running), the reference-swarm recipe at weight 20, letters and hand-drawn shapes for the weight-20 models, and a check of whether polish still equalises everything at the new level.

## 2026-10-30 tick 31: late-time emphasis settled; replicates running

`s8_global_aux5_tpow2` (global, geometric losses 5 + 5, late-time emphasis 2): IoU 0.69 / 0.65 / 0.65 (val / held-out / emoji, 64 steps), the same as without it (0.69 / 0.64 / 0.65). With the local model (tick 30) the result was the same, so **late-time emphasis does not help, on two models** (it hurts when used alone: local 0.64 / 0.58 / 0.58). Queued to replicate the global weight-20 result: `s6_global_aux_in20_rep20_seed1`. Running: weight 50 (local, global), weight-20 second seed (local), each penalty term alone at 5 and 20, reference swarm at weight 20, 60k iterations at weight 5, global weight 5 second seed.

## 2026-10-30 tick 32: weight 20 replicated; "stay inside" is the useful penalty; a regression on hand-drawn T shapes

**Replicates and ablations (thin shapes, scattered start; footprint IoU in the evaluator's protocol: 64 steps; val / held-out / emoji; ceiling about 0.82 / 0.78 / 0.79):**

| model | val | held-out | emoji | collisions | seeds |
| --- | --- | --- | --- | --- | --- |
| local r=0.6, no geometric loss | n/a | n/a | n/a | | (16-step table: 0.67 / 0.61 / 0.58) |
| local, inside + crowding, weight 5 | 0.69, 0.68 | 0.63, 0.64 | 0.64, 0.63 | 0.05 | 2 |
| local, inside + crowding, **weight 20** | **0.74, 0.73** | **0.70, 0.69** | **0.69, 0.69** | 0.02 to 0.03 | **2** |
| local, **inside only**, weight 5 | 0.69 | 0.63 | 0.63 | 0.07 | 1 |
| local, **inside only**, weight 20 | 0.73 | 0.69 | 0.68 | 0.08 | 1 |
| local, **crowding only**, weight 20 | 0.66 | 0.59 | 0.60 | 0.03 | 1 |
| local, weight 5, 60k iterations | 0.70 | 0.65 | 0.64 | 0.02 | 1 |
| global, weight 5 | 0.69, 0.67 | 0.64, 0.65 | 0.65, 0.65 | 0.03 | 2 |
| global, weight 20 | 0.72 | 0.68 | 0.69 | 0.04 | 1 (replicate running) |

- **The weight-20 improvement replicates for the local model on 2 seeds.** Global weight 5 replicates (2 seeds).
- **Attribution:** the "stay inside" penalty alone carries essentially all of the gain (weight 5: same overlap as both together; weight 20: 0.73 / 0.69 / 0.68 vs 0.74 / 0.70 / 0.69). The crowding penalty alone does nothing (0.66 / 0.59 / 0.60, plain level). Crowding's job is collisions: they fall from 0.08 (inside only) to 0.02 (both). One seed for each ablation; consistent across the two weights. More training (60k) adds only 0.01 to 0.02.
- **Letters (upright; 16 steps; 26 per font; 1 seed):** local r=0.6 weight 20: thick 0.71, thin 0.64; plain local 0.62 and 0.46; no-learning rule 0.59 and 0.56; global weight 20 0.69 and 0.60 (plain 0.60 and 0.50). Rendered: the weight-20 models draw clean, readable letters (E, H, B included, which the plain local model smears); the rule's outlines are still crisper. The weight-20 local model needs no polish (0.71 / 0.64 before and after).

**A regression: hand-drawn T shapes (`demo/test_drawn2.py`; bots inside the shape / shape covered; 16 steps; 1 seed; one drawing each).**

| shape | local plain | local weight 20 | global plain | global weight 20 |
| --- | --- | --- | --- | --- |
| fat T | 0.96 / 0.98 | **0.80 / 0.84** | 1.00 / 1.00 | **0.64 / 0.56** |
| thin T | 0.33 / 0.47 | **0.71 / 0.78** | 0.99 / 0.99 | **0.48 / 0.38** |
| disc, L, plus, ring | 0.98 to 1.00 | 1.00 | 0.99 to 1.00 | 1.00 |

The geometric losses help on emoji, letters and held-out families (mean over many shapes) but the global model gets **worse** on wide T shapes (cover 0.56), and the local one trades (thin T much better, fat T worse). Reading (hypothesis H49): "stay inside the shape" can be satisfied by sitting in the middle of a large part of the shape, leaving the far ends uncovered; a coverage penalty should fix exactly this. It is also a reminder that averages over a test set can hide a regression on particular shapes; the demo exposes it, which is how it was found.

**New penalty `aux_cov`** (sampled target pixels should have a bot nearby) added; runs queued on the best recipe: local weight 20 + coverage 20 and 5, global weight 20 + coverage 20. Still running: weight 50, reference swarm at weight 20, global weight-20 replicate, crowding-only at weight 5.

Queued (tick 32, to keep every GPU busy and answer a hardware question): `s6_r25_aux_in20_rep20` and `s6_r40_aux_in20_rep20`, two seeds each: do the geometric losses remove the need for a large sensing radius? (Without them: r=0.25 is far behind r=0.6.) Judged by IoU and by pictures.

## 2026-10-30 tick 33: weights 50, radius with geometric losses, reference swarm at weight 20

**Geometric-loss weight, local r=0.6 / global (thin shapes, scattered start; 16 steps; footprint IoU emoji / held-out / familiar; ceiling 0.79 / 0.78 / 0.83; rule 0.63 / 0.61 / 0.67):**

| weight | local r=0.6 | global |
| --- | --- | --- |
| 0 | 0.58 / 0.61 / 0.67 | 0.64 / 0.59 / 0.70 |
| 5 | 0.63 / 0.65 / 0.70 (2 seeds) | 0.66 / 0.64 / 0.70 |
| 20 | 0.70 / 0.71 / 0.75 (2 seeds, protocol-64: 0.74, 0.73 / 0.70, 0.69 / 0.69, 0.69) | 0.68 / 0.71 / 0.73 (2 seeds in protocol-64: 0.72, 0.71 / 0.68, 0.70 / 0.69, 0.69) |
| **50** | **0.74 / 0.74 / 0.77** | **0.72 / 0.74 / 0.76** |

Weight 50: **about 94% of the ceiling on held-out shapes**, 1 seed per model (second seed for local queued; weight 100 queued). Monotone in the weight so far. Rendered (eight emoji; scratchpad `w50_grid.png`): anchor, ring with stem, wrench, snowflake and star are recognisable; the butterfly's wings and the key's ring end are the weak spots; the hand-written rule is still crisper and has no such gaps.

**Radius with geometric losses (weight 20; 16 steps; emoji / held-out / familiar; 1 seed each so far for r=0.25 and 0.4, seed 0 running):** r=0.25: 0.70 / 0.68 / 0.74; r=0.4: 0.71 / 0.70 / 0.74; r=0.6: 0.70 / 0.71 / 0.75. Without the losses r=0.25 was far behind r=0.6 (the old `w2` gap 1.97 vs 1.54). In IoU and in the pictures (rows r=0.25 and r=0.4 next to r=0.6) **the sensing radius now matters little: r=0.25 is within 0.03 of r=0.6**. If that holds on the second seed, the geometric losses buy back most of the sensing range a swarm needs (2.5 spacings instead of 6): a hardware-relevant result, not yet replicated.

**Reference-swarm recipe (centred pile, 64-number map, noisy clock, repair batches; 64 steps; val / held-out / emoji; collisions 0.00 in all):** weight 5: 0.71 / 0.61 / 0.66; **weight 20: 0.74 / 0.65 / 0.70** (1 seed each); weight 50 queued.

**Hand-drawn T regression:** coverage-penalty runs and the global weight-20 replicate are still training; the T tests have not been repeated on these models. The weight-50 models have not been tested on hand-drawn shapes yet.

Demo: now lists the weight-50 models (labelled BEST) and the small-radius models; restarted.

## 2026-10-30 tick 34: the coverage penalty (1 seed): modest gain, partial fix of the T-shape regression

Local r=0.6, thin shapes, scattered start; footprint IoU at 16 steps (emoji / held-out / familiar): inside + crowding weight 20: 0.70 / 0.71 / 0.75; **+ coverage weight 5: 0.71 / 0.72 / 0.76; + coverage weight 20: 0.72 / 0.72 / 0.76** (+0.01 to +0.02, one seed each; the seed-to-seed spread of the weight-20 model was about 0.01, so this is at the edge of what I can call real). Hand-drawn shapes (inside / covered; one drawing each; 16 steps):

| shape | local plain | weight 20 | + coverage 5 | + coverage 20 |
| --- | --- | --- | --- | --- |
| fat T | 0.96 / 0.98 | 0.80 / 0.84 | 0.82 / 0.88 | **0.87 / 0.94** |
| thin T | 0.33 / 0.47 | 0.71 / 0.78 | **0.87 / 0.91** | **0.86 / 0.93** |
| disc, L, plus, ring | 0.98 to 1.00 | 1.00 | 1.00 | 0.98 to 1.00 |

Looked at (eight emoji; scratchpad `cov_grid.png`): fuller star points, a complete anchor ring, slightly more complete snowflake arms; the butterfly's left wing is still missed in every row. **H49 is partly supported**: coverage repairs most of the thin T and part of the fat T, but the fat T is still below the plain model (0.87 vs 0.96) and the global model with coverage is still training. Treat as provisional until a second seed and the global result exist.

Next, in order: second seed of weight 50 (local) and weight 100; weight 50 + coverage; hand-drawn tests for the weight-50 models; the radius result (r=0.25 / 0.4 second seeds); the reference swarm at weight 50.

Data test queued (tick 34): `shapes3` adds a union-of-rectangles family (T-, L-, E-, plus-like boxy shapes; **not letters**; letters stay a pure test set) to the thin-shape data. `s10_r60_aux20_v3` and `s10_global_aux20_v3` use exactly the weight-20 recipe of `s6_*_aux_in20_rep20`, changing only the data. Because the in-run evaluation sets come from the dataset, **compare these runs only on the fixed test sets** (emoji, the v2 held-out families, letters, hand-drawn shapes through `demo/iou_models.py`, `letters_eval.py`, `test_drawn2.py`), not on the numbers in `collect.py`. Seeds: 1 each; a gain would need a second seed before being claimed.

## 2026-10-30 tick 35: radius with geometric losses replicated; weight 50 replicated

Footprint IoU in the evaluator's protocol (64 steps; val / held-out / emoji; thin shapes; scattered start); one entry per seed. Ceiling about 0.82 / 0.78 / 0.79.

| model | weight | val | held-out | emoji | collisions | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| local r=0.25 | 20 | 0.73, 0.73 | 0.67, 0.67 | 0.69, 0.68 | 0.02 | **2** |
| local r=0.40 | 20 | 0.73, 0.73 | 0.69, 0.69 | 0.70, 0.69 | 0.03 | **2** |
| local r=0.60 | 20 | 0.74, 0.73 | 0.70, 0.69 | 0.69, 0.69 | 0.02 to 0.03 | **2** |
| local r=0.25 | 50 | 0.76 | 0.70 | 0.72 | 0.01 | 1 |
| local r=0.60 | 50 | 0.77, 0.77 | 0.73, 0.73 | 0.73, 0.73 | 0.02 | **2** |
| global | 20 | 0.72, 0.71 | 0.68, 0.70 | 0.69, 0.69 | 0.03 to 0.04 | 2 |
| global | 50 | 0.75 | 0.73 | 0.73 | 0.02 | 1 |
| global, weight 20 + coverage 20 | | 0.74 | 0.72 | 0.71 | 0.03 | 1 |
| reference swarm (pile, 64-number map) | 5 / 20 | 0.71 / 0.74 | 0.61 / 0.65 | 0.66 / 0.70 | 0.00 | 1 each |

- **Replicated (2 seeds): with geometric losses the sensing radius matters little.** At weight 20 the held-out IoU is 0.67 (r=0.25), 0.69 (r=0.4), 0.69 to 0.70 (r=0.6): a gap of 0.02 to 0.03 between 2.5 and 6 spacings of sensing, where before the losses the same models differed by a large margin (old `w2`: 1.97 vs 1.54). At weight 50, r=0.25 is 0.03 below r=0.6 (1 seed for r=0.25; second seed queued). For a physical swarm this matters: smaller sensing radius means fewer neighbours to hear from and a cheaper robot.
- **Weight 50 replicated (2 seeds, local r=0.6)**: 0.77 / 0.73 / 0.73 both times (about 94% of the ceiling on held-out shapes).
- Coverage penalty on top of weight 20 for the global model: held-out 0.72 vs 0.68 to 0.70 (1 seed), the same direction as for the local model.
- The reference-swarm recipe (centred pile, 64-number map, noisy clock, repair batches) reaches 0.74 / 0.65 / 0.70 at weight 20 with no collisions; weight 50 is running.
- Not yet checked this tick by eye: the weight-50 and r=0.25 models on the hand-drawn T/L/plus shapes and letters.

Running: weight 100 (local, global), weight 50 + coverage 50 (local, global), reference swarm at weight 50, the rectangle-shapes data test (local, global), and now `s6_r25_aux_in50_rep50_seed1` and `s9_r25_in50_rep50_cov50`.

## 2026-10-30 tick 36: weight 100; the rectangles data changes nothing; the T regression disappears at weight 100

**Weight sweep, local r=0.6 / global (thin shapes, scattered start; 16 steps, fixed sets; footprint IoU emoji / held-out / familiar; ceiling 0.79 / 0.78 / 0.83; rule 0.63 / 0.61 / 0.67):** 0: 0.58 / 0.61 / 0.67 | 5: 0.63 / 0.65 / 0.70 | 20: 0.70 / 0.71 / 0.75 | 50: 0.74 / 0.74 / 0.77 | **100: 0.75 / 0.75 / 0.79** (local; global at 100 in protocol-64: 0.78 / 0.74 / 0.75 val / held-out / emoji, local 0.78 / 0.74 / 0.75). Weight 50 and 100 differ by about 0.01 to 0.02 (saturating); weight 100 is 1 seed per model (second seed, weight 200 and r=0.25 at weight 100 are running).

**Letters (upright, 26 per font; ceiling 0.76; rule 0.59 thick / 0.56 thin):** local r=0.6 weight 100: **0.76 thick (at the ceiling), 0.70 thin**; weight 20: 0.71 / 0.64. Rendered: the weight-100 letters are tight and clean (the A keeps its hole; E, H, S, Z, B well formed), the best of any learned model; the rule's outline is still the crispest.

**Hand-drawn shapes (inside / covered; one drawing each; 16 steps; 1 seed per model):**

| shape | local, weight 20 | local, **weight 100** | global, weight 20 | global, **weight 100** |
| --- | --- | --- | --- | --- |
| fat T | 0.80 / 0.84 | **1.00 / 0.99** | 0.64 / 0.56 | **1.00 / 1.00** |
| thin T | 0.71 / 0.78 | **1.00 / 0.99** | 0.48 / 0.38 | **1.00 / 0.99** |
| disc, L, plus, ring | 0.99 to 1.00 | 0.99 to 1.00 | 0.99 to 1.00 | 1.00 |

**The T-shape regression of tick 32 was a weight that was too low**, not a flaw of the idea: at weight 100 the T shapes form fully. (It also supports the coverage-penalty explanation only in part: a large "inside" weight alone is enough here. Whether coverage still adds anything at weight 100 is untested.)

**Data attribution (matched pair; only the training data differs):** adding a union-of-rectangles family (boxy T, L, E, plus-like shapes; not letters) to the training shapes, weight 20, 1 seed each, changes nothing on the fixed test sets: emoji 0.70 vs 0.70, held-out 0.71 vs 0.71, familiar 0.74 vs 0.75, thick letters 0.71 vs 0.71, thin 0.63 vs 0.64; global the same within 0.01. **The loss weight, not the shape diversity, is what moved the results.** (Each is one seed; the differences are at the level of seed spread, so "no effect" means "no effect beyond 0.01 to 0.02".)

**Reference swarm (pile start, 64-number map, noisy clock, repair) weight 50:** 0.76 / 0.67 / 0.72 (64-step protocol, val / held-out / emoji), zero collisions; weight 20: 0.74 / 0.65 / 0.70; weight 100 running.

Demo: the weight-100 models are listed as BEST (restarted).

Queued (tick 36), each changing one thing from the weight-100 local recipe (`s6_r60_aux_in100_rep100`; thin shapes): `s11_r60_notime_aux100` and `s11_r40_notime_aux100` (no clock; earlier without these losses the no-clock local models fragmented: 7% / 61% connected), `s11_r60_pfoffset_aux100` (well-spaced pile at a random offset; earlier +0.5 error), `s11_r60_dmg_aux100` (repair batches in training). Hardware-relevant questions: do the losses make the shared clock, the rigid travel phase and the separate repair mode unnecessary?

## 2026-10-30 tick 37: weight 100 replicated; coverage not needed; stress test of the best model

**Weight 100 replicated (local r=0.6, 2 seeds):** 0.78, 0.78 / 0.74, 0.74 / 0.75, 0.75 (val / held-out / emoji; 64-step protocol). Global weight 100 (1 seed) the same; a second seed is queued. Weight 200 is running.

**The coverage penalty is not needed at high weight (1 seed each, 64-step protocol, val / held-out / emoji):** local r=0.6 weight 50 plus coverage 50: 0.77 / 0.72 / 0.73 vs weight 50 alone 0.77, 0.77 / 0.73, 0.73 / 0.73, 0.73; global 0.77 / 0.73 / 0.73 vs 0.75 / 0.73 / 0.73; r=0.25 0.76 / 0.71 / 0.73 vs 0.76 / 0.70 / 0.72. The earlier gain from coverage (weight 20) disappears once the inside weight is high; the T shapes are fixed by the weight, not by coverage. Dropped from the recipe.

**Reference swarm (centred pile, 64-number map, noisy clock, repair; val / held-out / emoji; collisions 0.00):** weight 5: 0.71 / 0.61 / 0.66; 20: 0.74 / 0.65 / 0.70; 50: 0.76 / 0.67 / 0.72; 100 running. Monotone but smaller gains than the plain model; its held-out score is limited by the 64-number map.

**Stress test of the best model (`s6_r60_aux_in100_rep100`, local radius 0.6, 96 val shapes, 16 steps, 1 model):**

| condition | inside | covered | collisions | IoU |
| --- | --- | --- | --- | --- |
| clean | 1.00 | 0.99 | 0.01 | **0.81** (ceiling about 0.82) |
| clock error 0.1 / 0.25 (per bot) | 0.93 / 0.86 | 0.98 / 0.97 | 0.03 / 0.04 | 0.74 / 0.66 |
| half the bots update per round | 0.96 | 0.97 | 0.10 | 0.76 |
| motion noise 0.3 spacings per step | 0.99 | 0.99 | 0.02 | 0.80 |
| 128 / 512 bots | 1.00 / 0.99 | 0.98 / 0.99 | 0.01 / 0.03 | 0.77 / 0.83 |
| hard-core spacing rule | 1.00 | 0.99 | **0.000** | unchanged |
| speed cap 1 spacing per step (from scattered noise) | 0.98 | 0.98 | 0.01 | `w2` 0.87 to 1.26 |

In clean conditions this model reaches `w2` 0.80, the floor, which means its placement is as good as an independent perfect sample. Weak spots: **clock errors** (not trained for: IoU 0.66 at the large setting) and a tight **speed cap from a scattered start** (the move averages 8 spacings and the far bots are capped; the pile start removes most of this). Asynchronous updates give 10% collisions. Motion noise and the bot count are fine. Hypothesis H50: noisy-clock training (as in the reference swarm) fixes the clock sensitivity at no cost in clean quality; `s13_r60_aux100_tn15` queued.

Queued and running: 64-number broadcast at weight 100 (`s12_*`, radii 0.6 and 0.25, and an 16x16 grid), r=0.4 at weight 100, global weight 100 seed 1, no-clock (`s11_*notime`), offset pile and repair-in-training (`s11_*`), reference swarm at weight 100, r=0.25 at weight 100, weight 200.

## 2026-10-30 tick 38: weight sweep saturated; switching from tuning to the open questions

**Weight 200 (local r=0.6, 1 seed; 64-step protocol, val / held-out / emoji): 0.79 / 0.75 / 0.76**, against 0.78 / 0.74 / 0.75 at 100 and 0.77 / 0.73 / 0.73 at 50 (2 seeds each): the weight sweep has saturated (about +0.01 per doubling past 50); no more weight runs. **r=0.25 at weight 100: 0.77 / 0.72 / 0.73** (1 seed; weight 50: 0.76 / 0.70 / 0.72): a 2.5-spacing sensing radius is within about 0.03 of radius 0.6.

**Open questions, each with its run** (none of these results are in yet; no claims until they are):
- H50: noisy-clock training fixes the clock sensitivity of the best model (IoU 0.66 at clock error 0.25) at no cost when clean: `s13_r60_aux100_tn15`.
- H51: with the geometric losses a local swarm no longer needs a shared clock (it did not before: 7% and 61% of shapes were connected): `s11_r60_notime_aux100`, `s11_r40_notime_aux100`.
- H52: the offset-pile start no longer costs about 0.5 with these losses: `s11_r60_pfoffset_aux100`.
- H53: repair batches still work at weight 100 and do not cost formation quality: `s11_r60_dmg_aux100`.
- H54: a 64-number broadcast costs little at weight 100 (at weight 0 it cost about 0.1 in `w2`): `s12_r60_g8_aux100`, `s12_r60_g16_aux100`, `s12_r25_g8_aux100`.
- H55: the whole recipe composed (64-number map, noisy clock, repair batches, centred pile, weight 100) at radii 0.4 and 0.25 keeps most of the quality; `s14_refv2_r40`, `s14_refv2_r25` queued now. This is the candidate "reference swarm v2" for Stage 2, to be tested end to end (travel and form, repair after a cut, clock jitter, speed cap, 128 to 512 bots) and looked at by eye, as the first reference swarm was.

**Remaining known weaknesses (looked at by eye):** the butterfly's far wing and the key's ring end; fine outlines are fuzzier than the hand-written rule's; everything so far is 2D and fixed at 256 bots in training.

## 2026-10-30 tick 39: no clock (1 seed): not enough; repair at weight 100 works; reference swarm saturated

Thin shapes, scattered start, weight 100; 16-step fixed-set protocol (emoji / held-out / familiar footprint IoU; recall in brackets; ceiling 0.79 / 0.78 / 0.83):

| model | emoji | held-out | familiar | one piece (64-step run) |
| --- | --- | --- | --- | --- |
| local r=0.6, clock | 0.75 (0.90) | 0.75 (0.90) | 0.79 (0.91) | 1.00 |
| local r=0.6, **no clock** | 0.74 (0.82) | 0.74 (0.80) | 0.78 (0.82) | 0.95 |
| local r=0.4, **no clock** | 0.71 (0.79) | 0.68 (0.72) | 0.68 (0.71) | 0.84 |
| local r=0.6, repair batches | 0.74 (0.87) | 0.74 (0.87) | 0.78 (0.89) | 0.99 |

- **H51 (no clock needed with the losses): not supported.** Overlap barely drops (0.74 vs 0.75) but **recall falls from 0.90 to 0.82 and the pictures show why**: the no-clock shapes are undersized and skeletal (bots along the middle of strokes, an empty margin around the outline), and at r=0.4 a few bots stay outside the shape (collisions 0.08 to 0.10, shapes connected 0.84). Overlap hid this because footprints of thin skeletons still fall inside the target. Better than without the losses (7% and 61% connected) but a shared clock stays in the recipe. One seed; a second seed of r=0.6 no-clock is queued. Lesson: **recall and a look at the picture matter alongside IoU**.
- **H53 (repair batches at weight 100): supported (1 seed).** Formation costs 0.01 (0.74 / 0.74 / 0.78 vs 0.75 / 0.75 / 0.79); repair test (clock restarted at 0): after a cut chunk cover 0.83 to 0.97, after losing 20% of bots 0.98 to 1.00, collisions 0.02. Pictures: the repair-trained model looks the same as the plain one. Second seed queued.
- **Reference swarm (centred pile, 64-number map, noisy clock, repair batches; 64-step protocol val / held-out / emoji):** weight 5: 0.71 / 0.61 / 0.66; 20: 0.74 / 0.65 / 0.70; 50: 0.76 / 0.67 / 0.72; **100: 0.76 / 0.68 / 0.73**; collisions 0.00. Saturated; held-out 0.68 vs 0.74 for the plain model (full map, scattered start), the gap is the 64-number map and the pile start.

Still running: no-clock second seed and a repair plus no-clock model (`s11_r60_notime_aux100_tn`, composed with repair batches), offset pile (`s11_r60_pfoffset_aux100`), noisy clock (`s13_r60_aux100_tn15`), 64-number broadcast at weight 100 (`s12_*`), the composed reference swarm v2 (`s14_refv2_r40`, `s14_refv2_r25`), global weight-100 second seed.

## 2026-10-30 tick 40: hardware hypotheses H50 to H55 scored; composed reference swarm v2 works end to end (seed 0)

Local models, thin shapes, weight 100, evaluator's protocol (64 steps; val / held-out / emoji footprint IoU; the start named in the row; ceiling 0.82 / 0.78 / 0.79):

| model | start | val | held-out | emoji | collisions | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| plain best, r=0.6 (reference) | scattered | 0.78 | 0.74 | 0.75 | 0.01 | 2 |
| **noisy-clock training** (H50) | scattered | 0.79 | 0.73 | 0.75 | 0.02 | 1 |
| **offset pile**, well spaced (H52) | pile at random offset | **0.80** | 0.74 | 0.76 | 0.01 to 0.03 | 1 |
| repair batches (H53) | scattered | 0.77 | 0.73 | 0.73 | 0.03 | 1 (+1 queued) |
| 16x16 map (256 numbers) | scattered | 0.78 | 0.73 | 0.75 | 0.01 to 0.02 | 1 |
| 8x8 map (64 numbers), r=0.25 | scattered | 0.71 | 0.63 | 0.67 | 0.04 to 0.05 | 1 |
| **composed v2**, r=0.40 (64-number map, noisy clock, repair, centred pile) | centred pile | 0.77 | 0.68 | 0.73 | 0.00 | 1 (+1 queued) |
| **composed v2**, r=0.25 | centred pile | 0.77 | 0.69 | 0.73 | 0.00 | 1 (+1 queued) |
| global, weight 100 | scattered | 0.78 | 0.74 | 0.74 | 0.01 | **2** |

- **H50 supported (1 seed):** the noisy-clock model keeps full quality under clock errors: IoU 0.83 clean, 0.83 at clock error 0.1, **0.79 at 0.25** (the plain model: 0.81 / 0.74 / 0.66), at no cost when clean.
- **H52 supported (1 seed):** with weight 100 a well-spaced pile at a random offset forms shapes as well as scattered noise (0.80 / 0.74 / 0.76 vs 0.78 / 0.74 / 0.75); the roughly 0.5 penalty of weight 0 is gone. (The rigid travel phase may no longer be necessary; not tested directly.)
- **H53 supported (1 seed):** formation costs 0.01.
- **H54:** a 16x16 map (256 numbers) is as good as the full map; an 8x8 map at r=0.25 from scattered noise is clearly weaker (held-out 0.63 vs 0.72 with a full map). The composed v2 r=0.25 with the same 8x8 map but a centred pile start reaches 0.69, so the pile start helps the small map; starts differ between these rows.
- **H55 supported (seed 0):** the composed recipe loses little at small radius (r=0.25: 0.69 held-out vs 0.68 at r=0.4; the first reference swarm at r=0.6 without the new losses: 0.61). Held-out 0.68 to 0.69 vs 0.74 for the plain best model: the gap is the 64-number map.

**End-to-end test of composed v2 r=0.25 (seed 0; 96 val shapes; 16 steps; 2.5 spacings of sensing):**

| check | result |
| --- | --- |
| travel (offset pile moves rigidly to the shape's centre), then form vs forming from the centred pile | `w2` 0.92 vs 0.92 (val), 1.08 vs 1.09 (held-out): the travel phase works and changes nothing |
| clock error 0.1 / 0.25 | IoU 0.83 / 0.82 (clean 0.83) |
| half the bots update per round | IoU 0.81, collisions 0.06 |
| motion noise 0.3 spacings per step | IoU 0.78, collisions 0.11 (weakest point; the hard-core rule is expected to fix) |
| 128 / 512 bots | IoU 0.81 / 0.83, collisions 0.00 |
| speed cap 1 spacing per step; mean path 2.2 spacings | no change |
| hard-core spacing rule | collisions 0.000 |
| repair: cover after a cut chunk, before / after | 0.83 / **0.91** (weaker than the radius-0.6 model's 0.97) |
| repair: cover after losing 20% of the bots | 0.97 / 0.97 |

**Looked at** (16 targets: 8 emoji, 8 letters; scratchpad `refv2.png`; rows: rule, plain best, composed v2 at r=0.4 and r=0.25, first reference swarm): the composed v2 models draw recognisable stars, rings, anchors, wrench, fish and clean letters; far better than the first reference swarm (blobs); slightly softer than the plain best model; **the snowflake collapses to a rounded blob** (an 8x8 map cannot carry six arms) and the key's ring end is only partly formed; the rule's outlines are still crisper.

**Replicates:** global weight 100 second seed 0.78 / 0.74 / 0.74 (first 0.78 / 0.74 / 0.75). Second seeds of composed v2 (both radii), 16x16 map at r=0.25, 8x8 at r=0.4, offset-pile and noisy-clock variants are queued or running; the no-clock and repair second seeds are running. Until they land H50, H52, H53, H55 are single-seed.

## 2026-10-30 tick 41: second seeds; and a correction about the "physical numbers" I kept asking for

**Replicated (2 seeds, thin shapes, weight 100, 64-step protocol, val / held-out / emoji):** repair batches 0.77 / 0.73 / 0.73 and 0.77 / 0.73 / 0.74 (plain: 0.78 / 0.74 / 0.75): **repair training costs about 0.01 of formation quality, on 2 seeds**. No clock, local r=0.6: 0.76 / 0.71 / 0.72 and 0.76 / 0.72 / 0.73, 95% connected both times: the by-eye finding of tick 39 (undersized, skeletal shapes; lower recall) holds on both seeds. Combining no clock with repair batches (`s11_r60_notime_aux100_tn`) is worse than either (0.68 / 0.63 / 0.65, collisions 0.14): not combined. Still running: composed v2 second seeds, 8x8 and 16x16 maps at other radii, offset-pile and noisy-clock second seeds (now queued), r=0.4 at weight 100.

**Correction: I asked for physical bot numbers that the repo already has.** `assets/microbots/microbot_v1/DESIGN.md` gives: ball diameter 30 mm (touching distance 34.5 mm), tip-to-tip 81 mm (arm reach), mass 19.3 g, 6 genderless connectors per bot, 3 N assumed per connector (to be measured), only one bot can cantilever sideways, lattice cell 28 x 28 x 81 mm. In my units one particle spacing is about one bot diameter, so **the 81 mm arm reach is about 2.3 to 2.7 spacings, almost exactly my radius-0.25 sensing model** (and the hard-core radius 0.7 spacings is about one body). What the note does not give, and what I still need: how neighbours are sensed (IR or BLE range and rate), how fast a bot moves per step (arm crawl or roll), and how many bots the target swarm has. With the repo numbers I can already make Stage 2's first model: finite-size bodies (hard-core rule), sensing = arm reach (r=0.25), connectors only where bots touch; the cantilever limit and docking are properties of the 3D lattice work in `linking/`.

## 2026-10-30 tick 42: broadcast size at weight 100; radius 0.4 equals 0.6 (1 seed each)

Thin shapes, scattered start, weight 100, 64-step protocol, val / held-out / emoji footprint IoU:

| model | map told to every bot | val | held-out | emoji | seeds |
| --- | --- | --- | --- | --- | --- |
| local r=0.6 | full map | 0.78 | 0.74 | 0.75 | 2 |
| local r=0.6 | 16x16 grid (256 numbers) | 0.78 | 0.73 | 0.75 | 1 |
| local r=0.6 | **8x8 grid (64 numbers)** | **0.73** | **0.65** | **0.70** | 1 |
| local r=0.4 | full map | 0.78 | 0.74 | 0.75 | 1 |
| local r=0.25 | full map | 0.77 | 0.72 | 0.73 | 1 |
| local r=0.25 | 8x8 grid | 0.71 | 0.63 | 0.67 | 1 |

- **A 64-number broadcast costs 0.05 to 0.09 held-out overlap from scattered noise (r=0.6 and r=0.25); 256 numbers cost nothing.** (At weight 0 the difference was about 0.1 in `w2`.) The composed v2 swarms use an 8x8 map but start from a centred pile, which is why they land at 0.68 to 0.69 held-out rather than 0.63 to 0.65: the starting arrangement carries information the map does not. Second seeds of the 8x8 r=0.6 model and r=0.4 full map are queued.
- **r=0.4 matches r=0.6 at weight 100** (0.78 / 0.74 / 0.75 both) and r=0.25 is 0.02 to 0.03 behind: a sensing radius of 2.5 spacings, about the arm reach of the v1 bot, loses little.
- The 16x16 composed variants (`s14_refv3_*`) test whether 256 numbers bring back the snowflake and the key's ring end that the 8x8 composed swarm loses.

## 2026-10-30 tick 43: composed swarm v3 (16x16 map): the candidate reference swarm for Stage 2

**Replicated (2 seeds):** composed v2 (8x8 map; centred pile start; noisy clock; repair batches; weight 100): r=0.25 0.77 / 0.69 / 0.73 and 0.76 / 0.68 / 0.73; r=0.40 0.77 / 0.68 / 0.73 and 0.77 / 0.68 / 0.74 (val / held-out / emoji; collisions 0.00 all four). Noisy-clock training, clean quality (local r=0.6): 0.79 / 0.73 / 0.75 and 0.78 / 0.73 / 0.74; under clock error 0.25: IoU 0.79 and 0.78 (clean 0.83 and 0.82; the model without it fell to 0.66): **H50 on 2 seeds**.

**New, composed v3 = v2 with a 16x16 map (256 numbers), 1 seed each (second seeds running):** r=0.25 **0.79 / 0.74 / 0.77**, r=0.40 **0.80 / 0.75 / 0.77**, collisions 0.00. Held-out 0.74 to 0.75 equals the plain best model that sees the full map from scattered noise (0.74), and it keeps the pile start, noisy clock and repair training. Pictures (16 targets; scratchpad `refv3.png`): the snowflake's arms, the star's points, the anchor, the A's hole and the letters are all there; the key's ring end is still partial; the rule's outlines are still crisper.

**End-to-end test of composed v3 r=0.25 (seed 0; 96 val shapes; 16 steps; sensing 2.5 spacings):**

| check | result |
| --- | --- |
| formation: val / held-out `w2` (floor about 0.8) | 0.84 / 1.00; clean IoU 0.84 |
| travel (offset pile, rigid move to the shape's centre) then form | `w2` 0.83 vs 0.84 from a centred pile (val); 0.98 vs 1.00 (held-out) |
| clock error 0.25 | IoU 0.83 (clean 0.84) |
| half the bots update per round | IoU 0.81, collisions 0.06 |
| motion noise 0.3 spacings per step | IoU 0.79, collisions 0.11 |
| 128 / 512 bots | IoU 0.82 / 0.84, collisions 0.00 |
| speed cap 1 spacing per step (mean path 2.3) | unchanged |
| hard-core rule | collisions 0.000 |
| repair: cut chunk, cover before / after | 0.84 / **0.94** |
| repair: 20% of bots lost | 0.98 / 0.99 |

Weak points (seed 0): motion noise (collisions 0.11, hard-core rule untested together with it), repair of a large cut at this small radius (0.94, vs 0.97 at r=0.6), the key's ring end. Demo: both v3 models added (labelled SWARM v3; start them from the centred pile).

## 2026-10-30 tick 44: CORRECTION on sensing radius in spacings; scaling to thousands of bots; a harness bug

**CORRECTION.** I have been converting the sensing radius into particle spacings wrongly, and then built a claim on it. Measured from the data: mean spacing (area-based, 256 bots) is 0.061 world units (10th to 90th percentile 0.039 to 0.079, because every shape is scaled to a similar extent, so thin shapes have smaller spacings). So **r=0.25 is 4.1 spacings (about 48 neighbours at full density), r=0.4 is 6.6 (about 122), r=0.6 is 9.8 (about 274)**, not 2.5, 4 and 6. The v1 bot's arm reach is 81 mm over a 34.5 mm touching distance, **about 2.35 spacings**; so the smallest model I have trained (r=0.25) senses about 1.75 times farther than the real arm reaches. My tick 41 statement that the 81 mm reach "is almost exactly my radius-0.25 model" and the tick 35 and 40 phrasing "2.5 spacings" are **retracted**. The comparisons between models (r=0.25 vs 0.4 vs 0.6) are unaffected; only the translation to hardware changes. Models at r=0.14 (about 2.3 spacings) and r=0.10 (about 1.6), plain and in the composed recipe, are now queued (`s16_*`). I also updated the demo labels and the README.

**Scaling to more bots (no retraining; models trained with 256 bots).** The sensing range was kept fixed in spacings (the radius shrinks with the square root of 256/N), which is what a physical swarm would do.
- Composed v3 swarm (r=0.25, 16x16 map; pile start), 5 shapes, one seed: inside 0.96 to 1.00 at all counts; shape covered 0.96 to 0.99 at 256 and 512 bots, 0.91 to 0.97 at 1024, **0.85 to 0.94 at 2048**; collisions 0.00 to 0.04 at up to 1024 and 0.00 to 0.11 at 2048 (0.11 only for the thin K). Pictures (six shapes at 256 / 1024 / 2048; scratchpad `scale.png`): the shapes are well formed and sharper at 1024 and 2048 (snowflake arms, anchor, K, B, S); the centre is a little denser at 2048.
- Plain best model (r=0.6, scattered start): covered 0.96 to 1.00 and collisions 0.07 to 0.12 at 2048.
- **Harness bug found on the way:** my pile generator drew only 1500 candidate points, so asking for 2048 bots picked duplicates (bots on top of each other, 45% collisions); the first 2048-bot result for the pile-start swarm (cover 0.75 to 0.87) was that bug, not the model. Fixed (the candidate count scales with N). This did not affect any earlier result (all used 256 bots, which needs 256 distinct of 1500 candidates).

## 2026-10-30 tick 45: three more replicates (2 seeds each); second seeds of the arm-reach-radius models queued

Thin shapes, weight 100, 64-step protocol, val / held-out / emoji; one entry per seed:
- **Offset pile start (H52): 0.80 / 0.74 / 0.76 and 0.80 / 0.73 / 0.76** (scattered start 0.78 / 0.74 / 0.75): a well-spaced pile at a random offset costs nothing at weight 100 (2 seeds); the roughly 0.5 penalty of weight 0 is gone.
- **64-number map (8x8), local r=0.6, scattered start: 0.73 / 0.65 / 0.70 and 0.73 / 0.65 / 0.69** vs full map 0.78 / 0.74 / 0.75: **a 64-number broadcast costs 0.05 to 0.09 held-out overlap (2 seeds)**; the 256-number map (16x16) cost nothing (1 seed).
- **Sensing radius: r=0.4 full map 0.78 / 0.74 / 0.75 and 0.78 / 0.74 / 0.74** = r=0.6 (2 seeds each). In spacings (corrected, tick 44): r=0.4 is 6.6, r=0.6 is 9.8.
- Composed v3 (16x16 map, centred pile, noisy clock, repair batches) second seeds finished training and are scored next tick; r=0.14 (about 2.3 spacings, the real arm reach) and r=0.10 models, plain and composed, are training, each with a second seed queued now.

## 2026-10-30 tick 46: the swarm at the real arm reach

**Composed v3 (16x16 map, centred pile, noisy clock, repair batches, weight 100) replicated on 2 seeds:** r=0.25 (4.1 spacings) 0.79 / 0.74 / 0.77 and 0.79 / 0.74 / 0.77; r=0.40 (6.6) 0.80 / 0.75 / 0.77 and 0.80 / 0.75 / 0.77 (val / held-out / emoji, collisions 0.00).

**Shorter sensing, 1 seed each (second seeds running):** composed v3 r=0.14 (**2.3 spacings = the v1 bot's arm reach**) **0.79 / 0.74 / 0.76**; r=0.10 (1.6 spacings) 0.78 / 0.72 / 0.75. The same recipe without the pile start, map or repair (plain, scattered start): r=0.14 0.73 / 0.68 / 0.68; r=0.10 0.70 / 0.67 / 0.66, against r=0.25 0.77 / 0.72 / 0.73. So with the composed recipe the real arm-reach range costs nothing on held-out shapes (0.74), and 1.6 spacings costs 0.02; the plain model pays 0.04 to 0.05.

**End-to-end test at r=0.14 (arm reach; seed 0; 96 val shapes; 16 steps):** formation `w2` 1.01 / 1.17 (val / held-out; floor about 0.8), clean IoU 0.83; rigid travel then form identical (1.00 vs 1.01); clock error 0.25 IoU 0.82; half the bots updating 0.81 with collisions 0.05; motion noise 0.3 spacings IoU 0.78, collisions 0.11; 128 / 512 bots IoU 0.81 / 0.84, collisions 0.00; speed cap no change; hard-core spacing rule collisions 0.000. **Repair is the clear weakness at this range:** a cut chunk refills only from 0.83 to **0.85** cover (r=0.25: 0.84 to 0.94; r=0.6: 0.83 to 0.97), while losing 20% of the bots at random is fine (0.98 to 0.98). Pictures (16 targets; scratchpad `arm.png`; rows: rule, v3 at 4.1, 2.3 and 1.6 spacings): shapes and letters recognisable at all three ranges; at 1.6 spacings the butterfly wing and key ring soften.

**H56 (new, with controls):** a bot that cannot see the hole cannot know bots are missing; a coarse swarm-density signal (8x8 histogram of where bots are, read at each bot's position) should let a short-range swarm refill a cut chunk. This revives the shared-state idea of ticks 25 to 27, which did not improve *formation* once judged by overlap and pictures; this time the base model is good and the gap is specific. Runs queued on the composed r=0.14 recipe: `s17_refv3_r14_dens8` (and a second seed), `s17_refv3_r14_dens8_shuf` (control: the density state comes from a different swarm; same parameters, no information about this one), `s17_refv3_r14_grid8` (learned grid). Judged by the repair test and by pictures, not by `w2`; the wrong-shared-state swap is reported for every run. Demo: the arm-reach swarms added.

## 2026-10-30 tick 47: arm-reach range replicated; the hard-core rule removes the collisions under noise

**Replicated (2 seeds; thin shapes, weight 100; 64-step protocol; val / held-out / emoji):**

| model | sensing | val | held-out | emoji |
| --- | --- | --- | --- | --- |
| composed v3, pile start, 16x16 map | 4.1 spacings (r=0.25) | 0.79, 0.79 | 0.74, 0.74 | 0.77, 0.77 |
| composed v3 | **2.3 spacings (r=0.14; the v1 arm reach)** | **0.79, 0.78** | **0.74, 0.74** | **0.76, 0.76** |
| composed v3 | 1.6 spacings (r=0.10) | 0.78, 0.77 | 0.72, 0.72 | 0.75, 0.75 |
| plain, scattered start, full map | 2.3 spacings | 0.73, 0.72 | 0.68, 0.68 | 0.68, 0.68 |
| plain | 1.6 spacings | 0.70, 0.69 | 0.67, 0.66 | 0.66, 0.65 |

**At the v1 bot's arm reach the composed swarm loses nothing on held-out shapes, and the plain model loses about 0.05 (both on 2 seeds).** The recipe (pile start, noisy clock, repair batches, 16x16 map) is what makes short sensing work.

**Hand-drawn shapes through the demo (disc, fat T, thin T, L, plus, ring; one drawing each; inside / covered 0.98 to 1.00 in every cell):** the swarm at 4.1 spacings, the swarm at 2.3 spacings and the plain best model all form every one of them.

**H41 supported (arm-reach swarm, seed 0; 96 val shapes):** the hard-core spacing rule, applied together with the noise, takes collisions from 0.11 to **0.00** with motion noise 0.3 spacings per step (overlap 0.78 to 0.80) and from 0.05 to 0.00 with half the bots updating per round (0.81 to 0.82). All collision weak points seen so far are removed by that one local rule.

(Process note: a robustness-script edit moved a no-grad decorator onto the wrong function and made the first run run out of memory; fixed and rerun. No result was affected.)

Queued for the remaining arm-reach weakness (a cut chunk refills only from 0.83 to 0.85 cover): more repair batches (80% instead of 50%; two seeds) and the 64-number map at the arm reach. Running: swarm-density state tests (`s17_*`: density state with a second seed, the matched control with the density state from another swarm, a learned grid).
