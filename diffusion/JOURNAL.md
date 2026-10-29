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
