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
