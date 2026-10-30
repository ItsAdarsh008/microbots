# Shared global state: plan, predictions and decision rules (written before any result)

**Question.** The local swarm denoiser (attention on a radius graph = a graph attention network) needs a sensing radius of about 0.6 (about 6 particle spacings) to match the all-to-all model on unseen and thin shapes; at r = 0.25 it fails (holes, fragments, thin arms) and without a clock it falls apart. All-to-all attention is not buildable (N^2 messages). **Can a small shared state that every bot contributes to and reads from buy back what radius is missing, and what kind of shared state does it?**

This is a graph-network "global attribute" (Battaglia et al. 2018) / virtual-node idea. Physically the aggregate would come from consensus or gossip, a relay (hop count) or a beacon; the cost of each is in the literature notes (D and E) when they land.

## Base setting (identical in every variant)

Thin-shape data (v2), scattered-noise start, full target map, local radius 0.25, 4 message layers, clock given, hidden width 192, 30,000 iterations, batch 32, same evaluation. Reference points from earlier runs (held-out-family error, spacings; lower is better): local r=0.25 **1.97** (1 seed; second seed queued as `s5a_r25_none_seed1`), r=0.40 1.69, 1.68, r=0.60 1.54, 1.59, global attention 1.47, 1.41. The no-learning baseline is 1.96.

## Variants (one thing changes at a time)

| name | what is shared | floats per round | rounds per step | structure |
| --- | --- | --- | --- | --- |
| none | nothing | 0 | 0 | |
| mean16 | mean of node features, bottlenecked to 16 | 16 | 4 | one vector, no layout |
| mean64 | same, 64 | 64 | 4 | one vector, no layout |
| isab4x16 | 4 learned query tokens read all nodes (Set Transformer inducing points); nodes read the tokens back through a 16-float bottleneck | 64 | 4 | few tokens, content-addressed |
| dens8 | raw swarm-density histogram on an 8x8 grid, read at own position with its gradient | 64 | 1 (input only) | spatial, no parameters |
| grid8 | learned 1-channel field on an 8x8 grid (bots scatter features, 3 conv layers on the grid, bots read back) | 64 | 4 | spatial, learned |
| mean64_shuf | **control**: mean64 but the broadcast comes from another swarm in the batch | 64 | 4 | |

Bandwidth is matched at 64 floats across mean64, isab4x16, dens8 and grid8, so differences between those four are structure, not bandwidth. mean16 vs mean64 isolates bandwidth. mean64_shuf has the same parameters and the same architecture but no information about this swarm, so any gain it shows is capacity or regularisation, not coordination. Parameter counts differ by at most about 1% (2.47M to 2.50M).

**Control applied to every model at evaluation time:** the shared state is swapped for that of another swarm. If the model uses it, error must rise (as it does for the wrong-map control). Reported as `control_wrong_global`.

Two seeds for every variant that survives the first screen; mean64, dens8 and grid8 have seed 1 in the first wave.

## Predictions (priors, to be scored afterwards)

1. dens8 and grid8 (spatial) help more than mean vectors on thin shapes: the failures are about *where* bots are relative to the target, and a pooled vector only carries counts, centroid and progress. Expected gain over r=0.25: 0.2 to 0.4 for dens8/grid8; 0.0 to 0.1 for mean16/mean64.
2. mean64 vs mean16: small difference (a pooled vector saturates quickly).
3. isab4x16 between the two.
4. mean64_shuf: no gain (within seed noise, about 0.1).
5. If nothing helps beyond 0.1, the conclusion is that sensing radius is the binding constraint and a small shared broadcast cannot replace it; that is a result.

## Decision rules

- "The shared state helps" needs **all** of: held-out error below the r=0.25 baseline mean by at least 0.15 on both seeds; wrong-shared-state swap raises error by at least 0.10; mean64_shuf does not show a comparable gain.
- "It is ignored" if the swap changes error by less than 0.03 (posterior-collapse-like behaviour).
- Where two variants differ by less than the seed spread (about 0.1 to 0.15) they are not ranked.

## Adjustments after the first literature report (note D)

- A pooled vector can carry centroid, count and progress but, by a set-function expressivity argument (Wagstaff et al. 2019), not "which region is a hole"; a spatial field can. This is the same as prediction 1.
- TarMAC reports that more communication rounds helped more than wider messages. Added: `s5a_r25_mean64_rounds1` and `s5a_r25_grid8_rounds1` (shared state used in the last layer only instead of all four). Comparing mean64 / grid8 with their rounds1 versions separates rounds from bandwidth.
- Evidence that shared messages are sometimes ignored exists (Lowe et al. 2019); that is what the wrong-shared-state swap control is for.

## Follow-up waves (each changes one thing relative to the first wave)

- B. Without a clock (v1 data, where r=0.40 no-clock has a baseline: 1.44 / 1.55, 60% of shapes in one piece): does a shared state replace the clock (progress is readable from the pooled state)?
- C. Offset pile start (v2, r=0.60 baseline 2.11, 2.10): can the swarm centroid/density in the shared state replace the rigid travel phase?
- D. Repair: damage-trained model with and without shared state; can it tell *where* the hole is?
- E. Same variants at r=0.40 (does it still help once the radius is enough?).
- F. Realism: noisy, stale or lossy aggregate (a consensus that has not converged); fewer than 4 aggregation rounds per step.

## What this does not claim

Each aggregate here is exact and available at every layer of every step. A real swarm gets approximate aggregates after several gossip rounds; wave F tests how much that matters. Nothing here is a statement about hardware until the physical numbers arrive.
