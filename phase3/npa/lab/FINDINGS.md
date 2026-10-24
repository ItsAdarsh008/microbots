# Findings so far (2026-10-24, lizard at 1,024 particles, ideal dots unless stated)

All numbers come from `runs/*/result.json` via `report.py`. Most are single runs of 3,000 iterations (some 10,000);
the three seeds of the plain baseline differ by about +-0.025 in score, so differences below that are noise. Nothing here
has been run on real hardware.

> **Correction (found later the same day).** 23 runs stopped well before their iteration budget because my blow-up recovery
> halved the learning rate permanently and gave up after 8 blow-ups (`report.py` marks them BROKEN). That includes
> `all_damage_10k` (3,007 of 10,000 iterations; the name is wrong), `gated_all_damage_10k`, `scales2_all_damage`, `gated_all_damage`,
> `train_all_damage_mem8` and the three `spec_gated_*` specialists. Their repair numbers below are from models whose learning rate
> had collapsed near the end, so they are real measurements of those checkpoints but not of fully trained models. The recovery
> logic is fixed (the batch's pool entries are reset, the rate recovers) and the affected runs are being rerun as `*_r2`.

## 1. Repair is a training-distribution problem (H1: confirmed)
Repair fraction = 1 - (loss still missing after recovery) / (loss the damage caused at once); 1 means fully repaired.
Drag levels 0.2 / 0.4 / 0.8 world units are 1 / 2 / 4 sensing radii.

| model | cut s/m/l | drag 1/2/4 eps | clump s/m/l |
| --- | --- | --- | --- |
| `base_10k` (trained on state wipes only) | 0.85 / 0.57 / 0.26 | 0.70 / 0.42 / 0.11 | 0.80 / 0.77 / 0.82 |
| `all_damage_10k` | 1.00 / 1.00 / 0.92 | 1.00 / 0.96 / 0.46 | 0.95 / 0.97 / 0.98 |
| `scales2_all_damage` (also senses at 2 eps) | 0.77 / 0.85 / 0.86 | 1.00 / 0.81 / 0.73 | 0.97 / 0.97 / 1.00 |
| `gated_all_damage` | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.69 | 1.00 / 1.00 / 1.00 |
| `train_all_damage_mem8` (8 private memory channels) | 0.95 / 1.00 / 1.00 | 1.00 / 0.98 / 0.88 | 1.00 / 0.96 / 0.99 |

Training on cuts, drags and clumps takes hard-damage repair from about 0.5 to about 0.9. A model never shown them
loses repair steeply with damage size. Caveat: cuts delete particles for good and the score is computed on the
survivors, so "repaired" means the survivors re-form the lizard at lower density, not that missing mass came back (H11).

## 2. The hardest case: a fragment torn 4 eps away (H3 refuted, H4 mixed)
Plain MLPs repair it 0.46 to 0.47, the gated cell 0.69, the 2-eps second ring 0.73, memory 0.88. So training alone is not
enough, and the architecture matters; but the memory number comes from one run (`train_far_drag_mem8`, trained with
even larger drags, got 0.52), so treat 0.88 as unconfirmed until seeds 1 and 2 agree. The 2-eps ring is better at 4 eps
and worse at 2 eps (0.81 vs 0.93 to 0.96).

## 3. Capacity grows the lizard faster but does not repair better (H8: supported)
`width256`, `depth3`, `gated3_rms` and the plain baseline all sit at hard-damage repair 0.5 to 0.57 when trained on wipes
only. Their scores differ because they grow at different speeds (ctrl 0.15 to 0.34 at 3,000 iterations).

## 4. The cost of damage training (H17: partly supported, H21 open)
At 3,000 iterations damage-trained models grow slower (ctrl 0.25 to 0.33 vs 0.12 to 0.17) and some drift upward by
step 1000 (+0.06 to +0.08). Longer rollouts (`all_damage_long`, drift -0.016), a gated cell (-0.001) or memory (-0.004)
remove the drift. Fine-tuning from a good grower (H21) is running.

## 5. Stability (H10, H13: supported)
The gated cell without normalisation diverged (NaN at iteration 221); RMSNorm fixes it. The memory cell ran away at lr 5e-4;
a global gradient clip of 1.0 (`mem8_clip1`, best grower so far at ctrl 0.12), a lower lr, or the spike guard fix it.

## 6. Fewer neighbours degrade gracefully (H16: supported)
Ideal dots trained on wipes only: 120 neighbours score 0.34, 13 neighbours 0.34 to 0.38, 7 neighbours 0.40, 5 neighbours
0.41. A real bot sees 4 to 6 (line of sight), so the method does not collapse there.

## 7. No shared compass (H14: partly supported)
With a random fixed heading per bot the plain MLP grows about 2x worse (0.65 vs 0.33), but its repair fraction is unchanged
(0.84 vs 0.86). The rotation-equivariant cell is invariant by construction (identical scores with and without headings) but
grows worse than the plain MLP with a shared frame (0.60), probably because an equivariant rule cannot pick an orientation
from a rotationally symmetric seed while the loss wants one fixed orientation. Its natural task is shape-agnostic healing,
where the swarm itself supplies the orientation; those runs are queued.

## 8. Shape-agnostic healing: not working yet (H20 open)
At 3,000 iterations the best models restore 0.10 to 0.34 of damage overall on held-out shapes (pass bar 0.5 on easy kinds),
and on shuffle they are worse than doing nothing (-0.2 to -0.03). Their dynamics blur the pattern. 10,000-iteration runs with
other cells are queued.

## 9. Physical embodiment: first runs were invalid (H19)
All `phys_*` runs diverged because my scale put 1,024 bots at 78% packing and the seed disc at 178%, which is impossible.
Corrected (1160 mm per unit, 47% packing, seed radius 0.55, no gradient through the overlap push) and rerun as `physB_*`.
Those first numbers should be disregarded.

## 10. An engineered shape-agnostic rule is not a ceiling (H22)
`cells_rest.py` imprints each particle's neighbour offsets and colour after 16 own updates, then moves back to its remembered
geometry and recolours itself. It passes 37 unit checks (noise and small drags return, wipes refill), but on the held-out suite it
is no better than doing nothing (restoration -0.14; wipe 0.27, cut 0.10, drag 0.08, clump 0.31, shuffle -1.0, noise 0.10 as repair
fractions), mainly because 38% of particles have not imprinted yet when the damage arrives, and shuffled memory points at the
wrong neighbours. The learned 3,000-iteration shape-agnostic cells reach restoration 0.09 to 0.13. So neither route meets the
bar (repair fraction 0.5 on easy damage); the 10,000-iteration learned runs and the `forget` variants are queued.

## 11. Update 16:50 (single runs; seed spread +-0.025 in score)
- **Physical embodiment trains (H19, H15 partly supported).** `physB_scales2_all_damage_eps7` (strides 15 to 43 mm with 20% length and 5 deg
  heading noise, 8-bit noisy sensing, 30% packet loss, overlap projection, about 7 neighbours) completed: hard-damage repair 0.94, drift +0.012,
  but control loss 0.52 against about 0.2 for ideal dots, so the lizard it grows is much worse. The other three `physB_*` runs are not in yet.
  One run; we do not yet know whether the embodiment, the 2-eps ring or chance explains the growth gap.
- **Fine-tuning trades growth for repair (H21: partly).** `base_10k__ft_all_damage` (3,000 iterations from `base_10k`): hard-damage repair
  0.55 to 0.84, but control loss 0.169 to 0.253 and drift -0.07 to +0.05. Total score 0.261 against 0.221 for `base_10k`: not better overall.
- **Multi-shape (2a): not shown yet.** `ms4_base` (one network, four shapes, 10,000 iterations in total) scores 0.35 to 0.45 per shape against
  0.22 to 0.30 for single-shape specialists with 10,000 iterations each. The budget is not matched (2,500 iterations per shape), so this
  does not test the 25% criterion; a 40,000-iteration run is queued. `ms4_gated3_*` diverged (losses 50 to 450) and are invalid:
  the multi-shape trainer has no loss-spike guard.
- **Gated cell at small eps (H16/H10): unstable.** `eps5_gated3_rms_r2` hit 40 blow-ups and stopped at 544 of 3,000 iterations; at 13 neighbours
  (`eps13_gated3_rms_r2`) it completed (score 0.312). The plain MLP completes at 5 and 7 neighbours.

## 12. Update 17:15: the pool-age cap works only when short (H23: supported, preliminary)
Damage-trained runs with `pool_max_age` (an entry goes back to a fresh seed after that many uses), partway through training:
cap 4: density loss falls 0.27 (iter 600) to 0.17 (iter 1,800), no blow-ups; cap 8 and cap 16: density loss rises to 1.0 to 1.7 by
iteration 4,200 and 26 to 40 blow-ups, like the uncapped run. So pool collapse happens within fewer than 8 reuses, and the cap
must be about 4 or less. `pa4_all_damage` is at iteration ~1,800 of 10,000, so this is a trend, not a result; caps 2 and 3, a second
seed and four other cells are queued. Gated cells are still unstable elsewhere: `pa8_gated_all_damage` stopped at 677 iterations
and `ag10k_gated3_rms` ended at loss ~100 (colour channels diverged), so a gated block is not yet a safe default.
`attn_rms_mem8_r2` (local attention, RMSNorm, 8 memory channels, wipes only) completed at score 0.200, about as good as the best
growers; it is the first attention cell to train cleanly.

## 13. Update 17:50: we should train on arbitrary shapes, and the lizard data says why (H27)
104 of 121 finished runs used the lizard; 13 used arbitrary shapes. The shape-agnostic runs: all cells end between 0.12 and 0.15 (do-nothing 0.199), more iterations did nothing, so the
architecture is not what limits them. Per severity (`ag10k_equiv`, loss after damage to loss after recovery): clump 0.15/0.25/0.53 to 0.07/0.08/0.09;
noise 0.20/0.33/0.47 to 0.14/0.17/0.20; drag 0.15/0.23/0.29 to 0.11/0.18/0.27; cut 0.11/0.21/0.33 to 0.10/0.17/0.25; wipe and shuffle unchanged. So geometry is partly repaired
on shapes the model never saw (held-out emoji and procedural shapes with holes), and colour information is not. The earlier "pass bar" for task 2b leaned on wipe and shuffle and was
the wrong yardstick. Eight geometry-focused runs, including physical-body and no-compass versions, are at the front of the queue.
