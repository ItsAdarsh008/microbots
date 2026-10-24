# Findings so far (2026-10-24, lizard at 1,024 particles, ideal dots unless stated)

All numbers come from `runs/*/result.json` via `report.py`. Most are single runs of 3,000 iterations (some 10,000);
the three seeds of the plain baseline differ by about +-0.025 in score, so differences below that are noise. Nothing here
has been run on real hardware.

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
