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

## 14. Update 18:00: damage training that keeps growth (H23, H21: supported by one run)
`pa4_all_damage` (damage on every kind, pool entries reset to a fresh seed after 4 uses, 10,000 iterations, completed, 0 blow-ups):
control loss 0.171 (the wipes-only `base_10k` is 0.169, the earlier damage-trained runs 0.19 to 0.26), hard-damage repair 0.81 (`base_10k` about 0.55),
hard-damage excess +0.015 (`base_10k` +0.100), drift +0.075. So it grows as well as the baseline and repairs far better; the cost is a drift of +0.075 by step 1,000
(the baseline drifts -0.07). Caps 8 and 16 stopped again at about 4,900 of 10,000 iterations (43 and 40 blow-ups), as did `pa4_mem8_all_damage` (830 iterations),
so the cap helps the plain MLP but not memory or gated cells (H26 stays open). One seed; `pa4_all_damage_s1`, caps 2 and 3, and the physical-body deployment
(`pa4_all_damage__phys`, body noise and strides only, ideal sensing radius) are queued.

## 15. Update 18:25: the best lizard models so far, and what they are (H23, H26 revised)

> **CORRECTED at 00:30 (section 30): `pa8_mem8_all_damage` did not replicate: seed 1 diverged (score 2.17). The two-radius model `pa8_scales2_all_damage` did replicate on three seeds.**
`pa8_scales2_all_damage` (plain MLP that also senses at 2 eps, 19k parameters, damage on every kind, pool entries reset after 8 uses, 10,000 iterations, no
blow-ups): score 0.085, control loss 0.084, grown-lizard IoU 0.955, hard-damage repair 0.99, drift 0.000. `pa8_mem8_all_damage` (gated cell with RMSNorm and 8 private
memory channels, same recipe, 36 blow-ups recovered): score 0.086, control 0.080, repair 0.92, drift 0. For comparison: `base_10k` 0.221, `pa4_all_damage` 0.184,
`mem8_clip1` 0.174. Per-kind scores of the first are 0.082 to 0.089 on every damage kind, i.e. damage costs almost nothing over the undamaged lizard. Checks: training
completed all iterations, the evaluator files are the same as `pa4_all_damage`'s (only `lab/evaluate.py`'s seed-radius edit separates them from `base_10k`, which
does not change anything at eps 0.2), and the loss fell steadily. One seed each; seeds 1 and 2 are queued, so treat the size of the gain as unconfirmed.
What this changes: the plain MLP needs a cap of 4 or less (cap 8 collapsed it), but with 2-eps sensing or memory a cap of 8 is enough, so the extra sensing range or memory
makes the damage-trained rule stable (H26 was wrong that memory cells cannot train; it was the combination of gated block and cap 4 that stopped at 830 iterations).
It is a lizard result. The same recipe is queued on four shapes in one network (`ms4_*_pa8`, 40,000 iterations), which is the transfer test that matters.

## 16. Update 18:50: three cells, one recipe; geometry-only agnostic training is too active (H29 supported, H27 revised, H30)

> **CORRECTED at 00:30 (section 30): the statement below that three cells reach 0.085 to 0.098 rested on single seeds; the gated memory cell diverged on its second seed.**
- `pa4_attn_mem8_all_damage` (local attention + RMSNorm + 8 memory channels, cap 4): score 0.098, control 0.093, hard-damage repair 0.97, drift 0.00. With `pa8_scales2_all_damage` (0.085)
  and `pa8_mem8_all_damage` (0.086), three different cells reach 0.085 to 0.098 with damage training and a pool-age cap, so the recipe is not specific to one cell (still one seed each).
  Plain MLP: cap 3 gives 0.180 (repair 0.85), cap 4 0.184 (0.81), cap 2 0.269 (0.49, growth suffers); caps 8 and 16 stopped at about 4,900 iterations.
- Shape-agnostic, geometry-focused training (`ag_geo_equiv`, `ag_all6_heavy_mlp`) scores 0.207 and 0.188 against a do-nothing floor of 0.199: worse than the earlier `ag10k_*` (0.12 to 0.14). Per
  severity they repair more (after damage 0.65 to 1.09, after recovery 0.17 to 0.22 for `ag_geo_equiv`) but they also change the undamaged swarm (control 0.154 against 0.005): the swarm settles
  into an attractor that does not depend on the damage and is not the original configuration. The training did what it was asked (repair damage) at the cost of fidelity. `warm_loss`
  (penalise drift during the undamaged warm-up) is the existing lever; five runs with it are queued.

## 17. Update 19:15: the browser engine now runs two-radius models
`web/npa.js` senses at eps and 2 eps when a model has `scales: [1, 2]`. Parity against `cells.Cell(scales=(1.0, 2.0))` in JAX over 5 steps with identical update masks:
7e-8 in position and 2e-6 in state after one step, 4e-6 and 1e-4 after five (my first version was wrong by 0.29 in state: it used the doubled radius in the gradient scale, caught by this
check). The plain-network parity (1e-5) is unchanged. `pa8_scales2_all_damage` now has a repair clip on the dashboard and is in the public demo. Frames at step 192 and 440 (head wiped,
tail cut to 970 particles, foot torn off) show a lizard with all legs and the curled tail, qualitatively consistent with its score of 0.085 (one seed).

## 18. Update 19:20: shape-agnostic, geometry-trained (H27, H30)
Loss after recovery against the do-nothing floor, by severity (small/medium/large), unseen shapes, ideal body unless stated:
- `ag10k_equiv` (damage mix p 0.2, 10,000 iterations): clump 0.07/0.08/0.09 vs 0.15/0.25/0.53; noise 0.14/0.17/0.20 vs 0.20/0.33/0.47; control 0.005.
- `ag_geo_attn_mem8` (geometry damage, 6,000 iterations): clump 0.11/0.12/0.13; noise 0.14/0.17/0.20; cut 0.12/0.16/0.23 (do-nothing 0.11/0.21/0.33); control 0.062.
- `ag_geo_scales2`: nearly identical to `ag_geo_attn_mem8` (control 0.073). So the cells are interchangeable again here; geometry-only training did not beat the mixed-damage run on fidelity.
- `ag_geo_phys_mlph_random` (physical body: strides, noisy 8-bit sensing, packet loss; every bot with its own random heading): NOT working yet. Control loss 0.217, worse than doing nothing at small and
  medium damage (e.g. clump 0.23 vs 0.15 at the small level), better only at the largest (clump 0.25 vs 0.53). The body noise and strides alone keep it from holding the original shape.
  One run, 6,000 iterations; the equivariant physical run and the `warm_loss` runs are queued. The headline for the robot goal is that shape-agnostic repair is shown with ideal bodies and
  unseen shapes, and not yet with the physical body.

## 19. Update 19:45: with the assumed body model, the cell and the compass do not matter (H14, H31)
Shape-agnostic, geometry-damage training, body model (strides, noisy 8-bit sensing, 30% packet loss, about 7 neighbours), 6,000 iterations, one run each: plain MLP in a shared frame
score 0.262 (undamaged control 0.234); plain MLP with a random heading per bot 0.251 (0.217); rotation-equivariant cell with random headings 0.282 (0.253). The spread between them (0.03)
is about the seed spread, so the data do not show a cost of having no compass, or any benefit of the equivariant cell, in this setting. All three have an undamaged control of 0.22 to 0.25, so the
body model's step and noise limits, not the cell, set the fidelity floor; the four ablations (`ag_phys_abl_*`) are the test of which part of the body model does it.

## 20. Update 20:10: the drift penalty fixes the fidelity loss (H30: supported)
`ag_geo_warm1_equiv` (rotation-equivariant cell, geometry damage, `warm_loss` 1.0, 6,000 iterations): undamaged control 0.025 (0.154 without the penalty), score 0.147; better than
doing nothing at every severity (clump 0.09/0.10/0.11 against 0.15/0.25/0.53; noise 0.14/0.18/0.20 against 0.20/0.33/0.47; cut 0.10/0.16/0.24 against 0.11/0.21/0.33; drag 0.12/0.19/0.27 against 0.15/0.23/0.29).
`ag_geo_warm1_mlp`: control 0.053, score 0.164, same pattern slightly weaker. This matches the earlier `ag10k_equiv` (score 0.122, 10,000 iterations, mixed damage) and does not beat it,
so on unseen shapes with an ideal body, the best we have is: clump and noise repaired well, tears and cuts only a little, colour wipes not at all. One run per cell. The unfixed gap is cuts
(survivors only) and large tears; both need either more mass than the swarm has or sensing beyond a fragment's neighbours.

## 21. Update 20:40: seeds, drift, the drift penalty across cells, and which part of the body model hurts
- **H29 (two seeds):** `pa8_scales2_all_damage_s1` scores 0.081 (control 0.078, repair 0.98) against 0.085 for seed 0; the difference, 0.004, is far below the +-0.025 seed spread. The other two winners' seeds are still running.
- **H28:** `pa4_long_all_damage` (pool cap 4, rollouts of 64 to 160 steps): score 0.135 (was 0.184), control 0.104, drift +0.019 (was +0.075), hard-damage repair 0.83. Longer rollouts under the cap cut the drift without losing repair. One seed.
- **H30 across cells:** `ag_geo_warm1_attn_mem8` control 0.019, `ag_geo_warm1_equiv` 0.025, `ag_geo_warm3_scales2` 0.035, `ag_geo_warm1_mlp` 0.053 (all 0.06 to 0.15 without the penalty). The drift penalty works for every cell.
- **H31 (assumed body model, shape-agnostic, one run each):** undamaged control with the full model 0.217; without sensing noise 0.224; without packet loss 0.221; with only the step cap and the no-overlap rule 0.089
  (`ag_phys_abl_body_only`). The sensing noise and packet loss do not matter; the stride rules (18 mm minimum stride, 20% stride noise) do. The `nostride` run is still pending. This result is
  now partly moot: the MuJoCo measurements on the CAD meshes (`assets/microbot_v1/mujoco/FINDINGS.md`) show the 18 mm minimum stride is not a property of this gait (strides scale at about 0.3 mm per degree);
  the real constraint found there is one-line motion (H33), tested by the `ag_axis_*` runs.

## 22. Update 21:00: sensing at about 7 neighbours keeps fidelity but costs repair (H32: partly supported, one run)

> **CORRECTED at 22:55 (section 26): a second seed has control 0.076, not 0.001; the fidelity claim here came from a lucky seed. The repair numbers replicate.**
`ag_eps7_ideal_warm1_equiv` (ideal body, eps 0.05 so about 7 neighbours instead of 120, equivariant cell, `warm_loss` 1.0, 10,000 iterations, unseen shapes): undamaged control 0.001 and score 0.141,
against control 0.025 and score 0.147 for the same cell at eps 0.2. So with an ideal body the swarm holds its shape perfectly while sensing only 7 neighbours; the fidelity loss in the body-model runs comes from
the body model's motion and noise, not from how few neighbours the bots see. Repair is weaker, though: gain over doing nothing at small/medium/large severity is clump 0.05/0.14/0.39 (eps 0.2: 0.06/0.15/0.42), noise -0.01/0.04/0.13 (0.05/0.16/0.27), drag 0.00/0.01/0.01 (0.03/0.04/0.02), cut 0.00/0.01/0.01 (0.01/0.05/0.09).
So at 7 neighbours the swarm keeps its shape but repairs noise, tears and cuts much less; only clump damage is repaired about as well. H32's claim holds for fidelity, not for repair.

## 23. Update 21:30: H31 confirmed; the harm was the stride rules I assumed (one run per ablation)
Undamaged control of the shape-agnostic runs under the assumed body model (geometry damage, `warm_loss` 1.0 where noted, about 7 neighbours): full model 0.217; without sensing noise 0.224; without packet loss 0.221;
without the stride rules (`ag_phys_abl_nostride`: no minimum stride, no stride noise) 0.075; only the step cap and the no-overlap rule (`ag_phys_abl_body_only`) 0.089. Sensing noise and packet loss do not matter; the stride
deadzone and stride noise do. The MuJoCo measurements on the CAD meshes (`assets/microbot_v1/mujoco/FINDINGS.md`) show that this bot has no minimum stride (about 0.3 mm per degree of azimuth sweep), so the harmful
part of the assumed model is not a property of the real body. The gated specialists (`spec_gated_*`) keep blowing up (H26) and are not used in the multi-shape comparison; `spec_base_*` are.

## 24. Update 22:10: a third seed for the winner, and the first result under the measured one-line body model

> **CORRECTED at 22:55 (section 26): the one-line-motion comparison in the second bullet used the wrong reference (120 neighbours); the lizard seed result in the first bullet stands.**
- **H29 (three seeds):** `pa8_scales2_all_damage` scores 0.085, 0.081 (_s1) and 0.085 (_s2); control 0.078 to 0.084; hard-damage repair 0.98 to 0.99. The spread, 0.004, is far below the +-0.025 seed spread of the earlier runs. Confirmed on the lizard for this cell; `pa8_mem8_all_damage_s1` is still running.
- **H33 (one run):** `ag_axis_equiv_random` (shape-agnostic, unseen shapes; body model with one-line motion per bot, measured drift, no minimum stride, 30% packet loss, sensing noise, each bot with its own random heading; equivariant cell; about 7 neighbours): undamaged control 0.077 (the old assumed body model: 0.253 for the same cell; free motion: 0.025), score 0.163 against a do-nothing floor of 0.181. Gain over doing nothing at small/medium/large severity:
clump 0.01/0.10/0.36 (free motion 0.06/0.15/0.42), noise -0.02/0.05/0.16 (0.05/0.16/0.27), drag -0.01/0.01/0.01, cut -0.02/0.01/0.03. So with the measured one-line motion the swarm keeps its shape and still repairs clump and noise damage at
medium and large severity, but less than with free motion, tears and cuts barely at all, and it is slightly worse than doing nothing at small damage. One run; `ag_axis_mlph_random`, `ag_axis_mlph_scales2` and the two `ag_axis_ideal_*` runs are still to come.

## 25. Update 22:35: one-line motion costs repair, not noise (H33, H36; three runs)

> **CORRECTED at 22:55 (see section 26): the comparison below is confounded. The free-motion reference used 120 neighbours, the one-line runs 7; against free motion at 7 neighbours the one-line constraint costs almost nothing. Read section 26 before using the conclusions of this section.**
Gain over doing nothing at small/medium/large severity, unseen shapes, body model with one-line motion (measured drift, no minimum stride), about 7 neighbours: `ag_axis_ideal_mlph` (no sensing noise or packet loss) clump 0.01/0.10/0.36, noise -0.03/0.01/0.11, drag -0.01/0.01/0.01, cut -0.01/0.01/0.02, control 0.064;
`ag_axis_mlph_random` (noise, packet loss, random headings) clump 0.01/0.10/0.36, noise -0.02/0.04/0.15, drag 0.00/0.01/0.01, cut -0.01/0.01/0.02, control 0.061; `ag_axis_equiv_random` as in section 24 (control 0.077). Free motion (`ag_geo_warm1_equiv`):
clump 0.06/0.15/0.42, noise 0.05/0.16/0.27, drag 0.03/0.04/0.02, cut 0.01/0.05/0.09, control 0.025. So the one-line constraint costs a third to a half of the noise and cut repair and a little of the clump repair; sensing noise and
packet loss do not (the noise-free run is no better than the noisy ones) and the cell makes no difference. Three runs, one seed each. A bracket for a bot that can turn (H36) is queued.

## 26. Correction 22:55: the one-line constraint costs almost nothing; I had compared against the wrong reference (H33, H36)
**What was wrong.** Sections 24 and 25 compared the one-line-motion runs (about 7 neighbours, eps 0.05) with the free-motion run `ag_geo_warm1_equiv` at eps 0.2, i.e. about 120 neighbours. That confounds the motion constraint with the number of neighbours. The right reference is free
motion at the same 7 neighbours: `ag_eps7_ideal_warm1_equiv` (seed 0) and `_s1` (seed 1; ideal body, equivariant cell, `warm_loss` 1.0, 10,000 iterations).

**The like-for-like numbers** (gain over doing nothing at small/medium/large severity; control = undamaged loss):

| Run (about 7 neighbours) | Control | Clump | Noise | Drag | Cut |
| --- | --- | --- | --- | --- | --- |
| free motion, seed 0 | 0.001 | 0.05/0.13/0.39 | -0.01/0.04/0.13 | 0.00/0.01/0.01 | 0.00/0.01/0.01 |
| free motion, seed 1 | 0.076 | 0.02/0.11/0.38 | -0.05/0.03/0.15 | -0.01/0.02/0.02 | -0.01/0.02/0.04 |
| one-line, noisy, random headings (`ag_axis_mlph_random`) | 0.061 | 0.01/0.10/0.36 | -0.02/0.04/0.15 | 0.00/0.01/0.01 | -0.01/0.01/0.02 |
| one-line, no noise (`ag_axis_ideal_mlph`) | 0.064 | 0.01/0.10/0.36 | -0.03/0.01/0.11 | -0.01/0.01/0.01 | -0.01/0.01/0.02 |
| one-line, equivariant (`ag_axis_equiv_random`) | 0.077 | 0.01/0.10/0.36 | -0.02/0.05/0.16 | -0.01/0.01/0.01 | -0.02/0.01/0.03 |

**What stands.** (1) At 7 neighbours, with free or one-line motion, the swarm repairs clump damage at medium and large severity (about 0.10 and 0.36 to 0.39) and noise at large severity (0.11 to 0.16), and essentially nothing else; the one-line constraint costs at most a few hundredths. (2) The big drop in repair
(noise 0.26 to 0.13, cut 0.09 to 0.02, drag 0.02 to about 0) comes from having 7 neighbours instead of 120, not from the body. (3) Fidelity at 7 neighbours is seed-dependent (control 0.001 and 0.076 for free motion); the one-line runs sit in that range (0.059 to 0.077), so no cost of the constraint on fidelity is
shown either. (4) At eps 0.2 two seeds of the free-motion run agree (control 0.025 and 0.030; gains within 0.02). (5) The turn-action bracket (`ag_dual*`, queued) will therefore probably show little gain; it remains a direct test. What this says for the project: sensing radius, not the motion constraint, is the limiting factor for repair on
the measured body, so the wider-ring and memory fixes (H35, runs queued) matter more than steering.

## 27. Update 23:20: a wider sensing ring recovers the repair lost at 7 neighbours (H35: supported, one seed); the one-line constraint then matters for cuts and tears
Replicates: `ag_geo_warm1_equiv` (free motion, 120 neighbours) scores 0.147, 0.160, 0.153 over three seeds (control 0.025, 0.030, 0.027); `ag_geo_warm1_attn_mem8` 0.148 and 0.148 over two (control 0.019, 0.021). Both replicate well.
Gain over doing nothing at small/medium/large severity at about 7 neighbours (eps 0.05):

| Run | Control | Clump | Noise | Drag | Cut |
| --- | --- | --- | --- | --- | --- |
| free, one ring (`ag_eps7_ideal_warm1_equiv`, 2 seeds) | 0.001, 0.076 | 0.05/0.13/0.39, 0.02/0.11/0.38 | -0.01/0.04/0.13, -0.05/0.03/0.15 | 0.00/0.01/0.01, -0.01/0.02/0.02 | 0.00/0.01/0.01, -0.01/0.02/0.04 |
| **free, two rings** (`ag_eps7_scales2_warm1`) | 0.029 | 0.03/0.13/0.40 | -0.01/0.07/0.21 | 0.02/0.04/0.03 | 0.00/0.05/0.10 |
| one-line, one ring (`ag_axis_mlph_random`) | 0.061 | 0.01/0.10/0.36 | -0.02/0.04/0.15 | 0.00/0.01/0.01 | -0.01/0.01/0.02 |
| one-line, two rings (`ag_axis_mlph_scales2`) | 0.048 | 0.03/0.13/0.39 | 0.00/0.07/0.19 | 0.00/0.02/0.01 | 0.00/0.02/0.03 |
| (reference: free, 120 neighbours, `ag_geo_warm1_equiv`) | 0.025 | 0.06/0.15/0.42 | 0.05/0.16/0.27 | 0.03/0.04/0.02 | 0.01/0.05/0.09 |

**Reading.** (1) A second sensing ring (out to twice the radius) at 7 neighbours brings back most of the repair of the 120-neighbour model: noise 0.21 (0.27), cut 0.10 (0.09), drag 0.03 to 0.04 (0.02 to 0.04), and keeps the control low. (2) With that sensing in place, the one-line
constraint does cost repair: with two rings, cut 0.10 free vs 0.03 one-line, drag 0.04 vs 0.01, while noise (0.21 vs 0.19) and clump hardly change. So the earlier conclusion (section 26) that the constraint costs almost nothing holds for a sensing-starved swarm, not once the
swarm can sense enough to repair cuts and tears. (3) Caveats: the free and one-line runs differ in more than motion (the one-line ones also have random headings, sensing noise and packet loss); each is one seed (replicates queued: `ag_eps7_scales2_warm1_s1`, `ag_axis_mlph_scales2_s1`); and the second ring is an assumption about
the hardware: it needs sensing or relayed messages out to about 4 body diameters (118 mm at this scale), which line-of-sight occlusion may prevent. The turn bracket with two rings (`ag_dual_mlph_scales2`, `ag_dualtax_mlph_scales2`, queued) is the direct test of whether turning gives the cut and tear repair back.

## 28. Update 23:55: the "turn tax" bracket was invalid; replaced by a controllable turn (H36, H39)
`ag_dualtax_equiv_random` (every tick, half the bots are forced to turn: no stride and a random 28 mm sideways shift) ends with an undamaged control of 0.428 and a final loss of about 0.43 for every damage kind and severity, which is the swarm sitting in a scrambled steady state
(do-nothing floor 0.181). That says nothing about turning: the model forces random shifts the cell cannot avoid, and a real bot turns only when it decides to. The other two tax runs (`ag_dualtax_mlph_random`, `ag_dualtax_mlph_scales2`) were already running when I saw this and are
invalid for the same reason; ignore them. The optimistic bracket without a tax (`ag_dual_*`, a random one of two lines per tick, free turning) is still a valid upper bound.
`embody.py` now has `axis_turn`: one extra mem channel per bot holds which of its two orthogonal lines it is on; a proposal mostly along the other line makes the tick a turn (no stride, the measured side shift of 32 mm along the old line with random sign), after which the
bot uses the new line; a proposal along its current line is a normal stride; a bot that is not updating never turns. Unit-tested (four cases) and smoke-tested through the trainer and evaluator. The perception frame is kept fixed (a real bot knows its turn from its IMU).
Three runs under it are queued: `ag_turn_mlph_scales2`, `ag_turn_mlph_random`, `ag_turn_equiv_random`. Compare with `ag_axis_mlph_scales2` and `ag_axis_mlph_random` (one line) and `ag_eps7_scales2_warm1` (free motion).

## 29. Update 00:10: memory also recovers repair at 7 neighbours; the optimistic turn bracket adds nothing at a single ring; the four-shape transfer test never ran (H35, H36, H25)
- **H35 (memory, one seed):** `ag_eps7_attn_mem8_warm1` (free motion, attention + 8 memory channels, 7 neighbours) gains 0.02/0.12/0.39 on clump, -0.04/0.06/0.20 on noise, 0.00/0.02/0.02 on drag and -0.02/0.03/0.07 on cut; the single-ring reference (`ag_eps7_ideal_warm1_equiv`) is
  noise 0.13 to 0.15 and cut 0.01 to 0.04 at large severity. So memory, like a second ring, brings back part of the lost repair (noise 0.20 vs 0.13 to 0.15; cut 0.07 vs 0.01 to 0.04), less than the two-ring cell on cuts (0.10). Undamaged control 0.075 (seed-dependent at 7 neighbours).
- **H36, single ring:** `ag_dual_mlph_random` (each tick a random one of the two lines, turning free) gains clump 0.01/0.10/0.36, noise -0.04/0.03/0.15, drag -0.01/0.01/0.01, cut -0.01/0.02/0.03: the same as one-line `ag_axis_mlph_random`. With one ring, even free turning adds nothing,
  consistent with H37 (sensing, not motion, is the limit there). The test that matters, with two rings (`ag_dual_mlph_scales2`, `ag_turn_*`), is still running.
- **Process failure found (mine):** the queue runner killed every job after 135 minutes (120 default plus 15 grace). `ms4_base_40k` was killed twice at iteration 31,000 of 40,000 and then failed; `ms4_scales2_pa8`, `ms4_mem8_pa8` and `ms4_scales2_pa4` (the four-shape transfer test, 40,000 iterations) were
  restarted from scratch three times each and `ag_geo_warm1_equiv_20k` was heading the same way. So **the multi-shape transfer test (H25) has produced nothing yet** despite about six hours of GPU time. The runner now allows about 0.6 s per iteration (or the config's `max_minutes`), and the five runs are requeued as `*_L` with `max_minutes`
  480 (240 for the 20k run); the old copies will keep wasting slots until their third attempt times out. `health.py` already reports a job in `failed/`; the recurring check now has to read the cause, including timeouts.

## 30. Correction 00:30: the memory-cell lizard winner does not replicate; and a reporting blind spot (H29)
- **`pa8_mem8_all_damage_s1`** (gated cell with RMSNorm and 8 private memory channels, damage on every kind, pool age cap 8, 10,000 iterations) **diverged**: 27 blow-up recoveries by iteration 1,193, then it finished with a training loss near 59 and scores 2.168 (control 2.098), against 0.086 for seed 0.
  So that cell's result is seed-dependent (one good seed in two); the gated-plus-memory block remains unstable under damage training (H26). The two-radius `pa8_scales2_all_damage` replicated on three seeds (0.085, 0.081, 0.085), so H29 stands for that cell only. `pa4_attn_mem8_all_damage` (0.098) is still a single seed.
- **Reporting blind spot (mine):** `lab/report.py` prints only the best 60 runs per shape by default, and the lizard has more than 100, so any run worse than rank 60 was silently omitted from every check that grepped its output, including this run's result for hours.
  The checks now call `report.py --top 400`. Earlier "no result yet" statements about lizard runs may have been this, not a missing result.

## 31. Update 00:55: the first controllable-turn run is a failure; the likely cause is a threshold in my turn rule (H39, H41); the optimistic two-ring bracket adds little
- **`ag_turn_mlph_random`** (one ring, controllable turn, 6,000 iterations, no blow-ups): the undamaged swarm ends at loss 0.314 against 0.061 (`ag_axis_mlph_random`) and 0.068 (`ag_dual_mlph_random`), and every damage kind is worse than doing nothing except large clump (+0.20) and large noise (+0.05): gain over do-nothing, small/medium/large: clump -0.17/-0.07/+0.20, noise -0.23/-0.11/+0.05, drag -0.18/-0.12/-0.09, cut -0.21/-0.13/-0.06. Final training loss 0.73. One seed. Gains are the per-severity do-nothing minus final loss, averaged over the held-out shapes.
- **Likely cause (mine, checked only in a unit test):** in `embody.py` a bot turned whenever its proposal exceeded 1e-7 in length and pointed mostly along the other line. A network that never outputs exactly zero therefore flips lines all the time and each flip is a 28 mm side shift (0.57 of the 0.05 sensing radius). A test with a 0.1 mm proposal confirmed the flip. I have not shown that this explains the run's loss.
- **Change:** `turn_min` (smallest proposal that can turn, world units) and `turn_margin` (the other line's component must exceed this multiple of the current line's); defaults reproduce the old rule. Re-queued as `ag_turn2_mlph_scales2`, `ag_turn2_mlph_random`, `ag_turn2_equiv_random` with `turn_min` 0.01 (11.6 mm) and `turn_margin` 2. The three old `ag_turn_*` runs measure the old rule; read them that way.
- **`ag_dual_mlph_scales2`** (two rings, each tick a random one of the two lines, turning free; an optimistic bracket): clump 0.03/0.12/0.38, noise -0.01/0.07/0.20, drag 0.00/0.02/0.02, cut 0.00/0.03/0.05; undamaged control 0.051. That is the same as one-line with two rings (`ag_axis_mlph_scales2`: cut 0/0.02/0.03, drag 0/0.02/0.01) and below free motion (cut 0/0.05/0.10, drag 0.02/0.04/0.03). Free turning of this random kind recovers little of the cut/tear gap, so a controllable turn is not expected to recover much more; the gap may come from the other differences between the free and one-line runs (section 27 caveat 3).


## 32. Update 01:15: the free two-ring result replicates; the old-rule turn run fails the same way with the equivariant cell (H35, H39, H41)
- **`ag_eps7_scales2_warm1_s1`** (free motion, two rings, 7 neighbours, seed 1) matches seed 0 (`ag_eps7_scales2_warm1`): clump +0.04/+0.14/+0.40 (seed 0: 0.03/0.13/0.40), noise -0.01/+0.07/+0.21 (same), drag +0.02/+0.04/+0.03 (same), cut 0.00/+0.05/+0.09 (0/0.05/0.10); undamaged control 0.035 (0.029). Two seeds now agree, so the free-motion two-ring numbers are not a lucky seed. The gap to one-line two-ring (cut 0/0.02/0.03, drag 0/0.02/0.01, two seeds) therefore stands as a difference between those two body settings, with the section 27 caveat that they differ in more than motion.
- **`ag_turn_equiv_random`** (old turn rule, equivariant cell): undamaged control 0.355, gains clump -0.21/-0.11/+0.16, noise -0.27/-0.16/+0.01, drag -0.22/-0.16/-0.11, cut -0.25/-0.16/-0.08. The same failure as `ag_turn_mlph_random` (control 0.314), with a different cell, which fits a problem in the body rule rather than the cell. This does not yet test H41: the `ag_turn2_*` runs do.

## 33. Update 01:36: the old-rule turn run with two rings fails like the other two (H39, H41)
`ag_turn_mlph_scales2` (old turn rule, two rings): undamaged control 0.291, gains clump -0.15/-0.05/+0.22, noise -0.21/-0.09/+0.07, drag -0.16/-0.10/-0.07, cut -0.19/-0.10/-0.03. All three old-rule turn runs (`mlph_random` 0.314, `equiv_random` 0.355, `mlph_scales2` 0.291) end with a wrecked undamaged swarm, across two cells and with or without a second ring, which points at the body rule (H41) and not at sensing. The H41 test is still running: `ag_turn2_mlph_random` was at iteration 1,400 with training loss 0.37 (the old-rule run was at 0.84 at iteration 500 and ended at 0.73), which is encouraging but is a training loss mid-run, not a result. The two h100 jobs in `failed/` are the old `ms4_mem8_pa8` and `ms4_scales2_pa4` copies (known; the `_L` versions are the live ones).

## 34. Update 02:15: the second sensing ring would be mostly blocked if it relied on line of sight (hardware question for H35, H38)
`lab/los_check.py` packs hard discs of body radius 15 mm (0.0129 world units) by random sequential addition in a square patch at the density where a bot should have about 7 neighbours within eps = 0.05 (891 bots, three seeds, bots near the patch edge excluded) and counts neighbours whose centre-to-centre segment does not cross another body.
- Within eps: 5.7 to 6.0 neighbours in range, 5.7 to 5.9 visible (the hard cores give 5.8, not 7, at this fill; the real swarm's density was not measured, only assumed).
- Within 2 eps: 25.8 to 26.4 in range, 10.5 to 10.8 visible (about 41%). So the second ring (between eps and 2 eps) has about 20 bots in range but about 5 visible (about 24%).
- Reading: a two-ring cell that assumes it senses all bots out to 2 eps cannot get that from an optical line-of-sight sensor in a packed swarm. A relayed radio message through neighbours would not be blocked, but that is a different hardware assumption, and the cell was trained on full rings. This is a 2-D hard-disc stand-in with no sensor model; it says nothing about range-limited radio. The `ag_*scales2` and `ag_eps7_attn_mem8` gains (H35) therefore rest on an unverified sensing assumption. H42 (below) is the test: train and evaluate the two-ring cell with the second ring randomly thinned to about a quarter of its neighbours. Not run yet.

## 35. Update 02:35: with the turn rule fixed the swarm is no longer wrecked, but controllable turning adds nothing over one-line motion (H39, H41)
Per-severity gains over do-nothing (small/medium/large, mean over held-out shapes), one seed each:
| run | undamaged control | clump | noise | drag | cut |
| --- | --- | --- | --- | --- | --- |
| `ag_turn2_mlph_scales2` (turn_min 0.01, margin 2, two rings) | 0.045 | 0.04/0.13/0.39 | 0.00/0.07/0.20 | 0.00/0.02/0.01 | 0.00/0.02/0.03 |
| `ag_axis_mlph_scales2` (one-line, two rings; two seeds) | 0.048 | 0.03/0.13/0.39 | 0.00/0.07/0.19 | 0.00/0.02/0.01 | 0.00/0.02/0.03 |
| `ag_turn2_mlph_random` (turn, one ring) | 0.062 | 0.01/0.10/0.36 | -0.02/0.04/0.15 | 0.00/0.01/0.01 | -0.01/0.01/0.02 |
| `ag_axis_mlph_random` (one-line, one ring) | 0.061 | 0.01/0.10/0.36 | -0.02/0.04/0.15 | 0.00/0.01/0.01 | -0.01/0.01/0.02 |
- **H41 (partly supported):** the undamaged control is back to 0.045 and 0.062 (old rule: 0.29 to 0.36), so the earlier failure came from the turn rule (noise triggering 28 mm jumps), as the unit test suggested. I only changed the rule and did not test other causes separately.
- **H39 (not supported so far):** the turn runs reproduce the one-line numbers to two digits, in both pairs. The cut and tear gains do not rise toward the free-motion values (cut 0/0.05/0.10, drag 0.02/0.04/0.03). A match this close suggests the cells hardly turn, because a turn needs a proposal of at least 11.6 mm and twice as large along the other line; I have not measured how often bots turn, so this is a guess. The result is "no gain observed under this rule and training", not "turning cannot help".
- Not run: the third run `ag_turn2_equiv_random` is still training; the thinned second ring (H42); a measurement of the turn rate.

## 36. Correction 03:10 to section 35: the fixed-rule turn runs never turn, so they say nothing about turning (H39, H41)
`lab/turn_rate.py` counts changes of the per-bot line-state bit over the whole shape-agnostic evaluation (growth, control and all damage runs, 118,374,400 bot-ticks counting every bot).
- `ag_turn2_mlph_random`: 0 turns. `ag_turn2_mlph_scales2`: 0 turns. 
- Instrument check: the old-rule `ag_turn_mlph_random` gives 17,190,258 turns (0.145 per bot-tick), so the counter does count turns when they happen.
- So the fixed rule (turn_min 0.01, margin 2) removed turning altogether: the runs are one-line runs, which is why they match `ag_axis_*` to two digits. **The reading in section 35 ("controllable turning adds nothing") is wrong; turning was not exercised.** H39 is untested, not unsupported; H41 (the wrecked swarm was a threshold problem) stands: the swarm is healthy exactly when it does not turn.
- Likely reason it never learns to turn (from reading `embody.py`, not tested): a turn is `stop_gradient`ed and replaces the move by a fixed side shift, so no gradient tells the cell that turning would help; the training signal can only teach it to avoid turns (the old rule jittered the swarm apart), not when to use one. A trained turn policy probably needs a differentiable relaxation of the turn or a reward for it; neither exists yet.
- Queued `ag_turn3_mlph_scales2` (turn_min 0.004 = 4.6 mm, margin 1.3) to see whether a lower threshold gives some turning without wrecking the swarm; turn rate to be measured when it lands. One run; a negative result would not end the question, for the gradient reason above.

## 37. Note 03:00: a fine-tune of the two-radius cell stopped early
`scales2__ft_all_damage_10k` (fine-tune of `scales2` on wipe, cut, drag and clump damage, no pool age cap) stopped at 3,992 of 10,000 iterations after 40 blow-up recoveries, ending at score 0.150 (control 0.139). It is the old recipe without `pool_max_age` (H23); the replicated lizard result for this cell is `pa8_scales2_all_damage` (0.081 to 0.085, three seeds). Not re-run; it adds nothing beyond what H23 already explains, but I have not checked that the cause is the missing cap rather than something else.

## 38. Note 03:50: the thinned-ring test (H42) is built and queued, not run
`lab/cells_mlphthin.py` (cell kind `mlphthin`, option `ring_vis`) is the random-heading plain cell whose second ring sees each pair beyond eps with probability `ring_vis` (default 0.25, from the line-of-sight count in section 34); the hash that decides visibility is symmetric in the pair and fixed while the swarm is still. It is an approximation of occlusion (real shadowing is correlated), not a model of it. `lab/test_thin.py` passes 7 of 7 checks (full visibility equals `npa.perceive` to 1.5e-6; about 25% of far pairs visible, measured 0.246; symmetric; finite step and gradients). Queued: `ag_thin_axis_mlph_scales2` (one-line body, same as `ag_axis_mlph_scales2` otherwise; compare cut 0/0.02/0.03, noise 0/0.07/0.19) and `ag_thin_free_scales2` (ideal body, same as `ag_eps7_scales2_warm1` otherwise; compare noise -0.01/0.07/0.21, cut 0/0.05/0.10). Nothing to read until they finish (about 1 to 3 hours).

## 39. Update 04:08: 20,000 iterations help a little at 120 neighbours (free motion, equivariant cell)
`ag_geo_warm1_equiv_20k_L` (20,000 iterations, finished) against the three 6,000-iteration seeds of `ag_geo_warm1_equiv`; gains over do-nothing, small/medium/large, mean over held-out shapes:
| run | control | clump | noise | drag | cut |
| --- | --- | --- | --- | --- | --- |
| 6k seeds 0/1/2 | 0.025/0.030/0.027 | 0.04-0.06 / 0.14-0.15 / 0.41-0.42 | 0.03-0.05 / 0.14-0.16 / 0.26-0.27 | 0.01-0.03 / 0.03-0.04 / 0.01-0.02 | 0.00-0.01 / 0.05 / 0.09-0.10 |
| 20k (one seed) | 0.014 | 0.06/0.16/0.43 | 0.06/0.17/0.28 | 0.04/0.06/0.05 | 0.01/0.06/0.12 |
The largest changes are drag at medium and large (0.03/0.02 to 0.06/0.05) and cut at large (0.09-0.10 to 0.12), plus a lower undamaged control (0.014). Clump and noise move by 0.01 to 0.02. One seed of the long run, so the drag gain is suggestive, not established; more iterations do not change the picture (cut and tear stay the weakest kinds, still far from full repair).

## 40. Update 04:55: the four-shape transfer test finally ran: 4x the iterations help a lot, but one plain network is still 28 to 38% worse on three of four shapes (H25)
`ms4_base_40k_L` (one plain 66-128-18 network trained on lizard, star, bone and butterfly together, 40,000 iterations, 197 minutes, no blow-ups; the first run to complete after the runner's time limit was fixed, section 29). Per-shape score (lower is better; control loss in brackets) against the single-shape specialists trained for 10,000 iterations:
| shape | `ms4_base_40k_L` | specialist | ratio | `ms4_base` (10k, for scale) |
| --- | --- | --- | --- | --- |
| lizard | 0.306 (0.258) | `base_10k` 0.221 (0.169) | 1.38 | 0.447 |
| star | 0.308 (0.226) | `spec_base_star_10k` 0.232 (0.153) | 1.33 | 0.448 |
| bone | 0.278 (0.184) | `spec_base_bone_10k` 0.218 (0.131) | 1.28 | 0.353 |
| butterfly | 0.325 (0.216) | `spec_base_butterfly_10k` 0.304 (0.220) | 1.07 | 0.405 |
- **H25 (partly):** the criterion was within 25% on each shape; only the butterfly meets it. The other three are 28 to 38% worse. Quadrupling the iterations cut the overall score from 0.413 to 0.304 and the damage excess is small (overall 0.083, mostly in cut and drag), so more training is helping, but one network does not yet match specialists at four times the per-shape budget.
- **Caveats (not checked):** the multi-shape and single-shape evaluators may use different damage levels and shape orientation, so the ratios are indicative; one seed; the specialists were trained for 10,000 iterations, not 40,000, so this is not an iteration-matched comparison for the specialists.
- Still running: the two-radius and memory versions of the same test (`ms4_scales2_pa8_L`, `ms4_mem8_pa8_L`, `ms4_scales2_pa4_L`).

## 41. Update 05:25: the two-radius and memory recipes make one network work on all four shapes (H25, H29)
The three remaining four-shape runs finished (40,000 iterations each, no blow-ups, one seed each). Per-shape score, with the undamaged control in brackets (lower is better), and the overall damage excess:
| run | lizard | star | bone | butterfly | overall score | excess |
| --- | --- | --- | --- | --- | --- | --- |
| `ms4_base_40k_L` (plain, section 40) | 0.306 (0.258) | 0.308 (0.226) | 0.278 (0.184) | 0.325 (0.216) | 0.304 | 0.083 |
| `ms4_scales2_pa8_L` (two radii, damage training, pool age cap 8) | 0.135 (0.135) | 0.098 (0.099) | 0.114 (0.113) | 0.169 (0.174) | 0.129 | -0.001 |
| `ms4_mem8_pa8_L` (gated cell with 8 memory channels, same recipe) | 0.124 (0.127) | 0.085 (0.082) | 0.098 (0.093) | 0.160 (0.160) | 0.117 | 0.001 |
| `ms4_scales2_pa4_L` (two radii, cap 4) | 0.141 (0.120) | 0.091 (0.090) | 0.124 (0.126) | 0.186 (0.186) | 0.136 | 0.005 |
- **Damage excess is about zero for every damage kind** (largest 0.021, wipe, `pa4`): with the two-radius or memory cell and damage training, each of the four shapes returns to its own undamaged level after a wipe, cut, drag, clump, shuffle or noise. The remaining loss is the growth quality (control 0.09 to 0.19), not repair. That matches the single-shape lizard result for this recipe (`pa8_scales2_all_damage`: score 0.085, control 0.084, excess 0.001).
- **Comparison with specialists, with a confound:** these scores are 0.4 to 0.6 times the plain 10,000-iteration specialists (`spec_base_*` 0.218 to 0.304; `base_10k` 0.221), but the specialists use the plain recipe, so this mostly shows the recipe, not multi-shape vs single-shape. The same-recipe lizard specialist (`pa8_scales2_all_damage`, 0.085) is 1.5 to 1.7 times better than the multi-shape lizard score (0.124 to 0.141); no same-recipe specialists for the other three shapes were compared. Also unchecked: whether the multi-shape and single-shape evaluators share damage levels.
- **Memory cell:** `ms4_mem8_pa8_L` is the best here (0.117), but the same cell diverged on a second seed on the lizard (section 30, H40). One seed here, so it is not shown to be stable.
- **What it does not show:** these are shapes the network was trained on (four known shapes, ideal dots, free motion); it says nothing about unseen shapes (that is the `ag_*` track, where repair of cuts and tears is still weak) or the robot body.

## 42. Update 05:40: the lower-threshold turn run also never turns; H39 is still untested
- **`ag_turn3_mlph_scales2`** (turn_min 0.004 = 4.6 mm, margin 1.3, two rings): undamaged control 0.045; gains clump 0.04/0.13/0.39, noise -0.01/0.07/0.20, drag 0.00/0.02/0.01, cut 0.00/0.02/0.03, identical to `ag_axis_mlph_scales2` (one-line). Measured turn rate with `lab/turn_rate.py`: **0 turns in 118,374,400 bot-ticks**. Even with the easier threshold the cell never proposes a turn, so the result is again a one-line result.
- **`ag_turn2_equiv_random`** (equivariant cell, turn_min 0.01, margin 2): control 0.065; clump 0.02/0.11/0.37, noise -0.01/0.05/0.17, drag -0.01/0.01/0.01, cut -0.01/0.01/0.02, like the one-line equivariant runs (turn rate not measured; I expect 0 but did not check).
- **What this supports:** with the old rule (threshold about 0) the cell turns 14.5% of the time and wrecks the swarm; with any threshold I tried it turns never. There was no setting in between that I saw. That fits the explanation in section 36 (the turn is cut off from the gradient, so training can only push the cell away from turning) but does not prove it; I have not tested a differentiable turn.
- **Next, not yet built:** a differentiable turn relaxation (blend the stride and the side shift by a smooth turn probability computed from the proposal) or an explicit turn reward. Until one of them is tried, H39 (do controllable turns recover cut and tear repair) remains unanswered; the one-line results (cut 0/0.02/0.03) are what the body model gives without a usable turn.

## 43. Update 06:00: the second-ring gains survive random thinning to a quarter of the ring (H42)
Per-severity gains over do-nothing (small/medium/large, mean over held-out shapes), one seed each; the thinned runs use `cells_mlphthin` with `ring_vis` 0.25 (each pair beyond eps visible with probability 0.25, symmetric hash, section 38):
| run | control | clump | noise | drag | cut |
| --- | --- | --- | --- | --- | --- |
| `ag_thin_axis_mlph_scales2` (one-line body, ring 2 thinned) | 0.052 | 0.03/0.12/0.38 | 0.00/0.07/0.19 | 0.00/0.01/0.01 | 0.00/0.02/0.03 |
| `ag_axis_mlph_scales2` (same, full ring 2) | 0.048 | 0.03/0.13/0.39 | 0.00/0.07/0.19 | 0.00/0.02/0.01 | 0.00/0.02/0.03 |
| `ag_axis_mlph_random` (one-line, one ring) | 0.061 | 0.01/0.10/0.36 | -0.02/0.04/0.15 | 0.00/0.01/0.01 | -0.01/0.01/0.02 |
| `ag_thin_free_scales2` (ideal body, ring 2 thinned) | 0.057 | 0.04/0.14/0.40 | -0.02/0.07/0.20 | 0.01/0.03/0.02 | -0.01/0.04/0.08 |
| `ag_eps7_scales2_warm1` (same, full ring 2) | 0.029 | 0.03/0.13/0.40 | -0.01/0.07/0.21 | 0.02/0.04/0.03 | 0.00/0.05/0.10 |
- **H42 supported, with limits.** Thinning the second ring to about a quarter (the visible fraction found in section 34) leaves the one-line numbers unchanged to two digits and costs the free-motion run only a little (cut 0.04/0.08 against 0.05/0.10, drag 0.03/0.02 against 0.04/0.03; control 0.057 against 0.029). The noise gain of the two-ring cells over one ring (0.04/0.15 to 0.07/0.19 to 0.20) is kept. So the two-ring benefit does not need the full ring; about five visible bots in ring 2 were enough here.
- **Caveats:** one seed each; the thinning is random and independent per pair, while real occlusion is correlated (a bot inside a clump sees little at range, one at the edge sees more), so this does not show the cell copes with correlated shadowing; the visibility rate came from a hard-disc count, not the real swarm; a radio relay would be a different hardware assumption. The one-line cut and tear repair is still the weak point (cut 0/0.02/0.03).

## 44. Update 06:20: a straight-through turn gradient is built and two runs are queued (H39, H43)
- **Idea (a design guess, not a result):** the turn runs never turn (sections 36 and 42), and a plausible reason is that a turn and the one-line projection both cut the gradient from the executed move to the cell's proposal along the other line. `embody.py` now has `turn_st` (default 0, off): the executed move is unchanged, but the backward pass sees the capped full 2-D proposal as if the bot had moved that way (`d + turn_st * (d_free - stop_gradient(d_free))`).
- **Checked (`lab/test_turn_rule.py`):** the forward move is identical with and without it; the gradient of the executed move with respect to the other-line proposal component goes from 0 to 0.98 for a bot that strides and from 0 to 0.64 for a bot that turns. That shows the mechanism works as intended, not that training will use it.
- **Queued:** `ag_turnst_mlph_scales2` (turn_min 0.004, margin 1.3) and `ag_turnst2_mlph_scales2` (turn_min 0.01, margin 2), both with `turn_st` 1.0, two rings, the one-line body, otherwise like `ag_turn3_mlph_scales2` (control 0.045, zero turns). Read: the undamaged control, the turn rate (`lab/turn_rate.py`), and cut and tear gains against `ag_axis_mlph_scales2` (cut 0/0.02/0.03, drag 0/0.02/0.01). A straight-through estimator can also just add noise; if the rate stays at 0 or the swarm is wrecked, the explanation in section 36 is wrong or incomplete.

## 45. Update 08:10: with the straight-through gradient the cell turns (2% of bot-ticks), but the undamaged swarm degrades and repair does not improve (H39, H43)
`ag_turnst2_mlph_scales2` (turn_min 0.01, margin 2, `turn_st` 1.0, two rings, one-line body, 6,000 iterations, no blow-ups; one seed). The identical run without `turn_st` (`ag_turn2_mlph_scales2`) never turned.
- **Turn rate (`lab/turn_rate.py`):** 2,306,838 turns in 118,374,400 bot-ticks, 0.0195 per bot-tick (the old rule that wrecked the swarm: 0.145; no `turn_st`: 0). Every bot is counted each tick but only about half update, so about 3.9% of updating ticks turn.
- **Quality:** undamaged control 0.213 (0.045 without `turn_st`; 0.29 to 0.36 with the old rule). Gains over do-nothing, small/medium/large: clump -0.08/+0.02/+0.29, noise -0.13/-0.02/+0.12, drag -0.09/-0.04/-0.02, cut -0.11/-0.03/+0.02. Worse than the one-line run (cut 0/0.02/0.03, drag 0/0.02/0.01) on everything except large clump and noise, which are lower too (0.39 and 0.19 before).
- **Reading (not proven):** H43 holds in the narrow sense that the straight-through gradient makes a trained cell turn at all (the earlier runs show 0 turns at the same thresholds). The turns it learns are not useful here: each is a 28 mm side shift (0.57 of the sensing radius) and the undamaged swarm drifts apart (control 0.213). It could be that turns genuinely cost more than they give in this body model, or that the straight-through estimator pushes the cell toward proposals the body does not execute; I have not separated the two. H39 (a controllable turn recovers cut and tear repair) is still unanswered, with one negative data point under one estimator and one threshold.
- `ag_turnst_mlph_scales2` (turn_min 0.004, margin 1.3) is still training; read it before drawing more.

## 46. Update 14:40: the lower-threshold straight-through turn run is also wrecked; the monitoring loop had stalled
- **`ag_turnst_mlph_scales2`** (turn_min 0.004, margin 1.3, `turn_st` 1.0; 6,000 iterations; one seed): undamaged control 0.297; gains small/medium/large clump -0.15/-0.05/+0.22, noise -0.21/-0.09/+0.07, drag -0.16/-0.10/-0.07, cut -0.19/-0.10/-0.02. That is as bad as the old-rule turn runs (control 0.29 to 0.36), so with the easier threshold the cell turns a lot and the swarm falls apart. Turn rate not measured; this is inferred from the control loss. With section 45 (`turn_st` at 0.01/2: control 0.213, turn rate 1.95% of bot-ticks) the pattern is: no turning gives one-line results, any turning that training finds wrecks the undamaged swarm under this body model. H39 is still unanswered.
- **Process failure (mine):** the 20-minute wakeup loop stopped firing after 08:28 and the watchdog tmux session had been dead since about 01:38 (its log stops there); I read `tmux ls` output at each check and did not notice the watchdog line was missing from about 01:50 on, so the dashboard was not being redeployed by the watchdog for about 13 hours (my own manual deploys covered some of it). Fleet and the GPU runs were unaffected (cycle 1334 at 14:37). Watchdog restarted at 14:37. The recurring check now has to verify BOTH sessions by name, not by reading the list.
