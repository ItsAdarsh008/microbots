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
- **H29 (three seeds):** `pa8_scales2_all_damage` scores 0.085, 0.081 (_s1) and 0.085 (_s2); control 0.078 to 0.084; hard-damage repair 0.98 to 0.99. The spread, 0.004, is far below the +-0.025 seed spread of the earlier runs. Confirmed on the lizard for this cell; `pa8_mem8_all_damage_s1` is still running.
- **H33 (one run):** `ag_axis_equiv_random` (shape-agnostic, unseen shapes; body model with one-line motion per bot, measured drift, no minimum stride, 30% packet loss, sensing noise, each bot with its own random heading; equivariant cell; about 7 neighbours): undamaged control 0.077 (the old assumed body model: 0.253 for the same cell; free motion: 0.025), score 0.163 against a do-nothing floor of 0.181. Gain over doing nothing at small/medium/large severity:
clump 0.01/0.10/0.36 (free motion 0.06/0.15/0.42), noise -0.02/0.05/0.16 (0.05/0.16/0.27), drag -0.01/0.01/0.01, cut -0.02/0.01/0.03. So with the measured one-line motion the swarm keeps its shape and still repairs clump and noise damage at
medium and large severity, but less than with free motion, tears and cuts barely at all, and it is slightly worse than doing nothing at small damage. One run; `ag_axis_mlph_random`, `ag_axis_mlph_scales2` and the two `ag_axis_ideal_*` runs are still to come.

## 25. Update 22:35: one-line motion costs repair, not noise (H33, H36; three runs)
Gain over doing nothing at small/medium/large severity, unseen shapes, body model with one-line motion (measured drift, no minimum stride), about 7 neighbours: `ag_axis_ideal_mlph` (no sensing noise or packet loss) clump 0.01/0.10/0.36, noise -0.03/0.01/0.11, drag -0.01/0.01/0.01, cut -0.01/0.01/0.02, control 0.064;
`ag_axis_mlph_random` (noise, packet loss, random headings) clump 0.01/0.10/0.36, noise -0.02/0.04/0.15, drag 0.00/0.01/0.01, cut -0.01/0.01/0.02, control 0.061; `ag_axis_equiv_random` as in section 24 (control 0.077). Free motion (`ag_geo_warm1_equiv`):
clump 0.06/0.15/0.42, noise 0.05/0.16/0.27, drag 0.03/0.04/0.02, cut 0.01/0.05/0.09, control 0.025. So the one-line constraint costs a third to a half of the noise and cut repair and a little of the clump repair; sensing noise and
packet loss do not (the noise-free run is no better than the noisy ones) and the cell makes no difference. Three runs, one seed each. A bracket for a bot that can turn (H36) is queued.
