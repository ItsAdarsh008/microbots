# Diffusion / flow matching for swarms: from noise to shape

**Question.** Can particles that start as noise end up as a structured shape if each one follows a learned flow-matching velocity field, and how much of that can be made *local* (what a real swarm could do)?

**Why flow matching fits.** Sampling a flow-matching model *is* a swarm dynamic: N particles start at random positions and each takes small steps along a velocity field. The step count is the physical time. A teammate trains plain MLP policies from a teacher on the same shape set; this folder is the generative side, so the two can be compared on the same shapes.

Earlier work in this repo (branch `phase3-npa-particle-nca`) trained Neural Particle Automata on **single** shapes (about 400 runs, ~1,000 particles). The new part here is **thousands of shapes** and **held-out shapes**.

## The ladder (each rung needs the one before it to work)

| Rung | What | Success | Falsified by |
| --- | --- | --- | --- |
| **S1a** | Free 2D particles, **global** attention, unconditional | Samples are single connected shapes with the right statistics, not copies of training shapes | Blobs, fragments, or memorised training shapes |
| **S1b** | Same, target map given (each particle reads the map at its own position) | Held-out and held-out-family shapes formed to within ~1.5x of in-distribution error; beats the no-learning baseline | Held-out error far above train, or no better than the classical baseline |
| **S1c** | **Local** denoiser: a particle sees only neighbours within radius r | Within ~20% of S1b; a clear knee in error vs r | Error stays high below the feature scale, or wrong topology (holes, arms) |
| **S1d** | No global clock (time-free), async updates, variable N, damage | Degrades < 25% | Collapse without `t` |
| **S1e** | Training-set size: 10^2, 10^3, 10^4 shapes | Held-out error saturates | Still falling at 10^4: broaden the generator |
| **S2** | Our 2D microbots: finite bodies, speed cap, motion noise, physical start | Coverage > 85% with < 1% overlaps | Loses to the teacher-MLP at equal local info |
| **S3** | 3D in simulation | Same metrics, gravity respected | Neighbour count explodes, support not learnable |

## What the literature says (details in `literature_notes/`)

Three reviews, about 12k words, mostly checked against arXiv abstract pages; each file tags every entry **[verified]**, **[snippet only]** or **[from memory]**, and lists the citations it is unsure of. Read those tags before quoting anything.

1. **Nobody has done this exact thing.** No paper found uses diffusion or flow matching for a swarm of *locally sensing* agents self-assembling into a shape. Multi-agent diffusion work (MADiff, SwarmDiff, Flow-Opt, SID) generates trajectories or policies, mostly with central or global-attention denoisers. Also no diffusion work found on modular / self-reconfigurable robots, and no strictly radius-limited flow-matching denoiser on unordered point sets. Good for novelty, bad for baselines.
2. **Closest template: Gen-Swarms** (arXiv 2408.15899): flow matching on 2,048-point clouds with collision avoidance (ORCA) inside the sampler; shape quality plateaus near 25 steps; simulation only.
3. **The baseline that matters is non-learned: mean-shift shape assembly** (Sun et al., Nat. Commun. 2023). Each robot stores a distance-transform map of the shape; 50 real robots, up to 1,024 simulated, with repair. Our `classical` baseline (slide to the shape, then spread by local repulsion) is a simplified cousin of it. A learned method has to win on something that baseline can't do.
4. **Few-step generation is realistic**: pi0 uses 10 Euler steps, Diffusion Policy 10 DDIM steps; one-step variants exist (MeanFlow, shortcut), only with global networks.
5. **Locality is plausible but untested for point sets.** For images, denoisers behave like local patch denoisers (Kamb & Ganguli 2412.20292; Niedoba et al.; Lukoianov et al.). Kamb & Ganguli's theory predicts a local denoiser gives locally consistent patches but can get **global topology** wrong. We test exactly that at S1c (hole count, connectivity).
6. **A clock may not be needed** (Sun et al. "Is noise conditioning necessary", Equilibrium Matching, Helbling 2026), but the argument needs high dimension and a local denoiser sees few neighbours. Highest-risk assumption; tested by the `notime` run.
7. **No paper quantifies how many training shapes are needed** for generalisation in assembly. Blocks Assemble (ICML 2022) generalises zero-shot with ~1e9 RL steps. A 10^2/10^3/10^4 curve (S1e) would be new.
8. **Flow-matching paths are straight lines from noise, so they cross through each other.** Hard collision-freedom holds only in the idealised OT limit; the physical projects use projection inside the sampler or a reactive filter.

### Ways this can look good in simulation and not be buildable

Global clock `t`; global attention; shape code broadcast every step; unbounded speed; starting from a Gaussian over the whole arena instead of a physical pile; particle identity; collisions; fixed N; absolute coordinates (= secretly having localisation); synchronous perfect updates. Stage 1 **knowingly** uses a global clock, absolute coordinates, N = 256 fixed and Gaussian starts; each is lifted at a later rung or in a named ablation.

## Experiment design (Stage 1)

- **Data** (`shapes.py`): 10,000 procedural shapes from 6 families (blobs, disc unions, strokes, convex polygons, holey, CSG), each with 256 evenly spaced particles (farthest-point sampling), a 64x64 occupancy map and signed-distance map. **Two families are held out** (stars, rings/C-shapes) and 12 hand-made emoji shapes (the NPA shapes) are a third test set. Validation: 512 unseen shapes from the same families.
- **Method** (`train.py`): flow matching, noise N(0, 0.7²), each noise set paired to its shape set by **exact optimal-transport matching inside the set**, 30k iterations, batch 32 sets, EMA. Rotation/flip augmentation.
- **Model** (`model.py`): permutation-equivariant set transformer with relative-position attention bias. `radius=None` is global; `radius=r` masks attention to neighbours within r, and `layers` = message rounds per step. The map is read **at the particle's own position** (signed distance, its gradient, occupancy at three blur scales).
- **Metrics** (`evalm.py`, in units of each shape's own particle spacing): `w2` (optimal-matching RMS error to the reference points), `inside`, `cover`, `coll` (collision fraction), `nn` (spacing). `skill = 1 - w2/w2_noise`.
- **Rows in every table**: `noise` (chance), `project` (one-shot projection), `classical` (no learning, with a map), `ref` (the data), and the model at K = 4…64 steps.
- **Controls**: shuffled-label training run; wrong-map test (feed another shape's map: error against the fed map must stay low, against the other shape high); real-vs-generated descriptor comparison (connectedness, holes, radius of gyration); best-IoU-with-training-set (memorisation) for generated vs real validation shapes.

## Layout

```
shapes.py   data generator            model.py   denoiser (global or radius-limited)
train.py    flow-matching trainer     evalm.py   metrics, baselines, controls, figures, GIF
worker.py   per-machine job queue     configs/   one JSON per job
runs/<name>/{result.json, samples.png, traj.gif, log.txt}   (on the GPU machines; collected into results/)
```

Run on the GPU boxes: `python shapes.py` once, then drop configs into `~/diffusion/queue/pending/` and start `python worker.py`.

## Results so far (see JOURNAL.md for every entry, caveats and one correction)

Error is optimal-matching RMS distance to a reference sample of the shape, in particle spacings (floor about 0.7 to 0.8: two independent samples of the same shape). No-learning baseline with the map: 1.45 familiar, 1.31 held-out family, 1.60 emoji. Chance: 9.7 / 7.1 / 8.8. Collisions: baseline 9%.

| Run | Familiar | Held-out family | Emoji | Connected | Collisions | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| global attention, with map | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 | 1.00 | 0.03 | 2 |
| local r=0.25, 4 rounds | 1.27, 1.29 | 1.63, 1.65 | 1.65, 1.68 | 0.95 | 0.02 | 2 |
| local r=0.40, 4 rounds | 1.30, 1.34 | 1.29, 1.25 | 1.51, 1.52 | 0.97 | 0.04 | 2 |
| local r=0.60, 4 rounds | 1.31 | 1.16 | 1.45 | 0.98 | 0.06 | 1 |
| local r=0.15, 4 rounds | 1.33 | 1.70 | 1.71 | 0.75 | 0.02 | 1 |
| local r=0.25, no clock | 1.65 | 1.86 | 1.90 | 0.07 | 0.05 | 1 |
| local r=0.40, no clock | 1.44, 1.49 | 1.55, 1.55 | 1.65, 1.69 | 0.61, 0.59 | 0.09, 0.13 | 2 |
| global, no clock | 1.35 | 1.20 | 1.45 | 0.99 | 0.05 | 1 |
| global, told only an 8x8 map (64 numbers) | 1.38 | 1.23 | 1.41 | 1.00 | 0.03 | 1 |
| global, told only a 4x4 map (16 numbers) | 1.49 | 1.58 | 1.71 | 1.00 | 0.04 | 1 |
| global, 10 / 30 / 100 / 300 / 1,000 shapes | 1.22 / 1.27 / 1.35 / 1.33 / 1.26 | 1.21 / 0.99 / 1.01, 1.17 / 1.18 / 1.11 | 1.55 / 1.32 / 1.33, 1.34 / 1.26 / 1.32 | 1.00 | 0.01 to 0.05 | 1 each (100: 2) |
| control: shuffled labels | 3.07 | 3.19 | 3.51 | 0.98 | 0.16 | 1 |
| unconditional global (no target) | n/a | n/a | n/a | 0.996 | | 1 |
| unconditional local r=0.25 | n/a | n/a | n/a | 0.68 | | 1 |

Read, with what is replicated (2 seeds) marked:
- A map-conditioned global model forms unseen shapes better than the no-learning baseline, with a third of its collisions. (2 seeds)
- A local denoiser matches global on familiar shapes; on unseen shape families it needs a sensing radius of about 0.4 to 0.6 (5 to 8 particle spacings); r=0.25 falls behind the baseline and bigger models or longer training help by only about 0.1. Rounds help at r=0.4 (2 seeds). (2 seeds)
- Local models need a clock (2 seeds); the global model does not (1 seed). A noisy-clock training run removes the collision problem under clock errors (1 seed).
- Number of training shapes (10 to 10,000) makes no visible difference for the map-conditioned task, within a seed spread of about 0.15 (mostly 1 seed per point).
- The swarm needs very little broadcast: an 8x8 occupancy grid (64 numbers) costs about 0.1 in error for the global model (2 seeds) and the r=0.4 local model (2 seeds); a 4x4 grid is too coarse.
- On thin branching shapes (trees, spiders) global and local r=0.4 both beat the no-learning baseline on held-out families (2 seeds each); spiders are the hardest family for every model.
- Without a map, a global model makes plausible compact shapes (2 seeds at 60k iterations); a local-only one fragments at r=0.25 (1 seed) and mostly holds together at r=0.4 (2 seeds).
- Physical constraints (1 seed): a hard-core spacing rule removes all collisions for 0.01 of error; a speed cap costs 0.2 from scattered noise but nothing from a pile. A well-spaced pile start gives zero collisions but forms held-out shapes 0.4 to 0.6 worse than scattered noise; against the no-learning baseline from the same pile (2.0 to 2.4 after 1500 steps) the learned model is better after 8 steps.
- Self-repair is not free: rewinding the clock does not refill a cut chunk, and a model that keeps stepping without a clock collapses into a clump. Repair-trained models are running.
- From a compact pile at a random offset, held-out error is 0.5 worse than from scattered noise (2 seeds), but almost all of that is distance: a pile centred on the shape forms shapes as well as scattered noise does (1 seed), and a two-phase swarm (rigid travel to the target site, then local formation) gets 1.53 (global) and 1.80 (local r=0.6) on held-out thin shapes, against 1.93 and 2.11 for a model trained on the whole move (1 seed per model for the two-phase test).
- Training with damaged starts in the mix gives repair (cover 0.97 to 0.99 after a cut chunk; 2 seeds) and, from a pile, also 0.1 better formation (2 seeds). It raises collisions (0.15 to 0.21 from scattered noise), which a hard-core spacing rule removes.

## Stage 1 reference swarm (journal tick 23)

Local radius 0.6, an 8x8 target grid (64 numbers) as the only broadcast, noisy-clock training, repair batches, centred-pile start, thin-shape data, 2 seeds. Held-out-family error 1.67 and 1.61 (no-learning baseline 1.96), collisions 0.00, repair cover 0.97 after a cut chunk, tolerant of clock jitter, speed caps and 128 to 512 bots; motion noise gives 10% collisions. The shared-global-state experiments that follow (`GLOBAL_STATE_PLAN.md`) start from the r=0.25 local model, where the gap to the all-to-all model is largest.

## Try the models (demo)

`demo/serve.py` serves 12 trained models behind a small web page: pick a model, pick or draw a target shape, choose the start (scattered, centred pile, offset pile with rigid travel), steps, bot count, clock error, motion noise and the hard-core spacing rule, then cut a chunk out and press repair. Models run on one GPU box (about 0.1 to 0.2 s per run; CPU also works, the models have 2.5M to 3.6M parameters). On the box: `python demo/serve.py --port 8000` (binds to localhost); from your machine: `ssh -L 8000:localhost:8000 ubuntu@<box>` and open http://localhost:8000. `demo/export.py` packs the EMA weights of runs into `demo_models/` (about 10 to 15 MB each; not in git).

## CORRECTION (journal tick 26): shape quality

Everything above that says a learned model "beats the no-learning baseline" is **withdrawn as a statement about how the shapes look**. It rests on `w2`, the distance to one random sample of the shape, which does not track visual quality. Rendered side by side, a simple rule (project each bot onto the shape, spread by local repulsion) draws crisp, evenly spaced shapes, while the trained models give fuzzy outlines with missing thin parts. By footprint IoU (ceiling 0.78 to 0.83) the learned models are level with the rule (0.59 to 0.70 against 0.61 to 0.67), and **a learned flow followed by 60 steps of the rule is better than both (0.69 to 0.76)**. Comparisons *among learned variants* are internally consistent in `w2` but not yet checked visually. IoU is now in every evaluation, and new runs are judged on pictures as well as numbers.

## Where things stand (journal tick 32, 2026-10-30)

- **Geometric training losses are what made the learned models good.** Penalising bots that would end outside the shape (and, secondarily, crowding) raises footprint IoU from about 0.58 / 0.61 / 0.67 (emoji / held-out / familiar) to 0.70 / 0.71 / 0.75 for a local radius-0.6 model, against 0.63 / 0.61 / 0.67 for the hand-written rule and a ceiling of 0.79 / 0.78 / 0.83 (16 steps, thin-shape data; weight 20; replicated on 2 seeds for the local model). The "stay inside" term carries the gain. Letters (a font test set nobody trained on) are readable, and IoU beats the rule on thick (0.71 vs 0.59) and thin (0.64 vs 0.56) letters.
- **Known regression:** the same losses make wide hand-drawn T shapes worse for the global model (cover 1.00 to 0.56); a coverage penalty is being tested.
- Older tables above that are in `w2` should be read with the correction at the top of the journal (tick 26): they compare learned variants with each other, not shape quality against the hand-written rule.
- The demo (`demo/`) lists the best models, 52 letters and the emoji upright.

### Update (journal tick 36)

The geometric-loss weight keeps helping up to about 100: local radius 0.6 reaches footprint IoU 0.75 / 0.75 / 0.79 (emoji / held-out / familiar) against 0.58 / 0.61 / 0.67 without it and 0.63 / 0.61 / 0.67 for the hand-written rule (ceiling 0.79 / 0.78 / 0.83). On letters the weight-100 local model scores 0.76 (thick) and 0.70 (thin) against the rule's 0.59 and 0.56. The T-shape regression seen at weight 20 disappears at weight 100. With these losses a small sensing radius is nearly as good (radius 0.25 is about 0.03 below 0.6, 2 seeds at weight 20). Adding a union-of-rectangles shape family to the training data changed nothing. Weight 100 is still one seed per model; replicates are running.

### Update (journal tick 40): a candidate reference swarm for Stage 2

One local network with a sensing radius of about 4 to 6.6 particle spacings (r=0.25 to 0.4), only an 8x8 (64-number) map broadcast, noisy-clock training, repair batches, a centred pile and the geometric losses at weight 100 forms thin shapes and letters recognisably (held-out footprint IoU 0.68 to 0.69 against 0.74 for the plain best model), with zero collisions; it tolerates clock errors, 128 to 512 bots and a speed cap; the rigid travel phase works; motion noise gives 11% collisions (the hard-core spacing rule removes collisions in all earlier tests); repair after a cut chunk is weaker at the small radius (cover 0.83 to 0.91). One seed so far; second seeds are running. The 64-number map costs the snowflake. No real bot dynamics yet: Stage 2 needs the physical numbers.

### CORRECTION (journal tick 44): sensing radius in bot spacings

Earlier sections convert the sensing radius to spacings wrongly (about 2.5 for r=0.25, about 4 for r=0.4, about 6 for r=0.6). Measured on the data (mean spacing 0.061 world units; it varies from 0.04 to 0.08 with the shape): **r=0.25 is 4.1 spacings, r=0.4 is 6.6, r=0.6 is 9.8.** The v1 bot's arm reach (81 mm over a 34.5 mm touching distance) is about **2.35 spacings**, so even the smallest model senses about 1.75 times farther than the real arm reaches. Models at r=0.14 (about 2.3 spacings) and r=0.10 are training. Statements such as "radius 0.25 matches the arm reach" in the journal are retracted.
