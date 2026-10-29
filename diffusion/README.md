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

## Results so far (see JOURNAL.md for every entry and the caveats)

Error is optimal-matching RMS distance to a reference sample of the shape, in particle spacings (floor about 0.7 to 0.8: two independent samples of the same shape). No-learning baseline with the map: 1.45 familiar, 1.31 held-out family, 1.60 emoji. Chance: 9.7 / 7.1 / 8.8.

| Run | Familiar | Held-out family | Emoji | Connected | Seeds |
| --- | --- | --- | --- | --- | --- |
| global attention, with map | 1.24, 1.33 | 1.14, 1.12 | 1.40, 1.41 | 1.00 | 2 |
| local r=0.25, 2 rounds | 1.29 | 1.64 | 1.68 | 0.92 | 1 |
| local r=0.25, 4 rounds | 1.27 | 1.63 | 1.65 | 0.95 | 1 |
| local r=0.15, 4 rounds | 1.33 | 1.70 | 1.71 | 0.75 | 1 |
| local r=0.25, 4 rounds, no clock | 1.65 | 1.86 | 1.90 | 0.07 | 1 |
| control: shuffled labels | 3.07 | 3.19 | 3.51 | 0.98 | 1 |
| unconditional global | n/a (no target) | n/a | n/a | 0.996 | 1 |

Read: a map-conditioned global model forms held-out shapes better than the no-learning baseline and with a third of its collisions (2 seeds); locality costs nothing on familiar shapes and about 0.5 spacings on unseen ones (1 seed each; replicate running); removing the clock breaks connectivity (1 seed); the unconditional global model makes plausible compact shapes with too many holes.
