# Stage 1 summary: particles that form shapes by flow matching (state of 2026-10-30)

Written for someone who has not followed the journal. Every number is from `JOURNAL.md` (which also holds the corrections). "2 seeds" = replicated; "1 seed" = not yet a finding. Quality is judged by **footprint IoU** (overlap of the bots' footprint with the target; ceiling 0.78 to 0.83) and by **looking at rendered pictures**, never by the earlier `w2` distance alone.

## What was built
- 2D, 256 particles, a procedural shape set (blobs, strokes, polygons, holes, trees, rectangle unions; stars, rings and spiders held out), 12 emoji and 52 letters as pure test shapes.
- Flow matching with an exact optimal-transport pairing inside each shape; a permutation-equivariant set transformer; sensing is either all-to-all or restricted to a radius (a graph network on a radius graph). Each bot reads the target map at its own position.
- Tools: `train.py` (jobs from `configs/`), `evalm.py`, `robust.py`, `physical.py`, `repair.py`, `travel_eval.py`, `demo/` (a web demo that serves the models).

## What works (and how sure)
1. **A learned local swarm forms recognisable shapes and letters and beats the hand-written rule** (project onto the shape, spread by repulsion), once trained with geometric losses that penalise bots ending outside the shape (weight 100; 2 seeds). Held-out IoU 0.74 against 0.61 for the rule; letters 0.76 (thick) and 0.70 (thin) against 0.59 and 0.56. By eye: clean letters, anchor, ring, wrench, snowflake; fuzzier outlines than the rule; the butterfly's far wing and the key's ring end are the weak spots.
2. **Sensing range can be short.** With the composed recipe the real v1 arm reach (2.3 spacings) costs nothing on held-out shapes (0.74; 2 seeds); 1.6 spacings costs 0.02. Without the recipe the same short range costs 0.05.
3. **The composed recipe** (pile start, 16x16 map = 256 numbers, noisy-clock training, repair batches, weight 100): held-out 0.74 to 0.75, zero collisions, tolerant of clock errors (0.82 vs 0.83 clean), 128 to 512 bots, and a speed cap; rigid travel then form works (2 seeds).
4. **Scales to more bots without retraining** (trained with 256; sensing kept fixed in spacings): well formed at 1024 and 2048 bots (1 model, 1 seed).
5. **Collisions are removed by one local rule** (push apart pairs closer than 0.7 spacings): 0.00 collisions under motion noise and asynchronous updates (1 seed).
6. **Repair has to be trained in** (cover 0.75 to 0.97 after a cut chunk at large range; 2 seeds); it costs about 0.01 of formation quality (2 seeds).
7. **A shared swarm-density state improves repair at the shortest range** (cover 0.85 to 0.91; matched control 0.86; 2 seeds), partially.

## What does not work, or is not enough
- A 64-number map costs 0.05 to 0.09 of held-out overlap from scattered noise (2 seeds); 256 numbers cost nothing (1 seed). Removing the shared clock gives undersized, skeletal shapes (2 seeds).
- Repair of a large cut at short range is still incomplete (0.91 cover; the pictures show unfilled regions).
- Motion noise is only handled with the hard-core rule. Everything is 2D with idealised moves (no arms, no docking, no link strength).

## Things that turned out wrong (kept in the journal)
`w2` ranking said the learned models beat the rule when the pictures said otherwise (tick 26); a shared-state wave looked good in `w2` and was invisible in the pictures (tick 25 to 27); sensing radius was converted to spacings wrongly (tick 44; corrected: r=0.25 is 4.1 spacings, the arm reach is 2.35); several evaluation mistakes (flipped letters, wrong start state, a pile generator that duplicated points beyond 1500 bots, a decorator moved onto the wrong function). Each is logged with its fix.

## What I would do next
Not more Stage 1 sweeps: they now fill in curves already understood. The candidates are (a) a first Stage 2 model with the repo's real body numbers (finite bodies, sensing equal to the arm reach, docking only where bots touch, the cantilever limit from `assets/microbot_v1/DESIGN.md`), or (b) 3D point clouds. Needed from the owner of the bot design: how neighbours are sensed (IR or BLE range and rate), how fast a bot moves per step, and the target swarm size.
