# Stage 3 plan: 3D (a plan, not a result; nothing here has been built or run)

## What already exists
`linking/` has a 3D cube-lattice swarm: bots slide and pivot on a grid, always attached to the structure and the floor; a hand-written teacher and a 52,627-weight local network trained by imitation (`linking/learn/`) build the wall, arch and an unseen tower; "what is the target" is a voxel set; `linking/README.md` already lists the hand-off from a particle model: "turn particle positions into a target grid of cells, the lattice swarm fills it" (option A there).

## Where the flow-matching work fits (three options)
- **A. Shape generator and target planner (recommended).** Extend Stage 1 to 3D: N = 512 to 1024 particles, a 3D signed-distance map (coarse 8x8x8 = 512 numbers or 16x16x16 = 4096 numbers as the broadcast), the same local network (radius in spacings), geometric losses, pile start on the floor. Output particles are snapped to lattice cells and handed to the existing lattice swarm as its target. What the flow adds: shapes from a tiny code, formation from a pile, repair after damage; what stays with the lattice policy: moves that never let anything fall.
- B. Flow as the move policy on the lattice: moves are discrete, constrained and gravity-bound; flow matching is a poor fit.
- C. Flow as a build-order prior (which cells to open next): uncertain value; the readiness rule already works locally.

## Work items for A (each with its test)
1. 3D shape generator, **support-aware** (every cell connected to the floor, overhangs limited by what the lattice can build); held-out families; ceiling and rule baselines as in 2D (a project-and-repel rule in 3D).
2. Model: positions in 3 dimensions, trilinear map reading, same attention with a radius; check against the 2D numbers on shapes that are really 2D.
3. Geometric losses in 3D (inside the shape, no crowding, coverage), weight about 100 as in 2D (an assumption to re-test).
4. Metrics: voxel IoU, recall, collisions, and **buildability**: run the lattice swarm on the snapped target and count shapes it completes (the check that matters).
5. Pictures every time (rendered 3D views), by eye next to the rule and the ceiling.
6. Physical limits from `assets/microbot_v1/DESIGN.md`: one bot can cantilever, arches and compression are strong; use them to reject unbuildable generated shapes before training.

## Risks (honest)
Gravity and support are not in the flow model, so a flow output can be unbuildable even when it looks right; snapping to a lattice may change thin parts; memory: attention with N = 1000 bots is 20 times the 2D cost (a radius mask or neighbour lists are needed); the 2D conclusions (weight 100, noisy clock, repair batches) may not transfer.

## Compute and time
The 2D runs took about 25 minutes each on one A100 with 256 particles; 3D with 4 times the particles and a 3D map is plausibly 10 times more per run, so about 5 GPUs for several hours per sweep. Not needed before the data generator and the lattice hand-off check are built.

## What I need to start
A yes on option A (or a different choice), and whether the 3D target should include overhangs or only floor-supported compact shapes at first.
