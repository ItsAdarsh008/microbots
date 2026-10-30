# Stage 2 plan: the swarm with the real bot's body (2D first)

Numbers from `assets/microbot_v1/DESIGN.md`: ball 30 mm; touching distance 34.5 mm; tip-to-tip arm reach 81 mm (about 2.35 touching diameters); 6 genderless connectors per bot (2 arm tips, 4 equatorial ports); 19.3 g; connectors assumed 3 N each (to be measured); only one bot can cantilever sideways; triangles are the rigid cell.
Units: 1 particle spacing = 1 touching bot diameter.

## v0 (done, journal tick 52): physical bodies and a speed limit, no retraining
Result: bodies and speed limits cost about 0.01 of overlap; a swarm that senses 2.3 spacings and starts as a pile forms the shape in 32 rounds at up to 0.5 spacings per round.

## v1 (done at graph level, journal tick 53): links under the connector limit
Result: with at most 4 ports + 2 tips per bot the finished swarm links into a connected, rigid (rank 2N-3) frame once links may reach 1.6 body diameters (touching only: rigidity 0.90). Not yet: arm geometry, minimal links, strength.

(original v1 notes)
- Choose links among bots within arm reach so that each bot has at most 6, the graph is connected, and ideally triangulated; measure how many bots can be bonded and whether the structure is rigid (about 4 links per bot in 2D).
- Add a bond-aware term to training or a final docking phase; compare. Decide whether the swarm should end in a triangular lattice rather than an even random spread (the training targets are random even samples, not lattices).
- Repair with docked bots: a bot cannot just walk through a docked structure.

## v1b (journal tick 54): flat structures
In the plane a v1 bot has 4 usable connectors (2 in-plane ports + 2 tips); a rigid pin-jointed 2D frame needs 3.97 links per bot on average, so it is at the counting limit (greedy reached rank 0.93 to 0.98 of 2N-3). Rigid planar structures need the out-of-plane ports (3D) or bending-resistant joints. Design decision needed.

## v2: 3D (the `linking/` lattice work: slide and pivot moves, the floor, the cantilever limit); decide whether the learned flow is the planner for target cells there.

## Needed from the owner of the design
How neighbours are sensed (IR or BLE range, rate, whether it senses direction), how fast a bot moves per step (arm crawl or roll), and the target swarm size.
