# From dots to microbots: embodiment decisions

Everything we learn here has to carry over to the real bot in `assets/microbot_v1/`. A dot in the simulation can
teleport, sense a hundred neighbours exactly and overlap; a bot cannot. These are the decisions, the numbers behind
them and which ones are assumptions that still need checking. Change a decision here first, then in code.

## The bot (from the CAD, `report.json` and `DESIGN.md`)

| Quantity | Value | Source |
| --- | --- | --- |
| Body | sphere Ø30 mm (r = 15 mm), 19.3 g | CAD |
| Docking ports | 4 on the equator, 14 mm from the centre; genderless Hirth face + electro-permanent magnet (3 N, an assumption) | CAD |
| Port-to-port body spacing | 2 x 14 = **28 mm** | CAD |
| Arms | 2, each azimuth (continuous) + pitch (+-100 deg) + roll; 22 mm long; 81 mm tip to tip | CAD |
| Triangle (tip docking) spacing | 62.5 mm (3-bot ring) | CAD |
| Joint speed | 120 deg/s | sandbox `sim.js` |
| Electronics | MCU, IMU, IR or BLE (PCB 1.2 g) | design note |
| Locomotion | **not defined by the design**: decided below | - |

## Decisions

**D1. Scale (corrected).** 1 world unit = **1160 mm**. Body diameter 30 mm = 0.0259 u (radius 0.0129). The first choice,
900 mm per unit, made 1,024 bots fill 78% of the lizard (hex packing maximum 91%) and the seed disc 178% packed, which is
impossible, so overlap projection exploded (`runs/phys_*`, H19). At 1160 mm per unit the lizard is 47% packed; a physical
run needs a seed disc of radius 0.55 (`seed_radius`). Derived: sensing eps 0.05 u = 58 mm = 2 body diameters; stride
18 to 50 mm = 0.0155 to 0.043 u; range noise 2 mm = 0.0017 u; port-to-port pitch 28 mm = 0.0241 u.

**D2. Sensing radius and neighbours.** The paper's lizard sensed about 120 neighbours (eps 0.2 u = 180 mm = 6 body
diameters). A bot cannot talk to 120 neighbours, and line of sight limits it to about 4 to 6. We test eps = 0.066 u (59 mm,
about 13 neighbours), eps = 0.05 u (45 mm, about 7: the realistic setting) and eps = 0.04 u (about 5).

**D3. What a bot senses (checked against the literature; see `locomotion_findings.md`).** The bot has an IMU but **no yaw
reference** (a magnetometer next to six EPMs and the motors is doubtful), so every bearing is measured **in the bot's own
frame**. BLE cannot give bearing; the design note's "IR or BLE" must become a **modulated-IR ring of 6 to 8 photodiodes
between the ports** (as in Droplets), plus contact sensing at the docked ports. Assumed accuracy: range sigma 2 mm
(Kilobot-class), bearing sigma 5 deg (Mona-class). Line of sight is needed, and 30 mm bodies block the second ring: expect
**4 to 6 visible neighbours, not 13**. Each state is 16 channels at 8 bits (160 bits per packet with ID and CRC, 5 ms at 30 kb/s).
Packet loss 10 to 30%.

**D4. How a bot moves (the CAD has no wheels or feet: this is our own proposal, kinematics only, not yet checked in
physics).** "Tip-pivot orbit": plant one arm tip, spin the azimuth joint so the ball rolls around the tip, then plant the
other arm and repeat. At 120 deg/s this gives 31 mm/s peak and 14 to 20 mm/s mean. It is **non-holonomic**: a pair of
pivots moves the bot 36 to 135 mm in a coarse direction (82 deg off the pole axis for small pivots, 12 deg for large), with
a **minimum useful stride of about 18 mm** and a maximum of about 50 mm per tick. Directions within about 12 deg of the pole
axis are unreachable in one stride; over several strides any direction can be reached. Stride noise assumed: +-20% length,
+-5 deg heading. **Control tick: 4 s.** Bots may not overlap. In the simulation: a proposed displacement shorter than 18 mm
becomes 0, longer than 50 mm is capped, and noise is added (`min_step`, `max_step`, `step_len_noise`, `step_dir_noise`).

**D5. How a docked structure rearranges ("lattice" embodiment, weak assumption).** Two port-docked bodies are 28 mm apart
and the bodies are 30 mm across, so a bot cannot pivot around a docked neighbour without first backing off, and the teeth
lock shear, so there is **no sliding hop**. A hop means: release the EPM, run a pivot stride, re-dock (needs about +-1 mm
alignment). A flat floor lattice with horizontal poles is anisotropic (28 mm across the pole axis, about 81 mm along it; 62.5
mm for the triangle cell). The 28 x 28 mm square sheet only works in a vertical plane under gravity. So `lattice` models a
vertical docked sheet or a future 3D structure; it uses `hop_success` below 1 and one hop per tick. A floor-docked
triangulated lattice is the next step.

**D8. No shared frame (new, from the feasibility check).** Everything the sim does with offset vectors in one global frame
(the paper's gradient and density-gradient perception, and the move it outputs) has no counterpart on a real bot. The rule
must work when every bot measures in its own, unknown, fixed heading. Emulated exactly by rotating each bot's perceived vector
features by -theta_i and its output move by +theta_i, with theta_i random but fixed per bot (`heading`). A rotation-equivariant
cell (invariant features in, equivariant move out) should be immune; a plain MLP should not. This is testable now (H14).

**D6. Training for the robot.** Two routes, both measured: (a) take a model trained on ideal dots and *deploy* it under the
embodiment without retraining (the size of the sim-to-real gap, `deploy.py`); (b) train *inside* the embodiment
(`"embody"` in the cell config), using a straight-through gradient for the non-differentiable parts.

**D7. What counts as transferable.** A result counts only if it survives the "phys" embodiment at eps 0.066. A result
that only holds for ideal dots at 120 neighbours is reported as such.

## Still unverified (do not treat as facts)
- the tip-pivot gait itself (a kinematic construction: needs the cannon-es sandbox or a bench test, and a real foot is a toothed Ø10 face that may slip),
- IR range-and-bearing accuracy of our own ring (literature figures are for other robots),
- 3 N holding force, and docking reliability when two bots slide into a port.

**D9. Stride granularity versus spacing (new, from the first physical shape-agnostic run).** Sensing at 58 mm forces a bot spacing of about 38 mm for
neighbours to see each other, and a stride is 18 to 50 mm, so a single move is about a whole spacing: bots cannot make fine adjustments. This is a property of
the proposed gait and sensor, not of the learning method; `ag_phys_abl_*` separate its parts. If it dominates, the hardware options are a finer gait (shorter minimum stride)
or a longer sensing radius with sparser swarms.
