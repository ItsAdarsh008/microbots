# Microbot v1: design note

Oct 23, 2026 · base CAD model for every later design

![The bot in four poses and the six test structures](renders/overview.png)

## What it is

One microbot is a Ø30 mm ball with an arm on each pole. Straight, it is 81 mm tip to tip, and it weighs about **19.3 g** (estimate, see Mass). Each arm has three joints, and every tip plus four docking ports around the ball's equator carry the **same genderless connector**, so any connector can lock onto any other.

| Joint | Axis | Range (checked in CAD) | What it does |
| --- | --- | --- | --- |
| Azimuth (turret) | the ball's pole axis | 360° (±180° with a cable loop) | swings the arm around the ball |
| Pitch (clevis hinge) | steel pin, perpendicular to the arm | **±100°** | tilts the arm down from the pole, 10° past the equator |
| Roll (tip collar) | the arm's own axis | continuous | lines the connector teeth up for docking |

Each arm can point anywhere inside a 100° cone around its pole, which is 59% of all directions. The two cones overlap in a 20° band around the equator, so between them the arms reach every direction, and both arms can point the same way (the U grip pose).

## The "kneecap" question: why not a ball-and-socket

You were right to doubt it. A socket only holds a ball if it wraps past the ball's equator, and that wrap is exactly what blocks the arm's neck. For a Ø10 ball with a Ø6 neck and a 15° wrap, the arm gets about **±38°** before the neck hits the rim. Pull-out is resisted only by that thin rim working in bending, which is the first thing to crack. There is also no good way to motor-drive a free ball at this size or to run the tip's power wires through it.

What gets the "arm moves anywhere around the ball" behaviour without the fragility is a **turret plus a clevis hinge**: the turret spins about the pole and the hinge tilts the arm. It is a two-axis gimbal with one axis that never runs out of travel.

- **Strength.** The Ø2 mm steel pin is in double shear (supported on both sides), good for roughly 900 N. The weakest load path is the nylon ears bearing on the pin, at about 160 N, which is around 850 times the bot's own weight. The joints are not where it breaks; the connectors are (below).
- **Range.** ±100° pitch, against about ±38° for a socket. The turret is a bar instead of a disc so the arm can swing down past it. That one change took pitch from ±84° to ±100°.
- **No gimbal lock in practice.** The only singular pose is the arm pointing straight along the pole, where the turret's spin just becomes a roll, and the tip has its own roll joint anyway.
- **Weight stays central.** The plan is a bevel differential, like a robot wrist: two motors in the ball drive two coaxial shafts up through the turret spigot. Spinning both together turns the turret; spinning them in opposite directions tilts the arm. Four of the six motors then sit in the ball, and only the small roll motors ride in the arms.

## The connector: genderless Hirth face plus an electro-permanent magnet

Every tip and port is a Ø10 mm face with 24 radial V-teeth (a Hirth coupling, used on turbine shafts) around an electro-permanent magnet (EPM) pole. An EPM switches on or off with a short current pulse and holds with zero power, as in MIT's Pebbles and M-Blocks.

- **Genderless.** There are no male or female ends. Two identical faces mesh when one is flipped onto the other. The tooth crests are offset a quarter pitch so the faces mesh at **every multiple of 15°** of relative roll, which includes the 60°, 90° and 120° that triangular, square and hex lattices need. In CAD, two faces overlap by 0.0 mm³ at 0°, 15°, 45° and 90°, and they collide at 7.5°, as they should.
- **Self-centring.** The magnet pulls the faces together and the tooth flanks slide them into alignment. Once meshed, the teeth carry shear and twist, so the magnet only has to resist pulling apart.
- **Tip to tip, tip to body, body to body.** All three use the same face.

## Connecting many bots: the maths

**Two tips alone only make chains.** If bots link only tip to tip, every junction joins exactly two bots, so the most you can build is chains and loops, never a branching 2D or 3D network. That is why the ball also has **four equatorial ports**, giving six connectors per bot.

**Why six is the right number (Maxwell's rule).** Treat each body as a node and each docked pair as a bar. Every bar uses two connectors, so N bots can make at most 6N / 2 = **3N bars**. A frame in space with pin joints is rigid when it has at least **3N − 6 bars** (in the plane, 2N − 3). Six connectors per bot is therefore exactly enough for rigid 3D structures, provided nearly every connector is used. Four would never be enough in 3D.

**Triangles are the rigid cells.** All of the following are in the assets, built by chaining the kinematics and checked to close to 0.000 mm:

| Structure | Bots | Pose | Body-to-body spacing | Rigid? |
| --- | --- | --- | --- | --- |
| `ring3_triangle` | 3 | arms 60° apart | 62.5 mm | yes, a triangle cannot flex |
| `ring4_square` | 4 | arms 90° apart | 70.1 mm | only with joints locked |
| `ring6_hexagon` | 6 | arms 120° apart | 76.1 mm | only with joints locked |
| `zigzag5` | 5 | arms 120° apart, no roll | open chain | a chain, not a structure |
| `block_2x2x2` | 8 | straight; bodies docked port to port | 28 × 28 × 81 mm grid | 3D, every body uses all 6 connectors |
| `t_junction` | 2 | one tip into a body port | n/a | the branching joint |

The same pose gives a ring or a zigzag depending only on the tips' relative roll (180° or 0°), so a bot changes the shape of a structure just by rolling a tip.

## Strength and weight distribution: what the physics allows

All figures assume an EPM holding force of **3 N**, which still has to be measured on real parts. Bot weight is 19.3 g, or 0.19 N.

- **Tension (hanging).** 3 N / 0.19 N means one connector can carry a chain of about **15 hanging bots**.
- **Bending is the weak spot.** A face can resist about 3 N × 5 mm ≈ 15 mN·m of bending before it peels open. One bot sticking straight out sideways puts 7.7 mN·m on its connector, and two in a row put 31 mN·m. So **only one bot can cantilever horizontally**. For k bots the moment grows as k², about 7.7·k² mN·m.
- **Compression gets stronger with load.** Weight pressing down adds to the magnet's clamp: the bending capacity is (3 N + weight above) × 5 mm, and the teeth's crushing limit is in the kilonewtons. Columns, arches and domes therefore work well. Long sideways beams don't.
- **Design rule for every later phase.** Build **triangulated (truss) and compressive (arch) shapes**, and keep each connector loaded in compression or tension rather than bending. The 3-bot triangle is the basic building cell.
- **Scaling.** Magnet force grows with area (size²) while weight grows with volume (size³), so the bending margin grows as 1 / size. A bot half this size is twice as strong relative to its own weight, which is why tiny movie-scale microbots are physically plausible.
- **Weight distribution.** The battery, PCB, four drive motors and four port EPMs all sit inside the ball. With the arms straight, the centre of mass is 0.2 mm from the ball's centre. With both arms bent fully to one side it moves only 3.1 mm. Each ball is effectively a point-mass node with light bars, which keeps structure loads predictable.
- **Motors don't hold structures up.** The pitch motor's torque, roughly 10–20 mN·m from a Ø4 mm gearmotor, is only enough to *move* one neighbour. Once a structure has formed, joints should be locked by a self-locking worm or brake so loads go through the pins and connectors, not the motors.

## Mass budget (19.3 g)

| Item | g |
| --- | --- |
| Printed nylon (PA12) parts: body 4.1, 2 × (turret 0.24 + arm 0.52 + tip 0.29) | 6.2 |
| 2 steel pins | 0.5 |
| LiPo, about 40 mAh | 1.4 |
| PCB (MCU, IMU, motor drivers, IR or BLE) | 1.2 |
| 4 × Ø4 mm gearmotors (differential drives, in the ball) | 4.4 |
| 2 × Ø4 mm roll gearmotors (in the arms) | 2.0 |
| 6 × EPMs (4 ports, 2 tips) | 3.0 |
| Bevel gears | 0.6 |

The printed-part masses come from the CAD volumes. The bought parts are catalogue-class estimates, and `build.py` holds the list.

## Files (`assets/microbot_v1/`)

| Path | What it is |
| --- | --- |
| `microbot.py` | the parametric model: every dimension is in `Params`, plus the parts, the joint kinematics and the docking maths (`mate`) |
| `build.py` | rebuilds all exports and runs the checks: range of motion, part-to-part interference, ring closure, connector mesh, mass and centre of mass |
| `report.json` | the output of those checks |
| `step/part_*.step`, `stl/part_*.stl` | single parts: body, turret, arm, tip, pin |
| `step/bot_*.step`, `stl/bot_*.stl` | one assembled bot in 5 poses: straight, V120, V60, U grip, twisted |
| `step/<structure>.step`, `stl/<structure>.stl` | the 6 structures in the table above |
| `renders/*.png` | previews of all of the above (only `overview.png` is in git) |
| `sandbox/` | the in-browser physics sandbox, see its README |

The STEP and STL exports (about 220 MB) and the individual renders are not in git: `build.py` regenerates them in about 6 minutes, and `.gitignore` keeps them out.

The STEP assemblies keep one named, coloured part per component, so they open as assemblies in Fusion, FreeCAD or Onshape.

To rebuild after changing a parameter:

```bash
pip install -r requirements-cad.txt   # CadQuery 2.8; kept out of the main requirements.txt
python build.py             # about 6 minutes; add --fast to skip the renders
```

**Checks in the current build:** 0 interferences in all 5 poses and all 6 structures; the rings close with 0.000 mm and 0.000° error; all 12 contacts in the block are exact; pitch is ±100° before the arm hits the body.

## What is not done yet

- The motors, gears, PCB and cables have a reserved space inside the ball but are not modelled as parts. Fitting the real differential is the next CAD step.
- The EPM holding force (3 N) is an assumption. Measure it, and the strength numbers above rescale linearly with it.
- At the rim the teeth are about 0.65 mm apart, which needs resin (SLA) or nylon powder (MJF) printing. For a home FDM printer, set `teeth=12` (30° steps).
- No finite-element stress analysis has been run. The strength figures are hand calculations.
- **Link to Phase 3.** The NCA's 32³ voxel target can map straight onto the `block_2x2x2` lattice, at a cell pitch of 28 × 28 × 81 mm, or onto a triangulated lattice of 3-bot cells. Choosing between them is part of agreeing the hand-off format in week 1.
