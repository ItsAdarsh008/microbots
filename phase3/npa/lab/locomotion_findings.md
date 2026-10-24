# Locomotion and sensing feasibility (agent report, 2026-10-24)

Source: a feasibility analysis of `assets/microbot_v1` (kinematics from `report.json`: ball R 15, pin 18.5 mm along the pole
axis, arm 22, pitch +-100 deg, 120 deg/s). **Rigid-body kinematics only: no friction, dynamics, cable limits or collisions.
Nothing here has been checked in physics.** Numbers marked "guess" are the analyst's.

- The CAD cannot move on a flat surface as designed: no wheels or feet.
- Gait A, tip-pivot orbit: plant one arm tip, spin the azimuth joint, the ball rolls around the tip (R*omega = 31 mm/s). A +-180 deg
  cable loop caps one orbit at 156 deg. Alternate arms. Pair displacement 36 mm (30 deg pivots, 2.6 s) to 135 mm (156 deg, 7.4 s),
  mean 14 to 20 mm/s. Directions within about 12 deg of the pole axis are unreachable per stride. Lift and plant time 0.7 s is a guess.
- Gait B, pendulum roll: hold one arm sideways and counter-rotate; drive torque 0.27 to 0.6 mN*m against rolling resistance of
  0.014 to 0.057 mN*m (the coefficient is a guess). About 20 mm/s along one line only; strictly non-holonomic.
- Stability: the support in Gait A is a line, with the centre of mass above its end: no margin about the roll axis; it depends on
  the planned self-locking worm drives.
- Lattice hop: spheres are 30 mm, ports 28 mm apart, teeth lock shear, so there is no slide. A hop is release, pivot stride, re-dock.
  A 28 mm shift with one tip docked needs 49 mm of reach; the maximum is 40.5 mm.
- Sensing: BLE gives no bearing. Use a modulated-IR ring (6 to 8 photodiodes) as in Droplets (arXiv 1709.07051); Kilobots give
  range only, +-2 mm at 10 cm (ICRA 2012); Mona IR bearing a few to 14 deg (PMC7309109); Pebbles pass data through the magnet
  contacts at 9600 bps (MIT dspace 1721.1/70987). UWB is useless at this size.
- Bandwidth is not the limit; **line of sight is**: expect 4 to 6 visible neighbours.
- Top transfer risks: (1) holonomic steps, (2) no shared frame, (3) occlusion, (4) lattice hop and the 3 N EPM, (5) slip,
  no odometry and cable wind-up. Cheapest experiments: run Gait A in `sandbox/sim.js`; add heading error; ray-cast the lizard
  for visible neighbours; bench-test a printed Hirth face; measure a printed tip-pivot's repeatability.
