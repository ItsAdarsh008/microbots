# Gait measurements on the CAD meshes (MuJoCo)

`bot.py` builds an articulated model of one bot from `../stl/part_*.stl` (STLs come from the `meshes` branch; they are git-ignored here). Chain, mass (19.3 g), joint speed (120 deg/s)
and servo torque (15 mN*m) as in `microbot.py` / `DESIGN.md`. Collision uses each part's convex hull, so the toothed face is a flat disc. `gait_pivot.py` runs the experiments below.
Everything is simulation: nothing has been built or measured on hardware, and the foot friction (a toothed PA12 face on an unknown floor) is the least certain input.

| Question | Measurement |
| --- | --- |
| Does it sit still? | Yes: zero drift, ball sinks 0.4 mm into the (soft) floor. (First version bounced: stiff servos on a 1e-8 kg*m^2 arm inertia need the implicit integrator, armature and softer gains.) |
| Does the tip-pivot orbit from the kinematic analysis work? | **No.** The tip's face touches the floor at only about 26 deg of pitch; beyond that the arm lifts the ball, the bot tilts and the other arm's tip lands on the floor (a ball + two tips tripod). During an azimuth sweep the tip slips 1.2 to 1.5 times as far as the body moves, at friction 0.4 to 1.5. |
| What does move it? | **Reaction rolling.** Sweeping one arm's azimuth rolls the ball about the pole axis (111 to 149 deg for a 90 deg sweep); the body moves 28 to 37 mm perpendicular to the pole axis (arc = 15 mm * roll angle). The sign of the sweep sets the direction. It is robust to friction. |
| Stride vs sweep (pitch 30, mu 0.6) | 10 deg: 2.6 mm; 20: 5.5; 30: 8.8; 45: 14.1; 60: 19.6; 90: 30.4; 135: 43.3; 180: 50.8 mm. About 0.3 mm per degree, so stride length is finely controllable (the assumed 18 mm minimum stride is not a property of this gait). Motion time = sweep / 120 deg/s. |
| Drift along the pole axis | -0.0064 * stride^2 (mm), toward the swept arm's far side: 1.3 mm at 14 mm stride, 5.9 at 30, 19 at 51. Same sign for both sweep directions, so alternating the two arms should cancel it. |
| Can it turn? | **Not with these moves.** Yaw change after one-arm +90, both arms +90/+90, and opposite +90/-90 sweeps: -0.8, -2.4 and 0.2 deg. Other gaits (planting both tips and rotating, docking to a neighbour) were not tried, so steering is unproven rather than impossible. |

**What this means for the swarm model.** As designed and with this gait, a bot slides along ONE line only, perpendicular to its own fixed pole axis, in either direction, in strides from a few mm to 50 mm. It is not a holonomic
planar mover. `lab/embody.py` now has `axis_lock` (moves projected onto each bot's allowed line, with the random per-bot pole-axis heading that the no-compass cells already use) and `axis_drift` (the measured drift).
