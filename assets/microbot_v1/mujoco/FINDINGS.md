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
| Can it turn? | **Not with these single-arm or paired-sweep moves** (a three-waypoint gait can: see the last section). Yaw change after one-arm +90, both arms +90/+90, and opposite +90/-90 sweeps: -0.8, -2.4 and 0.2 deg. Other gaits (planting both tips and rotating, docking to a neighbour) were not tried, so steering is unproven rather than impossible. |

**What this means for the swarm model.** As designed and with this gait, a bot slides along ONE line only, perpendicular to its own fixed pole axis, in either direction, in strides from a few mm to 50 mm. It is not a holonomic
planar mover. `lab/embody.py` now has `axis_lock` (moves projected onto each bot's allowed line, with the random per-bot pole-axis heading that the no-compass cells already use) and `axis_drift` (the measured drift).


## Steering exists (update, later the same day)
`gait_search.py` (plant one tip, drag the other arm) turns the bot by at most 11 deg and tips the pole axis by up to 30 deg: not useful. `gait_random.py` then searched 360 random sequences of three waypoints over
the four pitch and azimuth joints (angles in 20 and 45 deg steps, 120 deg/s slew, 0.4 s settle), keeping those that end upright (pole axis within 15 deg of horizontal, ball on the floor). Because the bot is end-to-end
symmetric, what matters is the yaw of the line it can move along, i.e. yaw modulo 180 deg: 264 of 360 ended upright, 168 changed the line by more than 20 deg and 79 by more than 45 deg (several top raw scores
of about 180 deg were somersaults that reversed the pole axis, not turns). The best three were then re-run 24 times each with every waypoint angle perturbed by 5 deg (normal) and friction drawn from 0.4 to 1.0
(`gait_robust.py`):

| Gait | Mean yaw of the line | Spread | Runs with change > 45 deg | Ended upright | Translation during the turn | Time |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | -84 deg | 11 deg | 100% | 100% | 36 mm | 4.9 s |
| 1 | -86 deg | 24 deg | 92% | 100% | 52 mm | 4.4 s |
| 2 | +68 deg | 15 deg | 92% | 88% | 58 mm | 3.8 s |
| 4 (rejected) | +7 deg | 42 deg | 25% | 75% | 64 mm | 4.6 s |

(Gait numbers follow the order in `gait_random_best.json` at the time of the test; the table lists them best-first. The first three entries of that file are the nominal top three.)

Waypoints of the nominal top three (each waypoint sets both azimuths and both pitches, in degrees; the sign of a turn is fixed by its sequence, so a turn the other way needs its own search):

| Nominal rank | Waypoints |
| --- | --- |
| 1 | (az_top 180, pitch_top -60, az_bot -180, pitch_bot -80) → (az_top 90, pitch_top -80, az_bot -135, pitch_bot -80) → (az_top 0, pitch_top -40, az_bot -45, pitch_bot 40) |
| 2 | (az_top -45, pitch_top -80, az_bot -90, pitch_bot 20) → (az_top -90, pitch_top 60, az_bot 0, pitch_bot 40) → (az_top -90, pitch_top 60, az_bot 45, pitch_bot -40) |
| 3 | (az_top 135, pitch_top -20, az_bot -135, pitch_bot -60) → (az_top -90, pitch_top -80, az_bot 0, pitch_bot 20) → (az_top -135, pitch_top -40, az_bot -45, pitch_bot -60) |

**Reading.** A bot can switch between two roughly orthogonal lines of motion (a quarter turn) for about 5 s and 40 to 60 mm of unwanted translation; with fine strides along each line that is enough in principle to reach any
position on the plane, in a Manhattan-like way. Caveats: simulation only, flat-disc feet, selected from 360 random samples and checked only against joint-angle error and friction (not against model error such as tooth
engagement, cable limits or a real floor); turning angles other than about 85 deg were not found; whether it also works with neighbours touching was not tested.


## Scaled search (9,600 random sequences on five hosts' CPU cores) and sensitivity test
`gait_random.py` was run with 22 to 46 worker processes per host: 4,800 sequences for turning (`turn_a.json`, `turn_b.json`) and 6,400 for motion along the pole axis (`axis_a.json`, `axis_b.json`; objective: displacement along the
pole axis minus half the sideways displacement, per second). The top 10 of each file were re-run 24 times with +-5 deg on every waypoint angle and friction from 0.4 to 1.0 (`gait_robust.py`).

**Turning, best robust results** (yaw of the line of motion, mod 180; "dx, dy" = body displacement in the start frame, mm):

| File, rank | Mean yaw | Spread | Change > 45 deg | Ended upright | dx, dy | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `turn_b` 14 | -86 deg | 17 deg | 100% | 100% | 11 +- 4, 32 +- 2 | 4.1 s |
| `turn_b` 13 | -88 deg | 9 deg | 100% | 71% | -1 +- 3, 28 +- 4 | 3.9 s |
| `turn_a` 7 | -68 deg | 11 deg | 96% | 96% | -42 +- 7, -51 +- 5 | 3.4 s |
| `turn_a` 3 | -79 deg | 14 deg | 100% | 96% | -61 +- 28, -69 +- 24 | 3.7 s |
| `turn_a` 10 | +87 deg | 16 deg | 100% | 92% | -36 +- 34, 30 +- 15 | 4.1 s |

`turn_b` 14 is the most useful: about the same turn as the earlier best, faster, and its side shift is predictable to about 2 mm, so a controller could compensate it. Both turn directions occur (`turn_a` 10 is positive). Many of the other top candidates were
flukes (mean yaw near 0 under perturbation), so a nominal result from one run is not evidence.

**Motion along the pole axis: no robust gait found.** None of the 40 best axis candidates moves a bot reliably along its pole axis: the spread of the displacement is about as large as its mean (for example dx -137 +- 63 mm,
-105 +- 37 mm, 129 +- 62 mm), and the sequences that do displace the bot along the old axis also turn it by about 75 deg, so they are turn-plus-roll combinations rather than a straight slide. Negative result from about 6,400 random three-waypoint
sequences with a fixed 20/45 deg grid; a smarter search (more waypoints, optimisation instead of sampling) might do better.

**Chaining is not solved.** Starting a second manoeuvre from the state the first leaves behind does not reproduce the clean-start result: after a roll the ball has rotated about its pole axis, so the arms' pitch plane is no longer vertical. Offsetting the azimuths
by the IMU roll angle restores only part of the turn (-36 to -56 deg instead of -75 deg in `demo_comp.py`). `render_demo.py` therefore shows each manoeuvre from a clean rest state.

**Compute note.** MuJoCo contacts run on the CPU, so the GPUs do not speed this up; the hosts' cores do (30 to 52 each). About 9,600 sequences took roughly 10 minutes across five hosts.


## Docked pair (dock_pair.py): a rigid weld does not give an in-place turn
Question from the literature review: can two docked bots do what one cannot, for example turn in place? Test: two bots, a weld constraint between their bodies at the initial pose (a stand-in for the connector; the toothed face, which face docks and the magnets' holding torque are NOT modelled), both run the one-roll script with azimuth +90 and -90 (opposite) or +90 and +90 (same). One run per case, no repeats, so nothing here is a statistic.
- **Side by side along the roll direction (offsets 32 mm, or 22 mm diagonal):** the pair barely moves (centre 3.6 to 5.1 mm, yaw 0), identical for welded, free, opposite and same-direction rolls: the neighbour blocks the roll. A single bot rolls about 18 mm in the same script.
- **Poles overlapping (offset 32 mm along the pole axis):** the arms collide at the start (22 contacts); numbers meaningless (98 mm of motion), discarded.
- **End to end, tips touching (offset 82 mm; each bot reaches 40.9 mm along its pole):** welded and opposite rolls: yaw 0.1 deg, the pair centre moves 44 to 48 mm and the body +Y z goes to +0.99, i.e. the pair flips over (single bots end at -0.34). Welded and same-direction: yaw -1.5 deg, centre 28 mm. Not welded, opposite rolls: yaw -22.7 deg, centre 6 mm (touching tips only). 
- **Reading:** with this rigid weld a docked pair does NOT turn in place in this model: side-by-side docking blocks the roll, end-to-end docking with opposite rolls tumbles the pair. An unwelded touching pair did yaw about 23 deg, which I did not investigate. A softer or one-sided (pivoting) dock, or a different dock face, could behave differently and was not tried.
