"""Demo: one bot rolls along its line, turns about a quarter turn with a three-waypoint gait, then rolls in the new direction.

    python demo_path.py            # prints the path
    python demo_path.py render out.gif   # also renders a GIF (needs MUJOCO_GL=osmesa or egl)

The turn gait is nominal rank 3 from gait_random_best.json (the best of the validated ones: -84 +- 11 deg).
"""
import json
import sys
from pathlib import Path
import numpy as np
from bot import BotSim, JOINT_SPEED
from gait_search import yaw_deg

HERE = Path(__file__).resolve().parent
TURN = json.load(open(HERE / "gait_random_best.json"))[2]["seq"]


def line_dir(s):
    """World direction (unit, xy) of the line the bot can roll along: perpendicular to its pole axis, which is body +x here."""
    R = s.data.xmat[s.model.body("bot").id].reshape(3, 3)
    a = R[:, 0][:2]
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else np.array([1.0, 0.0])


def phases(s, frame=None):
    """Run the whole sequence; `frame(label)` is called about every 1/20 s of simulated time."""
    log = []
    def run(label, seconds):
        steps = int(round(seconds / s.model.opt.timestep)); every = int(round(0.05 / s.model.opt.timestep))
        for n in range(steps):
            s.run(s.model.opt.timestep)
            if frame and n % every == 0:
                frame(label)
    def note(name):
        log.append((name, s.body_xy().copy(), yaw_deg(s), line_dir(s).copy()))
    s.run(0.3); note("start")
    # 1. roll along the line: plant the top arm at 30 deg, sweep its azimuth 120 deg
    s.goal(pitch_top=30); run("roll", 0.6)
    s.goal(az_top=120); run("roll", 120 / JOINT_SPEED + 0.5); note("after roll 1")
    # 2. neutral, then the turn gait
    s.goal(az_top=0, pitch_top=0, az_bot=0, pitch_bot=0); run("reset", 1.5); note("after reset")
    for wp in TURN:
        span = max(abs(wp[j] - s.target[j]) for j in wp)
        s.goal(**wp); run("turn", span / JOINT_SPEED + 0.4)
    note("after turn")
    # 3. neutral, then roll again
    s.goal(az_top=0, pitch_top=0, az_bot=0, pitch_bot=0); run("reset", 1.5); note("after reset 2")
    s.goal(pitch_top=30); run("roll", 0.6)
    s.goal(az_top=120); run("roll", 120 / JOINT_SPEED + 0.5); note("after roll 2")
    return log


if __name__ == "__main__":
    s = BotSim()
    log = phases(s)
    print(f"{'phase':>14} {'body xy (mm)':>20} {'pole yaw':>9} {'roll line direction':>22}")
    for name, xy, yaw, d in log:
        print(f"{name:>14} {np.round(xy,1)!s:>20} {yaw:9.1f} {np.round(d,2)!s:>22}")
    d0, d1 = log[1][3], log[-1][3]
    m1, m2 = log[1][1] - log[0][1], log[-1][1] - log[-2][1]
    print("roll 1 displacement", np.round(m1, 1), "| roll 2 displacement", np.round(m2, 1),
          "| angle between them %.0f deg" % np.degrees(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-9), -1, 1))))
