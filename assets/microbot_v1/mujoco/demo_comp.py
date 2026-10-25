"""IMU-compensated manoeuvres: roll angle of the ball about its pole axis, and gaits expressed relative to it.

The validated gaits assume the arms' pitch plane is vertical (body +Y pointing down). Rolling spins the ball about its pole axis,
so after any roll the plane is rotated by phi. A real bot knows phi from its IMU; offsetting every azimuth command by it
restores the gait's frame.
"""
import json
from pathlib import Path
import numpy as np
from bot import BotSim, JOINT_SPEED
from gait_search import yaw_deg

HERE = Path(__file__).resolve().parent
TURNS = json.load(open(HERE / "gait_random_best.json"))


def roll_phi(s):
    """Signed angle (deg) of body +Y around the pole axis, measured from straight down."""
    R = s.data.xmat[s.model.body("bot").id].reshape(3, 3)
    a, y = R[:, 2], R[:, 1]
    down = np.array([0.0, 0.0, -1.0])
    yp = y - (y @ a) * a; dp = down - (down @ a) * a                 # both projected onto the plane perpendicular to the pole axis
    yp /= np.linalg.norm(yp) + 1e-12; dp /= np.linalg.norm(dp) + 1e-12
    return float(np.degrees(np.arctan2(np.cross(dp, yp) @ a, dp @ yp)))


def run_gait(s, seq, sgn_top, sgn_bot, frame=None, label="turn"):
    """Run a waypoint sequence with the azimuths offset by the current roll angle."""
    phi = roll_phi(s)
    for wp in seq:
        wp = dict(wp)
        wp["az_top"] = wp["az_top"] + sgn_top * phi
        wp["az_bot"] = wp["az_bot"] + sgn_bot * phi
        span = max(abs(((wp[j] - s.target[j] + 180) % 360) - 180) if j.startswith("az") else abs(wp[j] - s.target[j]) for j in wp)
        s.goal(**wp)
        dur = span / JOINT_SPEED + 0.4
        steps = int(round(dur / s.model.opt.timestep)); every = int(round(0.05 / s.model.opt.timestep))
        for n in range(steps):
            s.run(s.model.opt.timestep)
            if frame and n % every == 0:
                frame(label)


if __name__ == "__main__":
    print("turn gait (nominal rank 3) after the ball has been rolled by different amounts; yaw change of the line (mod 180):")
    print(f"{'pre-roll sweep':>14} {'phi':>7} " + "  ".join(f"signs({a:+d},{b:+d})" for a in (1, -1) for b in (1, -1)))
    for pre in (0, 45, 90, 135):
        row = []
        for a in (1, -1):
            for b in (1, -1):
                s = BotSim(); s.run(0.3)
                if pre:
                    s.goal(pitch_top=30); s.run(0.6); s.goal(az_top=pre); s.run(pre / JOINT_SPEED + 0.6)
                    s.goal(pitch_top=0); s.run(0.6); s.goal(az_top=0); s.run(pre / JOINT_SPEED + 0.6)   # straighten the arms first, then un-swing (no reaction roll)
                phi = roll_phi(s); y0 = yaw_deg(s)
                run_gait(s, TURNS[2]["seq"], a, b)
                row.append(((yaw_deg(s) - y0 + 90) % 180) - 90)
        print(f"{pre:14d} {phi:7.1f} " + "  ".join(f"{y:13.1f}" for y in row))
