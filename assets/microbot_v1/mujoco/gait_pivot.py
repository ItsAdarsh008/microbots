"""Measure the tip-pivot gait on the CAD-mesh MuJoCo model: plant one arm tip, sweep the azimuth, record the move.

    python gait_pivot.py [friction]
"""
import sys
import numpy as np
from bot import BotSim, JOINT_SPEED


def one(pitch, sweep, friction=0.6, side="top"):
    s = BotSim(friction=friction)
    s.run(0.3)
    s.goal(**{f"pitch_{side}": pitch}); s.run(abs(pitch) / JOINT_SPEED + 0.6)
    planted = side_contact(s, side)
    b0, t0, z0 = s.body_xy().copy(), s.tip_xyz(side).copy(), s.body_z()
    s.goal(**{f"az_{side}": sweep}); s.run(abs(sweep) / JOINT_SPEED + 0.6)
    b1, t1 = s.body_xy().copy(), s.tip_xyz(side).copy()
    up = float(np.degrees(np.arccos(np.clip(-s.up_vector_body_y(), -1, 1))))   # how far the pitch plane has rotated from vertical
    return {"pitch": pitch, "sweep": sweep, "planted": planted, "move": b1 - b0, "dist": float(np.linalg.norm(b1 - b0)),
            "slip": float(np.linalg.norm(t1[:2] - t0[:2])), "z": s.body_z(), "tilt": up, "ok": bool(np.isfinite(b1).all())}


def side_contact(s, side):
    return f"tip_{side}" in s.contacts_with_floor()


if __name__ == "__main__":
    mu = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    print(f"friction {mu}; top arm; sweep +90 deg at {JOINT_SPEED} deg/s")
    print(f"{'pitch':>5} {'tip on floor':>12} {'body move (mm)':>22} {'|move|':>7} {'tip slip':>8} {'body z':>7} {'plane tilt':>10}")
    for p in (20, 30, 43, 55, 70, 85):
        r = one(p, 90, mu)
        print(f"{p:5d} {str(r['planted']):>12} {np.round(r['move'],1)!s:>22} {r['dist']:7.1f} {r['slip']:8.1f} {r['z']:7.1f} {r['tilt']:9.1f}°")
