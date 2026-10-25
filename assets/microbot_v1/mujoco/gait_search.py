"""Search for a way to TURN one bot (change the heading of its pole axis) using both arms.

    python gait_search.py

Protocol "anchor and drag": plant one arm tip (pitch p_a), then sweep the OTHER arm's azimuth by `sweep` at pitch p_b so its
tip drags on the floor. Measures the yaw of the pole axis (deg), the body move (mm) and whether the bot stays upright.
"""
import itertools
import numpy as np
from bot import BotSim, JOINT_SPEED


def yaw_deg(s):
    a = s.data.xmat[s.model.body("bot").id].reshape(3, 3)[:, 2]
    return np.degrees(np.arctan2(a[1], a[0]))


def tilt_deg(s):
    a = s.data.xmat[s.model.body("bot").id].reshape(3, 3)[:, 2]
    return np.degrees(np.arcsin(np.clip(a[2], -1, 1)))      # elevation of the pole axis above horizontal


def protocol(anchor, p_a, p_b, sweep, mu=0.6, settle=0.8):
    drag = "bot" if anchor == "top" else "top"
    s = BotSim(friction=mu); s.run(0.3)
    s.goal(**{f"pitch_{anchor}": p_a, f"pitch_{drag}": p_b}); s.run(max(p_a, p_b) / JOINT_SPEED + settle)
    y0, b0, t0 = yaw_deg(s), s.body_xy().copy(), tilt_deg(s)
    s.goal(**{f"az_{drag}": sweep}); s.run(abs(sweep) / JOINT_SPEED + settle)
    dy = (yaw_deg(s) - y0 + 180) % 360 - 180
    return {"yaw": dy, "move": s.body_xy() - b0, "tilt0": t0, "tilt1": tilt_deg(s), "z": s.body_z()}


if __name__ == "__main__":
    rows = []
    for anchor, p_a, p_b, sweep in itertools.product(("top",), (25, 40, 60, 80), (25, 40, 60, 80), (-90, 90)):
        r = protocol(anchor, p_a, p_b, sweep)
        rows.append((abs(r["yaw"]), anchor, p_a, p_b, sweep, r))
    rows.sort(key=lambda x: -x[0])
    print("anchor top tip at p_a, drag bottom arm (az sweep at pitch p_b); best yaw first")
    print(f"{'p_a':>4} {'p_b':>4} {'sweep':>5} {'yaw':>7} {'move (mm)':>16} {'tilt before->after':>20} {'z':>6}")
    for _, anchor, p_a, p_b, sweep, r in rows[:10]:
        print(f"{p_a:4d} {p_b:4d} {sweep:5d} {r['yaw']:7.1f} {np.round(r['move'],1)!s:>16} {r['tilt0']:8.1f} -> {r['tilt1']:6.1f} {r['z']:6.1f}")
    print(f"... {len(rows)} combos; yaw > 10 deg: {sum(1 for x in rows if x[0] > 10)}; > 30 deg: {sum(1 for x in rows if x[0] > 30)}")
