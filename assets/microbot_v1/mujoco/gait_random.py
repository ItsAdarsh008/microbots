"""Random search for a TURNING gait: sequences of 3 waypoints over the four pitch/azimuth joints of one bot.

    python gait_random.py [samples] [seed]

A waypoint sets (az_top, pitch_top, az_bot, pitch_bot); joints slew at 120 deg/s, then a 0.4 s settle. We score each
sequence by the change of the pole axis' horizontal heading (yaw), the body's translation, and whether the bot ends upright
(pole axis within 15 deg of horizontal, ball on the floor) so the move could be repeated.
"""
import sys
from multiprocessing import Pool
import numpy as np
from bot import BotSim, JOINT_SPEED
from gait_search import yaw_deg, tilt_deg

PITCH = np.arange(-80, 81, 20)
AZ = np.arange(-180, 181, 45)


def sample(rng, k=3):
    return [dict(az_top=float(rng.choice(AZ)), pitch_top=float(rng.choice(PITCH)), az_bot=float(rng.choice(AZ)), pitch_bot=float(rng.choice(PITCH)))
            for _ in range(k)]


def evaluate(seq, mu=0.6):
    s = BotSim(friction=mu); s.run(0.3)
    y0, b0 = yaw_deg(s), s.body_xy().copy(); t = 0.0
    for wp in seq:
        span = max(abs(wp[j] - s.target[j]) for j in wp)
        dur = span / JOINT_SPEED + 0.4
        s.goal(**wp); s.run(dur); t += dur
    yaw = (yaw_deg(s) - y0 + 180) % 360 - 180
    yaw_eff = (yaw + 90) % 180 - 90            # the bot is end-to-end symmetric: only the LINE it can move along matters, so yaw mod 180
    return {"seq": seq, "yaw": yaw, "yaw_eff": yaw_eff, "move": float(np.linalg.norm(s.body_xy() - b0)), "tilt": tilt_deg(s), "z": s.body_z(), "time": t,
            "ok": bool(np.isfinite(s.body_xy()).all())}


def work(args):
    i, seed = args
    return evaluate(sample(np.random.default_rng(seed * 100003 + i)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with Pool(3) as p:
        res = p.map(work, [(i, seed) for i in range(n)], chunksize=5)
    good = [r for r in res if r["ok"] and abs(r["tilt"]) < 15 and r["z"] < 16.5]
    good.sort(key=lambda r: -abs(r["yaw_eff"]) / r["time"])
    print(f"{n} sequences; ended upright: {len(good)}; effective yaw (mod 180) > 20 deg: {sum(abs(r['yaw_eff'])>20 for r in good)}; > 45 deg: {sum(abs(r['yaw_eff'])>45 for r in good)}")
    print(f"{'yaw':>7} {'yaw mod180':>10} {'deg/s':>6} {'move mm':>8} {'tilt':>6}  waypoints (az_top, pitch_top, az_bot, pitch_bot)")
    for r in good[:8]:
        print(f"{r['yaw']:7.1f} {r['yaw_eff']:10.1f} {abs(r['yaw_eff'])/r['time']:6.1f} {r['move']:8.1f} {r['tilt']:6.1f}  " + " | ".join(f"({w['az_top']:.0f},{w['pitch_top']:.0f},{w['az_bot']:.0f},{w['pitch_bot']:.0f})" for w in r["seq"]))
    import json
    json.dump(good[:20], open("gait_random_best.json", "w"), indent=1)
    print("\nrobustness of the best 6 (effective yaw at friction 0.4 / 0.6 / 1.0; 0.6 is the search value):")
    for r in good[:6]:
        ys = [evaluate(r["seq"], mu)["yaw_eff"] for mu in (0.4, 0.6, 1.0)]
        print("  " + "  ".join(f"{y:7.1f}" for y in ys) + f"   (move {r['move']:.0f} mm, {r['time']:.1f} s)")
