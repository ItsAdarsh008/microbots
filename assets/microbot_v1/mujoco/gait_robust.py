"""How sensitive are the best turning gaits to joint-angle error and floor friction?

    python gait_robust.py [n_per_gait]

Takes the top gaits from gait_random_best.json (ranked by effective yaw per second), re-runs each n times with every waypoint
angle perturbed by N(0, 5 deg) and friction drawn from U[0.4, 1.0], and reports the effective yaw (mod 180) and translation.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gait_random import evaluate

SIGMA = 5.0


def work(args):
    seq, seed = args
    rng = np.random.default_rng(seed)
    pert = [{k: (v if k.startswith("pitch") is False and False else float(v + rng.normal(0, SIGMA))) for k, v in wp.items()} for wp in seq]
    for wp in pert:
        for k in wp:
            if k.startswith("pitch"):
                wp[k] = float(np.clip(wp[k], -100, 100))
    return evaluate(pert, mu=float(rng.uniform(0.4, 1.0)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    import os
    files = os.environ.get("FILES", "gait_random_best.json").split(",")
    top = int(os.environ.get("TOP", 4))
    best = [dict(g, src=f) for f in files for g in json.load(open(Path(__file__).resolve().parent / f))[:top]]
    print(f"{n} perturbed runs per gait (angles +-{SIGMA} deg, friction 0.4 to 1.0)")
    print(f"{'gait':>4} {'src':>11} {'nominal':>8} {'mean':>7} {'std':>6} {'|yaw|>45':>9} {'upright':>8} {'dx mean+-std':>14} {'dy mean+-std':>14} {'move':>6} {'time':>5}")
    with Pool(int(os.environ.get("WORKERS", 3))) as p:
        for i, g in enumerate(best):
            res = p.map(work, [(g["seq"], 1000 * i + j) for j in range(n)])
            y = np.array([r["yaw_eff"] for r in res]); up = np.mean([abs(r["tilt"]) < 15 and r["z"] < 16.5 for r in res])
            # circular mean of the line orientation (period 180)
            ang = np.deg2rad(2 * y); mean = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2
            spread = np.rad2deg(np.sqrt(-2 * np.log(max(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()), 1e-9)))) / 2
            dxs = np.array([r["dx"] for r in res]); dys = np.array([r["dy"] for r in res])
            print(f"{i+1:4d} {g['src']:>11} {g['yaw_eff']:8.1f} {mean:7.1f} {spread:6.1f} {np.mean(np.abs(y) > 45):9.2f} {up:8.2f} {dxs.mean():7.0f}+-{dxs.std():4.0f} {dys.mean():7.0f}+-{dys.std():4.0f} {np.mean([r['move'] for r in res]):6.0f} {g['time']:5.1f}")
