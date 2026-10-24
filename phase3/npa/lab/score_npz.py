"""Score a checkpoint written by train.py (w1, b1, w2, eps, eps0) with the lab's evaluator.

    python lab/score_npz.py runs/baseline_mlp_n1024.npz --name baseline_mlp_10k --particles 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import cells  # noqa: E402
import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--name", required=True)
ap.add_argument("--particles", type=int, default=1024)
args = ap.parse_args()

d = np.load(args.npz)
params = {k: jnp.asarray(d[k]) for k in ("w1", "b1", "w2")}
eps, eps0 = float(d["eps"]), float(d["eps0"])
cell = cells.Cell()
result = evaluate.run(cell, params, eps, eps0, args.particles)
sm = result["summary"]
print(f"score {sm['score']:.4f}  excess {sm['excess']:.4f}  control {sm['control_loss']:.4f}  drift {sm['drift']:.4f}")
print("per-kind score:", {k: round(v, 3) for k, v in sm["per_kind_score"].items()})
out = HERE.parent / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)
row = {"name": args.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {"source": args.npz, "cell": {}},
       "params": cell.param_count(params), "summary": sm, "evaluator": evaluate.fingerprint()}
(out / "result.json").write_text(json.dumps({**row, "eval": result}))
with open(HERE / "results.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
