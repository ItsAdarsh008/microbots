"""How often do bots actually turn in a trained turn run? Re-runs the shape-agnostic evaluation with a counter on the line-state bit.

    python lab/turn_rate.py ag_turn2_mlph_random
Prints turns per bot-tick over the whole evaluation (growth, control and all damage runs). Needs the run's params.npz.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import cells  # noqa: E402
import embody  # noqa: E402
import evaluate_agnostic as ea  # noqa: E402

name = sys.argv[1]
d = json.loads((HERE.parent / "runs" / name / "result.json").read_text())
cfg = d["config"]
cell = cells.build(**cfg["cell"])
z = np.load(HERE.parent / "runs" / name / "params.npz")
params = {k: jnp.asarray(z[k]) for k in z.files if k not in ("eps", "eps0")}
eps = float(z["eps"]); eps0 = float(z["eps0"]) if "eps0" in z.files else eps

count = {"turns": 0.0, "ticks": 0.0}


def _add(t, n):
    count["turns"] += float(t); count["ticks"] += float(n)


orig = embody.Embodied._stride


def patched(self, d, key, L=None):
    out = orig(self, d, key, L) if L is not None else orig(self, d, key)
    if L is not None:
        jax.debug.callback(_add, jnp.sum(out[1] != L), jnp.asarray(L.shape[0], jnp.float32))
    return out


embody.Embodied._stride = patched
ea.run(cell, params, eps, eps0, cfg["particles"])
print(f"{name}: {count['turns']:.0f} turns in {count['ticks']:.0f} bot-ticks (counting every bot, updating or not) = {count['turns'] / max(count['ticks'], 1):.5f} per bot-tick")
