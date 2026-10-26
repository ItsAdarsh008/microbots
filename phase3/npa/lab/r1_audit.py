"""R1 audit (ROBOT_RESEARCH.md): does 'returns to the shape' imply 'strong'? Repaired and undamaged swarms vs the packed start swarm.

    python lab/r1_audit.py RUN [RUN ...] --shapes donut butterfly proc:6 --out lab/r1_audit.json
Rolls each run through warm-up (32 steps), damage (none, cut, clump, noise at the evaluator's large level) and 120 recovery steps, as in
render_swarm.py, then builds contact graphs at two bond radii (touching = 1.15 body diameters; arm reach = 56 mm, tip-to-body) and computes
lab/strength.py metrics against the start swarm of the same shape (the packed reference). 1 world unit = 1160 mm.
"""
import argparse, json, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import agnostic, cells, damage, npa, strength as S

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--shapes", nargs="+", default=["donut", "butterfly", "proc:6"])
ap.add_argument("--out", default=str(HERE / "r1_audit.json")); ap.add_argument("--warm", type=int, default=32); ap.add_argument("--recover", type=int, default=120)
a = ap.parse_args()
MM = 1160.0; RADII = {"touch_34mm": 34.5, "arm_reach_56mm": 56.0}
CASES = {"none": None, "cut": ("cut", 0.30), "clump": ("clump", 0.7), "noise": ("noise", 0.5)}


def metrics(x, alive, ref_x):
    pos = np.asarray(x)[np.asarray(alive) > 0] * MM
    out = {"n": int(len(pos))}
    nn = np.linalg.norm(pos[:, None] - pos[None], axis=-1); np.fill_diagonal(nn, 1e9)
    out["median_nn_mm"] = float(np.median(nn.min(1)))
    for name, r in RADII.items():
        b = S.contact_bonds(pos, r); rb = S.contact_bonds(ref_x * MM, r)
        n = len(pos)
        out[name] = {"mean_degree": S.mean_degree(n, b), "largest_component": S.largest_component_fraction(n, b),
                     "bridge_fraction": S.critical_fraction(n, b), "Q_rel": S.q_rel(pos, b, ref_x * MM, rb),
                     "ref_mean_degree": S.mean_degree(len(ref_x), rb)}
    return out


results = {}
for run in a.runs:
    d = json.loads((HERE.parent / "runs" / run / "result.json").read_text()); cfg = d["config"]
    z = np.load(HERE.parent / "runs" / run / "params.npz")
    params = {k: jnp.asarray(z[k]) for k in z.files if k not in ("eps", "eps0")}
    eps = float(z["eps"]); eps0 = float(z["eps0"]) if "eps0" in z.files else eps
    n = cfg["particles"]; cell = cells.build(**cfg["cell"])

    def _step(x, s, mem, alive, k, cell=cell, params=params, eps=eps, eps0=eps0, n=n):
        update = jax.random.bernoulli(k, npa.UPDATE_PROB, (n, 1)).astype(x.dtype) * alive[:, None]
        x, s, mem, _ = cell.step(params, x, s, mem, update, eps, eps0, None, n, alive)
        return x, s, mem
    step = jax.jit(_step)
    for shape in a.shapes:
        mask, rgb = agnostic.get_shape(shape) if shape.startswith("proc:") else agnostic.emoji_shape(shape)
        x0, s0 = agnostic.sample_swarm(mask, rgb, n, 910_000); x0 = jnp.asarray(x0); s0 = jnp.asarray(s0); x0 = x0 - x0.mean(0)
        for case, spec in CASES.items():
            t0 = time.time(); x, s, mem, alive = x0, s0, jnp.zeros((n, cell.mem_dims)), jnp.ones(n)
            kw, kd = jax.random.split(jax.random.PRNGKey(7)); keys = jax.random.split(kw, a.warm + a.recover)
            for t in range(a.warm + a.recover):
                if t == a.warm and spec is not None:
                    x, s, mem, alive = damage.apply(spec[0], spec[1], kd, x, s, mem, alive, eps)
                x, s, mem = step(x, s, mem, alive, keys[t])
            m = metrics(np.asarray(x), np.asarray(alive), np.asarray(x0))
            results[f"{run}|{shape}|{case}"] = m
            print(run, shape, case, "n", m["n"], "nn", round(m["median_nn_mm"], 1), "mm |",
                  " | ".join(f"{k}: deg {v['mean_degree']:.2f} comp {v['largest_component']:.2f} bridge {v['bridge_fraction']:.2f} Q {v['Q_rel']:.3f}" for k, v in m.items() if isinstance(v, dict)),
                  f"({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
print("saved", a.out)
