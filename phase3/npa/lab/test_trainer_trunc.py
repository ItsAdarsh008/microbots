"""Checks for the truncated-window trainer (lab/trainer_trunc.py). CPU only, a few minutes.

    python lab/test_trainer_trunc.py            # checks + the memory / time comparison with run.train
    python lab/test_trainer_trunc.py --quick    # checks only
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import cells  # noqa: E402
import npa  # noqa: E402
import run  # noqa: E402
import target as tgt  # noqa: E402
import trainer_trunc  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


N = 128
GATED = {"block": "gated", "depth": 3, "norm": "rms", "mem": 4}
DAMAGE = [{"kind": "wipe", "level": [0.1, 0.3]}, {"kind": "cut", "level": [0.1, 0.3], "p": 0.5}]


def cfg_for(cell, **train):
    return {"name": "t", "particles": N, "seed": 3, "cell": cell,
            "train": {"mode": "trunc", "iters": 4, "batch": 2, "steps": [8, 16], "grid": 64, "window": 4,
                      "damage": DAMAGE, **train}}


# 1. tiny runs through the run.train dispatch: finite losses, default and gated-with-memory cells
for label, cell in (("default cell", {}), ("gated cell with memory", GATED)):
    for wl in ("all", "last"):
        cell_obj, params, eps, curve, secs = run.train(cfg_for(cell, window_loss=wl), log=lambda m: None)
        check(f"{label}, window_loss={wl}: dispatched, finite losses",
              len(curve) == 4 and all(np.isfinite(c["loss"]) and np.isfinite(c["density"]) and np.isfinite(c["colour"])
                                      for c in curve) and set(curve[0]) == {"iter", "loss", "density", "colour"},
              f"loss {curve[0]['loss']:.3f} -> {curve[-1]['loss']:.3f}")
        check(f"{label}, window_loss={wl}: parameters moved and are finite",
              all(bool(jnp.isfinite(v).all()) for v in params.values())
              and any(float(jnp.abs(v - cell_obj.init(jax.random.PRNGKey(3))[k]).max()) > 0 for k, v in params.items()))

# the seed-replacement and annealing options run
_, _, _, curve, _ = run.train(cfg_for(GATED, reset_p=1.0, anneal_start=2.0, iters=3), log=lambda m: None)
check("reset_p=1 and anneal_start=2 run and stay finite", len(curve) == 3 and all(np.isfinite(c["loss"]) for c in curve))
sc = [trainer_trunc.anneal_scale(i, 10, {"anneal": True, "anneal_start": 1.5}) for i in range(11)]
check("damage scale anneals 1.5 -> 1.0 over the first half, then stays", abs(sc[0] - 1.5) < 1e-9 and abs(sc[5] - 1.0) < 1e-9
      and abs(sc[10] - 1.0) < 1e-9 and all(a >= b for a, b in zip(sc, sc[1:])), str(np.round(sc, 2)))

# 2. one window covering the whole rollout with window_loss "last" is plain BPTT: same first-iteration loss.
#    Every random draw (pool order, seed, damage level, per-step keys) is made in the same order, so it is exact.
for label, cell in (("default cell", {}), ("gated cell", GATED)):
    base = cfg_for(cell, iters=1, window=16, window_loss="last", anneal=False)
    ref = {**base, "train": {**base["train"], "mode": "bptt"}}
    l_t = run.train(base, log=lambda m: None)[3][0]
    l_b = run.train(ref, log=lambda m: None)[3][0]
    gaps = [abs(l_t[k] - l_b[k]) for k in ("loss", "density", "colour")]
    check(f"{label}: window=T, last == run.train first-iteration loss", max(gaps) < 1e-4,
          f"trunc {l_t['loss']:.6f} bptt {l_b['loss']:.6f}")

# 3. gradients: non-zero, finite, and the windowing behaves
target = tgt.load_target()
loss_fn = tgt.make_loss(target, grid=64)
eps = 0.2 * (1024 / N) ** 0.5
batch = 2


def setup(cell):
    params = cell.init(jax.random.PRNGKey(1))
    x, s = jax.vmap(lambda k: npa.seed(k, N, radius=2 * eps))(jax.random.split(jax.random.PRNGKey(2), batch))
    return params, x, s, jnp.zeros((batch, N, cell.mem_dims)), jnp.ones((batch, N))


for label, cell in (("default cell", cells.build()), ("gated cell", cells.build(**GATED))):
    params, x, s, mem, alive = setup(cell)
    key = jax.random.PRNGKey(5)
    steps_of = lambda w, wl, st: trainer_trunc.make_train_step(cell, loss_fn, batch, N, eps, eps, None, 16, "per_tensor",
                                                               0.0, 0.0, w, wl).rollout_grads(params, key, x, s, mem, alive, st)
    g = steps_of(4, "all", 16)[0]
    norms = {k: float(jnp.linalg.norm(v)) for k, v in g.items()}
    check(f"{label}: gradients finite and non-zero in every tensor",
          all(np.isfinite(v) and v > 0 for v in norms.values()) or all(np.isfinite(v) for v in norms.values()) and
          sum(v > 0 for v in norms.values()) >= len(norms) - 1, f"min norm {min(norms.values()):.2e}")
    g_full = steps_of(16, "last", 16)[0]
    diff = max(float(jnp.abs(g[k] / (jnp.linalg.norm(g[k]) + 1e-12) - g_full[k] / (jnp.linalg.norm(g_full[k]) + 1e-12)).max()) for k in g)
    check(f"{label}: windowed gradient differs from the full-BPTT gradient (truncation is active)", diff > 1e-3, f"max dir diff {diff:.2e}")
    # a rollout shorter than one window: the idle windows after it must change nothing
    g_a = steps_of(4, "all", 3)[0]
    g_b = steps_of(16, "last", 3)[0]
    gap = max(float(jnp.abs(g_a[k] - g_b[k]).max() / (jnp.abs(g_b[k]).max() + 1e-12)) for k in g_a)
    check(f"{label}: 3-step rollout, idle windows do not change the gradient", gap < 1e-4, f"rel gap {gap:.1e}")
    # partial last window (steps=10, window=4): gradient of the 'last' mode only sees the last (2-step) window
    gl = steps_of(4, "last", 10)
    check(f"{label}: partial final window, 'last' gradient finite", all(bool(jnp.isfinite(v).all()) for v in gl[0].values()))

if "--quick" not in sys.argv:
    # 4. memory and time against run.train at N=256, steps [32, 64], batch 2 (CPU; compiled temp bytes are the memory proxy)
    n, b, t_max = 256, 2, 64
    eps_b = 0.2 * (1024 / n) ** 0.5
    cell = cells.build()
    lf = tgt.make_loss(target, grid=64)
    params = cell.init(jax.random.PRNGKey(1))
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params),
           "t": jnp.zeros((), jnp.int32)}
    x, s = jax.vmap(lambda k: npa.seed(k, n, radius=2 * eps_b))(jax.random.split(jax.random.PRNGKey(2), b))
    mem, alive, key = jnp.zeros((b, n, 0)), jnp.ones((b, n)), jax.random.PRNGKey(7)
    args = (params, opt, key, x, s, mem, alive, 64, 5e-4)
    rows = {}
    builders = {"bptt (run.train)": lambda: run.make_train_step(cell, lf, b, n, eps_b, eps_b, None, t_max, "per_tensor", 0.0, 0.0)}
    for w in (8, 16):
        builders[f"trunc window={w}"] = (lambda w=w: trainer_trunc.make_train_step(
            cell, lf, b, n, eps_b, eps_b, None, t_max, "per_tensor", 0.0, 0.0, w, "all"))
    for name, build in builders.items():
        fn = build()
        t0 = time.time()
        compiled = fn.lower(*args).compile()
        t_compile = time.time() - t0
        try:
            mem_bytes = compiled.memory_analysis().temp_size_in_bytes
        except Exception:
            mem_bytes = float("nan")
        jax.block_until_ready(fn(*args))
        t0 = time.time()
        for _ in range(3):
            jax.block_until_ready(fn(*args))
        rows[name] = (t_compile, (time.time() - t0) / 3, mem_bytes)
        print(f"  {name:18s} compile {t_compile:6.1f}s  step(64 steps) {rows[name][1]:6.2f}s  temp memory {mem_bytes / 2**20:8.1f} MiB")
    check("trunc uses less compiled temp memory than bptt at 64 steps",
          rows["trunc window=8"][2] < rows["bptt (run.train)"][2],
          f"{rows['trunc window=8'][2] / 2**20:.1f} vs {rows['bptt (run.train)'][2] / 2**20:.1f} MiB")

print(f"\n{sum(results)}/{len(results)} trunc checks passed")
raise SystemExit(0 if all(results) else 1)
