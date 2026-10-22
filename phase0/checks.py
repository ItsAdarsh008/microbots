"""Run the Phase 0 checks from the roadmap and print PASS or FAIL for each.

    python phase0/checks.py
"""

import time

import jax
import numpy as np

import random_walk_jax as rwj
import random_walk_numpy as rwn

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# 1. Shapes: (frames, bots, x/y) in both versions.
traj_np = rwn.simulate(n_bots=100, n_steps=300, seed=0)
traj_jax = np.asarray(rwj.simulate(jax.random.PRNGKey(0), 100, 300))
check("NumPy shape is (300, 100, 2)", traj_np.shape == (300, 100, 2), str(traj_np.shape))
check("JAX shape is (300, 100, 2)", traj_jax.shape == (300, 100, 2), str(traj_jax.shape))

# 2. Walls: no bot ever leaves the box.
for label, traj in (("NumPy", traj_np), ("JAX", traj_jax)):
    inside = bool(((traj >= 0) & (traj <= rwn.BOX)).all())
    check(f"{label}: every bot stays inside the box", inside,
          f"min {traj.min():.3f}, max {traj.max():.3f}")

# 3. Reproducibility: same seed gives the same walk, a different seed a different one.
again = np.asarray(rwj.simulate(jax.random.PRNGKey(0), 100, 300))
other = np.asarray(rwj.simulate(jax.random.PRNGKey(1), 100, 300))
check("JAX: same seed gives identical results", np.array_equal(traj_jax, again))
check("JAX: different seed gives different results", not np.array_equal(traj_jax, other))

# 4. Bots move independently: almost every bot takes a different first step.
#    (A few could match by bouncing off walls, so require most to be unique.)
moves = np.diff(traj_jax[:2], axis=0)[0]          # first step of every bot
unique_moves = np.unique(moves.round(6), axis=0).shape[0]
check("JAX: bots move independently", unique_moves > len(moves) // 2,
      f"{unique_moves} of {len(moves)} first steps are different")

# 5. Speed: one step for 10,000 bots, compiled (jit) vs not compiled.
n = 10_000
key = jax.random.PRNGKey(0)
pos = jax.random.uniform(key, (n, 2), maxval=rwj.BOX)
vel = jax.numpy.zeros((n, 2))
keys = jax.random.split(key, n)
fast_step = jax.jit(rwj.step_all)


def time_it(fn, repeats=20):
    fn(pos, vel, keys)[0].block_until_ready()      # warm-up (and compile, for jit)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(pos, vel, keys)[0].block_until_ready()  # wait for JAX to actually finish
    return (time.perf_counter() - start) / repeats * 1000


slow_ms, fast_ms = time_it(rwj.step_all), time_it(fast_step)
check("jit makes a 10,000-bot step faster", fast_ms < slow_ms,
      f"no jit {slow_ms:.2f} ms, jit {fast_ms:.2f} ms, {slow_ms / fast_ms:.1f}x faster")

print(f"\n{sum(results)}/{len(results)} checks passed")
